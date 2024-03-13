
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from moviepy.editor import ImageSequenceClip
import wandb
import gc
import logging
import math
import os
from pathlib import Path
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
import diffusers
import torch
import torchvision
import torchvision.transforms as transforms
import shutil
from diffusers.optimization import get_scheduler

def quantize(x, centroids):
    d = abs(x[None, :] - centroids[:, None])
    x = torch.argmin(d, 0)
    return x


def basic_train_setup(args, logger):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(32),
         transforms.CenterCrop(32),
         ]
    )

    # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=transform)

    # train_dataset = torchvision.datasets.CelebA(root='./data', split='train',
    #                                         download=True, transform=transform)

    train_dataset = torchvision.datasets.Flowers102(root='./data', split='train',
                                                    download=True, transform=transform)

    gc.collect()
    torch.cuda.empty_cache()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    return args, accelerator, overrode_max_train_steps, train_dataloader, train_dataset, weight_dtype


def prepare_model_and_optimizer(model, args, accelerator, train_dataloader):

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    optimizer = optimizer_class(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    return model, optimizer, lr_scheduler, train_dataloader


def save_model(model, args, accelerator, logger, global_step):
    if global_step % args.checkpointing_steps == 0:
        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
        if args.checkpoints_total_limit is not None:
            checkpoints = os.listdir(args.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
            if len(checkpoints) >= args.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).state_dict()
            torch.save(state_dict, os.path.join(args.output_dir, f"model_{global_step}.pt"))
            del state_dict


def resume_model(model, args, accelerator, num_update_steps_per_epoch):
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("model")]
        dirs = sorted(dirs, key=lambda x: int(x.split("_")[-1].replace(".pt", "")))
        path = dirs[-1] if len(dirs) > 0 else None

    accelerator.print(f"Resuming from checkpoint {path}")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, path)))
    global_step = int(path.split("_")[-1].replace(".pt", ""))

    initial_global_step = global_step
    first_epoch = global_step // num_update_steps_per_epoch

    return model, global_step, initial_global_step, first_epoch

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema_weights = {name: param.data.clone() for name, param in model.named_parameters()}
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                self.ema_weights[name] = (1.0 - self.decay) * self.ema_weights[name] + self.decay * param.data

    def apply_ema_to_model(self, model):
        for name, param in model.named_parameters():
            param.data = self.ema_weights[name].clone()


def angular_distance(angle1, angle2):
    # Calculate the difference
    diff = angle2 - angle1

    # Adjust the difference to be in the range [-π, π]
    diff_adjusted = torch.atan2(torch.sin(diff), torch.cos(diff))

    return diff_adjusted


def angular_loss(angle1, angle2):
    distance = angular_distance(angle1, angle2)

    return torch.square(distance)


def get_1d_freqs_from_2d(array, debug=False):
    b, c, h, w = array.shape
    # split along w, inclusive
    is_even_h = h % 2 == 0
    is_even_w = w % 2 == 0

    split_point = w // 2 + 1 if is_even_w else w // 2
    array = array[:, :, :, :split_point]
    mapping = torch.empty_like(array)
    total_size = array.shape[2] * array.shape[3]
    unfolded = torch.empty((b, c, total_size))

    def log_point(unfold, arr, coord, pt):
        unfold[:, :, pt] = arr[:, :, coord[0], coord[1]]
        pt = pt + 1
        return unfold, pt

    # center point
    ptr = 0
    center_w = w // 2 if is_even_w else w // 2 + 1
    center_h = h // 2 if is_even_h else h // 2 + 1

    cur_coord = center_h, center_w

    print(array.shape) if debug else None

    unfolded, ptr = log_point(unfolded, array, cur_coord, ptr)
    print(cur_coord, ptr) if debug else None

    # make the rounds
    for idx in range(1, w // 2):
        # bottom point
        cur_coord = center_h + idx, center_w
        print(cur_coord, ptr) if debug else None
        unfolded, ptr = log_point(unfolded, array, cur_coord, ptr)

        # move left
        for i in range(idx):
            cur_coord = cur_coord[0], cur_coord[1] - 1
            print(cur_coord, ptr) if debug else None
            unfolded, ptr = log_point(unfolded, array, cur_coord, ptr)

        # move up
        for i in range(idx * 2):
            cur_coord = cur_coord[0] - 1, cur_coord[1]
            print(cur_coord, ptr) if debug else None
            unfolded, ptr = log_point(unfolded, array, cur_coord, ptr)

        # move right
        for i in range(idx):
            cur_coord = cur_coord[0], cur_coord[1] + 1
            print(cur_coord, ptr) if debug else None
            unfolded, ptr = log_point(unfolded, array, cur_coord, ptr)

    if is_even_h:
        # start from bottom left corner and go all the way up then all the way right
        cur_coord = h - 1, 0
        for i in range(h - 1):
            unfolded, ptr = log_point(unfolded, array, cur_coord, ptr)
            cur_coord = cur_coord[0] - 1, cur_coord[1]
            print(cur_coord, ptr) if debug else None

        for i in range(w // 2 + 1):
            unfolded, ptr = log_point(unfolded, array, cur_coord, ptr)
            cur_coord = cur_coord[0], cur_coord[1] + 1
            print(cur_coord, ptr) if debug else None

    return unfolded


def do_paint(array, break_idx=-1):
    b, h, w = array.shape
    # split along w, inclusive
    is_even_h = h % 2 == 0
    is_even_w = w % 2 == 0

    split_point = w // 2 + 1 if is_even_w else w // 2
    array = array[:, :, :split_point]
    total_size = array.shape[1] * array.shape[2]

    def paint(arr, coord, pt):
        arr[:, coord[0], coord[1]] = pt
        pt = pt + 1
        return arr, pt

    # center point
    ptr = 0
    center_w = w // 2 if is_even_w else w // 2 + 1
    center_h = h // 2 if is_even_h else h // 2 + 1

    cur_coord = center_h, center_w

    # print(array.shape)
    array, ptr = paint(array, cur_coord, ptr)
    # print(cur_coord, ptr)

    # make the rounds
    for idx in range(1, w // 2):
        # bottom point
        cur_coord = center_h + idx, center_w
        # print(cur_coord, ptr)
        array, ptr = paint(array, cur_coord, ptr)

        # move left
        for i in range(idx):
            cur_coord = cur_coord[0], cur_coord[1] - 1
            # print(cur_coord, ptr)
            array, ptr = paint(array, cur_coord, ptr)

        # move up
        for i in range(idx * 2):
            cur_coord = cur_coord[0] - 1, cur_coord[1]
            # print(cur_coord, ptr)
            array, ptr = paint(array, cur_coord, ptr)

        # move right
        for i in range(idx):
            cur_coord = cur_coord[0], cur_coord[1] + 1
            # print(cur_coord, ptr)
            array, ptr = paint(array, cur_coord, ptr)

        if idx == break_idx:
            break

    if is_even_h:
        # start from bottom left corner and go all the way up then all the way right
        cur_coord = h - 1, 0
        for i in range(h - 1):
            array, ptr = paint(array, cur_coord, ptr)
            cur_coord = cur_coord[0] - 1, cur_coord[1]
            print(cur_coord, ptr)

        for i in range(w // 2 + 1):
            array, ptr = paint(array, cur_coord, ptr)
            cur_coord = cur_coord[0], cur_coord[1] + 1
            print(cur_coord, ptr)

    plt.imshow(array[0])

    return array


def get_2d_freqs_from_1d(seq, orig_h, orig_w, debug=False, limit=None):
    b, s, c = seq.shape

    is_even_h = orig_h % 2 == 0
    is_even_w = orig_w % 2 == 0

    split_point = orig_w // 2 + 1 if is_even_w else orig_w // 2

    array = torch.zeros((b, c, orig_h, orig_w)).to(seq.device)

    def log_point(unfold, arr, coord, pt):
        arr[:, :, coord[0], coord[1]] = unfold[:, pt, :]
        pt = pt + 1
        return arr, pt

    # center point
    ptr = 0
    center_w = orig_w // 2 if is_even_w else orig_w // 2 + 1
    center_h = orig_h // 2 if is_even_h else orig_h // 2 + 1

    is_finished = False

    print(center_h, center_w)

    cur_coord = center_h, center_w

    array, ptr = log_point(seq, array, cur_coord, ptr)
    print(cur_coord, ptr) if debug else None

    if limit is not None and ptr >= limit:
        is_finished = True
        return array

    # make the rounds
    for idx in range(1, orig_w // 2):

        # bottom point
        if not is_finished:
            cur_coord = center_h + idx, center_w
            array, ptr = log_point(seq, array, cur_coord, ptr)
            print(cur_coord, ptr) if debug else None
        else:
            is_finished = True
            break

        # move left
        for i in range(idx):
            if not is_finished:
                cur_coord = cur_coord[0], cur_coord[1] - 1
                array, ptr = log_point(seq, array, cur_coord, ptr)
                print(cur_coord, ptr) if debug else None
                if limit is not None and ptr >= limit:
                    is_finished = True
            else:
                break

        # move up
        for i in range(idx * 2):
            if not is_finished:
                cur_coord = cur_coord[0] - 1, cur_coord[1]
                array, ptr = log_point(seq, array, cur_coord, ptr)
                print(cur_coord, ptr) if debug else None
                if limit is not None and ptr >= limit:
                    is_finished = True
            else:
                break

        # move right
        for i in range(idx):
            if not is_finished:
                cur_coord = cur_coord[0], cur_coord[1] + 1
                array, ptr = log_point(seq, array, cur_coord, ptr)
                print(cur_coord, ptr) if debug else None
                if limit is not None and ptr >= limit:
                    is_finished = True
            else:
                break

        if is_finished:
            break

    if is_even_h:
        # start from bottom left corner and go all the way up then all the way right
        cur_coord = orig_h - 1, 0
        for i in range(orig_h - 1):
            if not is_finished:
                array, ptr = log_point(seq, array, cur_coord, ptr)
                print(cur_coord, ptr) if debug else None
                cur_coord = cur_coord[0] - 1, cur_coord[1]
                if limit is not None and ptr >= limit:
                    is_finished = True
            else:
                break

        for i in range(orig_w // 2 + 1):
            if not is_finished:
                array, ptr = log_point(seq, array, cur_coord, ptr)
                print(cur_coord, ptr) if debug else None
                cur_coord = cur_coord[0], cur_coord[1] + 1
                if limit is not None and ptr >= limit:
                    is_finished = True
            else:
                break

    # # now we'll need to mirror what we have
    array[:, :, :, split_point:] = torch.flip(torch.flip(array[:, :, :, :split_point - 2], dims=(3,)), dims=(2,))

    # phase needs to be inverted
    array[:, 3:, :, split_point:] = -array[:, 3:, :, split_point:]

    return array


def get_fft(array, norm=None):
    # gets frequency and phase information from image
    fft = torch.fft.fft2(array, norm=norm)
    fft = torch.fft.fftshift(fft)

    mag = torch.abs(fft)
    mag = torch.log(mag + 1e-9)
    phase = torch.angle(fft)

    return mag, phase


def inverse_fft(mag, phase):
    fft = torch.exp(mag) * torch.exp(1j * phase)
    fft = torch.fft.ifftshift(fft)
    array = torch.fft.ifft2(fft)
    return array


###############

def get_one_2d_freq_from_1d(seq, orig_h, orig_w, debug=False, index=None):
    b, s, c = seq.shape

    is_even_h = orig_h % 2 == 0
    is_even_w = orig_w % 2 == 0

    split_point = orig_w // 2 + 1 if is_even_w else orig_w // 2

    array = torch.zeros((b, c, orig_h, orig_w)).to(seq.device)

    def log_point(unfold, arr, coord, pt):
        arr[:, :, coord[0], coord[1]] = unfold[:, pt, :]
        return arr

    # center point
    ptr = 0
    center_w = orig_w // 2 if is_even_w else orig_w // 2 + 1
    center_h = orig_h // 2 if is_even_h else orig_h // 2 + 1

    cur_coord = center_h, center_w

    if index == ptr:
        array = log_point(seq, array, cur_coord, ptr)
        print(cur_coord, ptr) if debug else None
        return array
    else:
        ptr = ptr + 1

    # make the rounds
    for idx in range(1, orig_w // 2):

        # bottom point
        if index == ptr:
            cur_coord = center_h + idx, center_w
            array = log_point(seq, array, cur_coord, ptr)
            print(cur_coord, ptr) if debug else None
            return array
        else:
            ptr = ptr + 1

        # move left
        for i in range(idx):
            if index == ptr:
                cur_coord = cur_coord[0], cur_coord[1] - 1
                array = log_point(seq, array, cur_coord, ptr)
                print(cur_coord, ptr) if debug else None
                return array
            else:
                ptr = ptr + 1

        # move up
        for i in range(idx * 2):
            if index == ptr:
                cur_coord = cur_coord[0] - 1, cur_coord[1]
                array = log_point(seq, array, cur_coord, ptr)
                print(cur_coord, ptr) if debug else None
                return array
            else:
                ptr = ptr + 1

        # move right
        for i in range(idx):
            if index == ptr:
                cur_coord = cur_coord[0], cur_coord[1] + 1
                array = log_point(seq, array, cur_coord, ptr)
                print(cur_coord, ptr) if debug else None
                return array
            else:
                ptr = ptr + 1

    if is_even_h:
        # start from bottom left corner and go all the way up then all the way right
        cur_coord = orig_h - 1, 0
        for i in range(orig_h - 1):
            if index == ptr:
                array = log_point(seq, array, cur_coord, ptr)
                print(cur_coord, ptr) if debug else None
                cur_coord = cur_coord[0] - 1, cur_coord[1]
                return array
            else:
                ptr = ptr + 1

        for i in range(orig_w // 2 + 1):
            if index == ptr:
                array = log_point(seq, array, cur_coord, ptr)
                print(cur_coord, ptr) if debug else None
                cur_coord = cur_coord[0], cur_coord[1] + 1
                return array
            else:
                ptr = ptr + 1

    # # now we'll need to mirror what we have
    array[:, :, :, split_point:] = torch.flip(torch.flip(array[:, :, :, :split_point - 2], dims=(3,)), dims=(2,))

    # phase needs to be inverted
    array[:, 3:, :, split_point:] = -array[:, 3:, :, split_point:]

    return array


def visualize_reconstruction(img, one_freq_dim=0, h=64, w=64):
    img = img.resize((h, w), Image.BICUBIC)
    img = torch.from_numpy(np.array(img)) / 255
    fft = torch.fft.fftshift(torch.fft.fft2(img.permute(2, 0, 1)))
    mag = torch.log(torch.abs(fft))
    phase = torch.angle(fft)

    stacked = torch.cat([mag, phase], dim=0)
    sequence = get_1d_freqs_from_2d(stacked[None, :, :, :])
    sequence = sequence.permute(0, 2, 1)

    single_freqs = []
    progress_frames = []
    for i in tqdm(range(sequence.shape[1])):
        arr = get_2d_freqs_from_1d(sequence, h, w, limit=i)
        new_mag, new_phase = arr[:, :3], arr[:, 3:]
        reconstructed = inverse_fft(new_mag, new_phase)
        reconstructed = torch.abs(reconstructed)
        reconstructed[:, :, 0, 0] = reconstructed[:, :, 0, 1]  # just a fix for visual purposes

        one_freq = get_one_2d_freq_from_1d(sequence, h, w, index=i)
        one_freq_mag, one_freq_phase = one_freq[:, :3], one_freq[:, 3:]
        reconstructed_one_freq = inverse_fft(one_freq_mag, one_freq_phase)
        reconstructed_one_freq = torch.abs(reconstructed_one_freq)
        reconstructed_one_freq[:, :, 0, 0] = reconstructed_one_freq[:, :, 0, 1]  # just a fix for visual purposes

        single_freq = reconstructed_one_freq[0].permute(1, 2, 0)[:, :, one_freq_dim] * 255
        single_freq = np.clip(single_freq, 0, 255)

        reconstructed = reconstructed[0].permute(1, 2, 0) * 255
        reconstructed = np.clip(reconstructed, 0, 255)

        single_freqs.append(single_freq.numpy())
        progress_frames.append(reconstructed.numpy())

    progress_frames = [Image.fromarray(x.astype(np.uint8)).resize((256, 256), Image.NEAREST) for x in progress_frames]
    single_freqs = [Image.fromarray(x.astype(np.uint8)).resize((256, 256), Image.NEAREST) for x in single_freqs]

    return single_freqs, progress_frames


def create_video_moviepy(frames, output_path, filename="movie", fps=25):
    if isinstance(frames[0], PIL.Image.Image):
        frames = [np.array(frame.convert("RGB")) for frame in frames]
    clip = ImageSequenceClip(frames, fps=fps)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    clip.write_videofile(f"{output_path}/{filename}.mp4", codec="libx264")

    # Close the clip to release resources
    clip.close()



@torch.no_grad()
def log_validation(model, args, accelerator, logger, step, batch_size=4):
    logger.info("Running validation... ")
    model.eval()

    image_logs = []

    whole_sequence, images = model.gen_sample(batch_size, sample_topk=args.sample_topk)

    image_logs.append(
        {"images": images, }
    )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]

                formatted_images = []

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images("1", formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]

                for image in images:
                    image = wandb.Image(image, caption="1")
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        model.train()

        return image_logs