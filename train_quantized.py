
import math
import os
import torch.utils.checkpoint
from accelerate.logging import get_logger
from tqdm.auto import tqdm
from types import SimpleNamespace
from utils import get_1d_freqs_from_2d, get_fft,  log_validation, basic_train_setup, save_model, prepare_model_and_optimizer, resume_model, quantize
from model import FFTDecoderQuantized
import torch
import torch.nn.functional as F

base_args = dict(
    output_dir="quantized",
    seed=123,
    resolution=32,
    train_batch_size=20,
    num_train_epochs=200,
    max_train_steps=None,

    # saving
    checkpointing_steps=400,
    checkpoints_total_limit=5,
    # resume_from_checkpoint="/home/ubuntu/seq_diffusion/unnamed/model_output/model_400.pt",
    resume_from_checkpoint=None,

    # optimizer
    learning_rate=8e-5,
    lr_scheduler="linear",
    lr_warmup_steps=250,
    lr_num_cycles=1,
    lr_power=1.0,
    use_8bit_adam=False,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_weight_decay=1e-2,
    adam_epsilon=1e-08,
    max_grad_norm=1.0,

    dataloader_num_workers=12,
    logging_dir="logs",
    allow_tf32=True,
    report_to="wandb",
    mixed_precision="bf16",
    set_grads_to_none=True,
    max_train_samples=None,
    num_validation_images=4,
    validation_steps=100,
    tracker_project_name="seqdiffusion",

    importance_weighting=True,
    vocab_size=8192,

    normality_loss_weight=0.0,
    normality_std=1.5,

    query_dim=1024,
    heads=16,
    dropout=0.0,
    ff_mult=3,
    num_layers=12,
    ctx_len=4000,

    npz_path="/home/ubuntu/seq_diffusion/imagenet"
)
base_args = SimpleNamespace(**base_args)


logger = get_logger(__name__)


def main(args):
    args, accelerator, overrode_max_train_steps, train_dataloader, train_dataset, weight_dtype = basic_train_setup(args, logger)

    # Load model
    model = FFTDecoderQuantized(
        query_dim=args.query_dim,
        in_channels=1,
        heads=args.heads,
        dropout=args.dropout,
        ff_mult=args.ff_mult,
        num_layers=args.num_layers,
        ctx_len=args.ctx_len,
        vocab_size=args.vocab_size
    )

    model, optimizer, lr_scheduler, train_dataloader = prepare_model_and_optimizer(model, args, accelerator, train_dataloader)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        model, global_step, initial_global_step, first_epoch = resume_model(model, args, accelerator, num_update_steps_per_epoch)
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    if args.normality_loss_weight > 0:
        reference_dist = torch.distributions.normal.Normal(0, args.normality_std)
        vals = model.vocab.data.clone().detach()
        reference_dist = reference_dist.log_prob(vals)
        reference_dist = reference_dist.to(accelerator.device).float()
        reference_dist = F.softmax(reference_dist, dim=-1)
        reference_dist = reference_dist[2:]

    grad_norm = 0
    normality_loss = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                with torch.no_grad():
                    # Convert images to latent space
                    # b, c, h, w
                    pixel_values = batch[0].cuda().float()

                    # (b, c, h, w) x 2
                    mag, phase = get_fft(pixel_values)

                    # cat on channel dim
                    inputs = torch.cat([mag, phase], dim=1)

                    # unroll into sequence
                    inputs = get_1d_freqs_from_2d(inputs).cuda().float()

                    # permute to token
                    inputs = inputs.permute(0, 2, 1)
                    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1] * inputs.shape[2])

                    # quantize
                    new_inputs = []
                    for i in range(inputs.shape[0]):
                        new_inputs.append(quantize(inputs[i], model.vocab))
                    inputs = torch.stack(new_inputs, dim=0)

                    targets = torch.cat([inputs, torch.tensor([1]).long().repeat(inputs.shape[0], 1).to(inputs.device)],
                                        dim=-1)
                    inputs = torch.cat([torch.tensor([0]).long().repeat(inputs.shape[0], 1).to(inputs.device), inputs],
                                       dim=1)

                preds = model(inputs)

                loss = F.cross_entropy(preds.reshape(-1, preds.shape[-1]).float(), targets.reshape(-1), reduction='none')

                # regularization loss, we know empirically the amplitudes are normal about 0 with std about 1.4
                if args.normality_loss_weight > 0:
                    # ignore bos/eos terms
                    preds = preds[:, :, :-1]
                    preds = F.log_softmax(preds.float(), dim=-1)
                    normality_loss = F.kl_div(preds.reshape(-1, preds.shape[-1]).float(),
                                              reference_dist[None, :].expand(preds.shape[0] * preds.shape[1], -1),
                                              reduction='batchmean').mean()
                    loss = loss + args.normality_loss_weight * normality_loss

                loss = loss.mean()

                # if global_step % 100 == 0 and global_step > 0:
                #     # choose random example from losses
                #     idx = np.random.randint(0, mag_loss.shape[0])
                #     loss_example = [mag_loss[idx].clone().detach().cpu(), phase_loss[idx].clone().detach().cpu()]
                #     torch.save(loss_example, f"loss_example_{global_step}.pt")

                # if args.importance_weighting:
                #     weights = torch.linspace(1, 0.1, mag_loss.shape[1])
                #     weights = weights.to(mag_loss.device).to(mag_loss.dtype)
                #     mag_loss = mag_loss * weights[None, :, None]
                #     phase_loss = phase_loss * weights[None, :, None]
                #
                # loss = mag_loss.mean() + phase_loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    save_model(model, args, accelerator, logger, global_step)

                    if global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            model,
                            args,
                            accelerator,
                            logger,
                            global_step,
                            batch_size=args.num_validation_images,
                        )

            logs = {"loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm,
                    # "mag_loss": mag_loss.mean().detach().item(),
                    # "phase_loss": phase_loss.mean().detach().item(),
                    }
            if normality_loss is not None:
                logs["normality_loss"] = normality_loss.mean().detach().item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        state_dict = accelerator.unwrap_model(model).state_dict()
        torch.save(state_dict, os.path.join(args.output_dir, f"model_{global_step}.pt"))
        del state_dict

    accelerator.end_training()


if __name__ == "__main__":
    main(base_args)
