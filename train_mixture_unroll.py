
import math
import os
import torch.utils.checkpoint
from accelerate.logging import get_logger
from tqdm.auto import tqdm
from types import SimpleNamespace
from utils import get_1d_freqs_from_2d, get_fft,  log_validation, basic_train_setup, save_model, prepare_model_and_optimizer, resume_model
from model import FFTDecoderMixtureUnrolled
import torch
import torch.nn.functional as F


base_args = dict(
    output_dir="mixture_unroll",
    seed=123,
    resolution=32,
    train_batch_size=20,
    num_train_epochs=200,
    max_train_steps=None,

    # saving
    checkpointing_steps=400,
    checkpoints_total_limit=5,
    # resume_from_checkpoint="/home/ubuntu/seq_diffusion/unnamed/model_output/model_10400.pt",
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
    validation_steps=200,
    tracker_project_name="seqdiffusion",

    importance_weighting=False,
    variance_loss_factor=0.00025,
    # variance_loss_factor=0.0,
    phase_out_of_range_penalty=0.00025,
    sample_scale=1.0,
    sample_topk=20,

    num_gaussians=30,
    # num_gaussians=8,
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
    model = FFTDecoderMixtureUnrolled(
        query_dim=args.query_dim,
        in_channels=1,
        heads=args.heads,
        dropout=args.dropout,
        ff_mult=args.ff_mult,
        num_layers=args.num_layers,
        ctx_len=args.ctx_len,
        num_gaussians=args.num_gaussians,
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

    grad_norm = 0
    phase_out_of_range_loss = None
    variance_loss = None
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
                    # flatten
                    inputs = inputs.reshape(inputs.shape[0], -1)

                    # append a bos value basically
                    inputs = torch.cat([torch.zeros(inputs.shape[0], 1, device=accelerator.device), inputs], dim=1)

                    targets = inputs[:, 1:]
                    inputs = inputs[:, :-1]

                loss = torch.zeros(1, device=accelerator.device)  # , dtype=weight_dtype)

                means, stds, mix_ps = model(inputs)
                phase_mask = [0, 0, 0, 1, 1, 1]
                phase_mask = torch.tensor(phase_mask, device=accelerator.device).bool().repeat(
                    (means.shape[1] + 1) // 6)
                log_probs = model.gmm_log_prob(means, stds, mix_ps, targets, phase_mask)

                # punish variance, i think we want to try to have many narrow modes where possible
                if args.variance_loss_factor > 0:
                    variance_loss = F.relu(stds.norm(dim=(-1)).mean())
                    loss = loss + variance_loss * args.variance_loss_factor

                # we do a modulo thingy so that a prediction like 3.1416 is understood to be = to -3.1414 but lets penalize this
                if args.phase_out_of_range_penalty > 0:
                    phases = means[:, phase_mask]
                    phases = phases / 3.14159
                    phase_out_of_range_loss = F.relu(phases.abs() - 1).mean()
                    loss = loss + phase_out_of_range_loss * args.phase_out_of_range_penalty

                main_loss = log_probs * -1
                orig_loss = main_loss.detach()
                orig_loss = orig_loss.mean().item()

                # TODO make based off expected amplitude
                if args.importance_weighting:
                    weights = torch.linspace(1, 0.25, main_loss.shape[1] // 6).repeat_interleave(6, dim=-1)
                    weights = weights.to(loss.device).to(main_loss.dtype)
                    main_loss = main_loss * weights[None, :]

                loss = loss + main_loss.mean()

                if global_step % 100 == 0 and global_step > 0:
                    with torch.no_grad():
                        max_g = 10
                        logger.info(f"sample of means at pos 0: {means[0, 0, :max_g]}")
                        logger.info(f"sample of stds at pos 0: {stds[0, 0, :max_g]}")
                        logger.info(f"sample of mix_ps at pos 0: {F.softmax(mix_ps[0, 0, :], dim=-1)}")

                        logger.info(f"sample of means at pos 100: {means[0, 100, :max_g]}")
                        logger.info(f"sample of stds at pos 100: {stds[0, 100, :max_g]}")
                        logger.info(f"sample of mix_ps at pos 100: {F.softmax(mix_ps[0, 100, :], dim=-1)}")

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

            logs = {"loss": orig_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm,
                    }
            if phase_out_of_range_loss is not None:
                logs["phase_out_of_range_loss"] = phase_out_of_range_loss.detach().item()
            if variance_loss is not None:
                logs["variance_loss"] = variance_loss.detach().item()
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
