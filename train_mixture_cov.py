
import os
# export TRITON_CACHE_DIR=/home/ethan/.triton/cache
os.environ["TRITON_CACHE_DIR"] = "/home/ethan/.triton/cache"

import os, tempfile
os.environ.setdefault("TMPDIR", "/home/ethan/tmp")
tempfile.tempdir = "/home/ethan/tmp"


import math
import os
import torch.utils.checkpoint
from accelerate.logging import get_logger
from tqdm.auto import tqdm
from types import SimpleNamespace
from utils import get_1d_freqs_from_2d, get_fft,  log_validation, basic_train_setup, save_model, prepare_model_and_optimizer, resume_model
from model import FFTDecoderMixtureWithCovariance
import torch
import torch.nn.functional as F


base_args = dict(
    output_dir="mixture_cov",
    seed=123,
    resolution=32,
    train_batch_size=256,
    num_train_epochs=200,
    max_train_steps=None,

    # saving
    checkpointing_steps=400,
    checkpoints_total_limit=5,
    # resume_from_checkpoint="/home/ubuntu/seq_diffusion/unnamed/model_output/model_10400.pt",
    resume_from_checkpoint=None,

    # optimizer
    learning_rate=1.5e-4,
    lr_scheduler="linear",
    lr_warmup_steps=250,
    lr_num_cycles=1,
    lr_power=1.0,
    use_8bit_adam=False,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
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

    importance_weighting = True,
    variance_loss_factor = 0.00025,
    # variance_loss_factor=0.0,
    phase_out_of_range_penalty = 0.00025,
    sample_scale = 1.0,
    sample_topk = 8,
    min_variance = 0.05,
    kl_prior_variance = 0.5,
    kl_weight = 1e-3,
    mixture_entropy_weight = 1e-3,
    mixture_entropy_min = 1.5,

    input_noise_prob=0.2,
    input_noise_std=0.02,


    num_gaussians=8,
    query_dim=1024,
    heads=8,
    dropout=0.0,
    ff_mult=3,
    num_layers=12,
    ctx_len=700,

    compile=False,
    compile_mode="reduce-overhead",
    compile_fullgraph=True,
)
base_args = SimpleNamespace(**base_args)


logger = get_logger(__name__)


def main(args):
    args, accelerator, overrode_max_train_steps, train_dataloader, train_dataset, weight_dtype = basic_train_setup(args, logger)

    # Load model
    model = FFTDecoderMixtureWithCovariance(
        query_dim=args.query_dim,
        in_channels=6,
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
    import copy
    model_copy = copy.deepcopy(model)

    if args.compile:
        model = torch.compile(model, mode=args.compile_mode, fullgraph=args.compile_fullgraph)

    grad_norm = 0
    phase_out_of_range_loss = None
    variance_loss = None
    kl_reg_loss = None
    mixture_entropy_loss = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            noise_mask_fraction = None
            with accelerator.accumulate(model):
                with torch.no_grad():
                    # Convert images to latent space
                    # b, c, h, w
                    pixel_values = batch[0].to(
                        device=accelerator.device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )

                    # (b, c, h, w) x 2
                    mag, phase = get_fft(pixel_values)

                    # cat on channel dim
                    inputs = torch.cat([mag, phase], dim=1)

                    # unroll into sequence
                    inputs = get_1d_freqs_from_2d(inputs)

                    # permute to token
                    inputs = inputs.permute(0, 2, 1)

                    targets = inputs[:, 1:, :]
                    inputs = inputs[:, :-1, :]

                    inputs = inputs.to(
                        device=accelerator.device,
                        dtype=weight_dtype,
                        non_blocking=True,
                    )
                    targets = targets.to(
                        device=accelerator.device,
                        dtype=weight_dtype,
                        non_blocking=True,
                    )

                if args.input_noise_prob > 0 and args.input_noise_std > 0:
                    noise_mask = (
                        torch.rand(inputs.shape[0], inputs.shape[1], 1, device=inputs.device)
                        < args.input_noise_prob
                    )
                    noise = torch.randn_like(inputs) * args.input_noise_std
                    inputs = inputs + noise * noise_mask.to(inputs.dtype)
                    noise_mask_fraction = noise_mask.float().mean().item()

                loss = torch.zeros(1, device=accelerator.device)#, dtype=weight_dtype)

                means, covs, mix_ps = model(inputs)

                means, covs, mix_ps = model.post_process(means, covs, mix_ps)

                log_probs = model.gmm_log_prob(means, covs, mix_ps, targets)

                if args.kl_weight > 0:
                    dim_cov = covs.shape[-1]
                    covs_flat = covs.reshape(-1, dim_cov, dim_cov)
                    covs_float = covs_flat.to(torch.float32)
                    trace_cov = covs_float.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                    _, logdet_cov = torch.linalg.slogdet(covs_float)
                    kl_terms = (
                        trace_cov / args.kl_prior_variance
                        - dim_cov
                        + dim_cov * math.log(args.kl_prior_variance)
                        - logdet_cov
                    )
                    kl_reg_loss = 0.5 * kl_terms.mean()
                    loss = loss + kl_reg_loss * args.kl_weight

                # we do a modulo thingy so that a prediction like 3.1416 is understood to be = to -3.1414 but lets penalize this
                if args.phase_out_of_range_penalty > 0:
                    phases = means[:, :, :, 3:]
                    phases = phases / 3.14159
                    phase_out_of_range_loss = F.relu(phases.abs() - 1).mean()
                    loss = loss + phase_out_of_range_loss * args.phase_out_of_range_penalty

                main_loss = log_probs * -1
                orig_loss = main_loss.detach()
                orig_loss = orig_loss.mean().item()

                #TODO make based off expected amplitude
                # if args.importance_weighting:
                #     mag_channels = targets.shape[-1] // 2
                #     magnitudes = targets[:, :, :mag_channels].detach().abs().mean(dim=-1)
                #     weights = magnitudes / (magnitudes.mean(dim=1, keepdim=True) + 1e-6)
                #     main_loss = main_loss * weights

                if args.mixture_entropy_weight > 0:
                    mix_log_probs = F.log_softmax(mix_ps, dim=-1)
                    mix_probs = mix_log_probs.exp()
                    entropy = -(mix_probs * mix_log_probs).sum(dim=-1)
                    entropy_gap = torch.relu(args.mixture_entropy_min - entropy)
                    mixture_entropy_loss = entropy_gap.mean()
                    loss = loss + mixture_entropy_loss * args.mixture_entropy_weight

                loss = loss + main_loss.mean()

                # if global_step % 100 == 0 and global_step > 0:
                #     with torch.no_grad():
                #         max_g = 10
                #         logger.info(f"sample of means at pos 0: {means[0, 0, :max_g, :]}")
                #         logger.info(f"sample of stds at pos 0: {covs[0, 0, :max_g, torch.arange(6), torch.arange(6)]}")
                #         logger.info(f"sample of mix_ps at pos 0: {F.softmax(mix_ps[0, 0, :], dim=-1)}")

                #         logger.info(f"sample of means at pos 100: {means[0, 100, :max_g, :]}")
                #         logger.info(f"sample of stds at pos 100: {covs[0, 100, :max_g, torch.arange(6), torch.arange(6)]}")
                #         logger.info(f"sample of mix_ps at pos 100: {F.softmax(mix_ps[0, 100, :], dim=-1)}")

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
                        # copy weights from model to model_copy
                        model_copy.load_state_dict(model.state_dict())
                        image_logs = log_validation(
                            model_copy,
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
            if kl_reg_loss is not None:
                logs["kl_reg_loss"] = kl_reg_loss.detach().item()
            if mixture_entropy_loss is not None:
                logs["mixture_entropy_loss"] = mixture_entropy_loss.detach().item()
            if noise_mask_fraction is not None:
                logs["input_noise_fraction"] = noise_mask_fraction
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
