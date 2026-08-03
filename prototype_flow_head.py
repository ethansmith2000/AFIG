"""Controlled head-to-head: normalizing flow vs the diffusion head, same contexts.

The AR trunk from an existing trained run is frozen and used to produce context
vectors, so the only thing that varies is the head.  That isolates the question
we actually care about: on identical conditioning, how far past a *Gaussian* can
each head get?  The diffusion path reached only ~10% of the way past a Gaussian
fit (DIAGNOSIS.md section 4), and that shortfall is what makes samples decode to
mush.

Baselines evaluated on the same held-out contexts:
  * conditional Gaussian -- ridge regression x ~ W h + b with a full residual
    covariance.  This is the exact analogue of the "linear/Gaussian floor" used
    throughout the diffusion analysis, so the comparison is apples to apples.
  * marginal Gaussian -- ignores context entirely, an upper bound on NLL.

All numbers are nats per dimension; lower is better.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Tuple

import torch
import torchvision
from torchvision import transforms

from flow_decoder import ConditionalFlowDecoder, FlowDecoderConfig
from latent_autoencoder_interface import FrozenLatentAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latent_interface", required=True)
    parser.add_argument("--ar_checkpoint", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_images", type=int, default=8192)
    parser.add_argument("--test_images", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--hidden_width", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def build_contexts(
    interface: FrozenLatentAutoencoder,
    model,
    data_root: str,
    train: bool,
    count: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Frozen AR trunk hidden states and their prediction targets."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        data_root, train=train, download=False, transform=transform
    )
    count = min(count, len(dataset))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(count)),
        batch_size=256,
        shuffle=False,
        num_workers=8,
    )
    metadata = interface.position_features
    contexts, targets = [], []
    for images, _ in loader:
        latents = interface.encode_images(images.to(device))
        inputs = model.shifted_inputs(latents, metadata)
        hidden, _ = model.forward_backbone(inputs, metadata)
        contexts.append(hidden.reshape(-1, hidden.shape[-1]).float())
        targets.append(latents.reshape(-1, latents.shape[-1]).float())
    return torch.cat(contexts, dim=0), torch.cat(targets, dim=0)


def gaussian_baselines(
    train_context: torch.Tensor,
    train_target: torch.Tensor,
    test_context: torch.Tensor,
    test_target: torch.Tensor,
    ridge: float,
) -> Dict[str, float]:
    """Conditional and marginal Gaussian NLL per dimension, fit on train."""
    dim = train_target.shape[-1]
    ones = torch.ones(train_context.shape[0], 1, device=train_context.device)
    design = torch.cat([train_context, ones], dim=-1).double()
    target = train_target.double()
    gram = design.T @ design
    gram += ridge * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    weights = torch.linalg.solve(gram, design.T @ target)

    def nll(context: torch.Tensor, values: torch.Tensor, conditional: bool) -> float:
        values = values.double()
        if conditional:
            padded = torch.cat(
                [context.double(), torch.ones(context.shape[0], 1, device=context.device, dtype=torch.double)],
                dim=-1,
            )
            residual = values - padded @ weights
        else:
            residual = values - train_target.double().mean(dim=0, keepdim=True)
        # Residual covariance is fit on TRAIN residuals, then applied here.
        return residual

    train_residual = nll(train_context, train_target, True)
    covariance = (train_residual.T @ train_residual) / train_residual.shape[0]
    covariance += 1e-6 * torch.eye(dim, dtype=covariance.dtype, device=covariance.device)
    sign, logdet = torch.linalg.slogdet(covariance)
    precision = torch.linalg.inv(covariance)

    test_residual = nll(test_context, test_target, True)
    quad = (test_residual @ precision * test_residual).sum(dim=-1).mean()
    conditional_nll = 0.5 * (logdet + dim * math.log(2 * math.pi) + quad) / dim

    marginal_residual = nll(train_context, train_target, False)
    marginal_covariance = (marginal_residual.T @ marginal_residual) / marginal_residual.shape[0]
    marginal_covariance += 1e-6 * torch.eye(dim, dtype=covariance.dtype, device=covariance.device)
    _, marginal_logdet = torch.linalg.slogdet(marginal_covariance)
    marginal_precision = torch.linalg.inv(marginal_covariance)
    test_marginal = nll(test_context, test_target, False)
    marginal_quad = (test_marginal @ marginal_precision * test_marginal).sum(dim=-1).mean()
    marginal_nll = 0.5 * (marginal_logdet + dim * math.log(2 * math.pi) + marginal_quad) / dim

    return {
        "conditional_gaussian_nll_per_dim": float(conditional_nll),
        "marginal_gaussian_nll_per_dim": float(marginal_nll),
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    interface = FrozenLatentAutoencoder(
        args.checkpoint, args.latent_interface, sample_posterior=False
    ).to(device)

    from train_latent_continuous import load_latent_checkpoint as load_ar

    model, step = load_ar(args.ar_checkpoint, interface)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(f"frozen AR trunk from step {step}")

    train_context, train_target = build_contexts(
        interface, model, args.data_root, True, args.train_images, device
    )
    test_context, test_target = build_contexts(
        interface, model, args.data_root, False, args.test_images, device
    )
    print(f"train {tuple(train_context.shape)}  test {tuple(test_context.shape)}")

    baselines = gaussian_baselines(
        train_context, train_target, test_context, test_target, args.ridge
    )
    print(f"conditional Gaussian NLL/dim : {baselines['conditional_gaussian_nll_per_dim']:.4f}")
    print(f"marginal Gaussian NLL/dim    : {baselines['marginal_gaussian_nll_per_dim']:.4f}")

    flow = ConditionalFlowDecoder(
        FlowDecoderConfig(
            token_dim=train_target.shape[-1],
            context_dim=train_context.shape[-1],
            num_layers=args.num_layers,
            hidden_width=args.hidden_width,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.learning_rate, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    print(f"flow parameters: {sum(p.numel() for p in flow.parameters()) / 1e6:.2f}M")

    history = []
    for iteration in range(args.steps):
        index = torch.randint(0, train_context.shape[0], (args.batch_size,), device=device)
        output = flow.loss(train_target[index], train_context[index])
        optimizer.zero_grad(set_to_none=True)
        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if (iteration + 1) % 500 == 0:
            flow.eval()
            with torch.no_grad():
                chunks = [
                    flow.loss(test_target[i : i + 8192], test_context[i : i + 8192])["loss"]
                    for i in range(0, test_context.shape[0], 8192)
                ]
                test_nll = float(torch.stack(chunks).mean())
            flow.train()
            history.append({"step": iteration + 1, "train_nll": float(output["loss"]), "test_nll": test_nll})
            print(f"  step {iteration+1:>5}  train {float(output['loss']):.4f}  test {test_nll:.4f}")

    report = {
        "ar_step": step,
        **baselines,
        "flow_test_nll_per_dim": history[-1]["test_nll"] if history else None,
        "history": history,
    }
    conditional = baselines["conditional_gaussian_nll_per_dim"]
    marginal = baselines["marginal_gaussian_nll_per_dim"]
    if history:
        flow_nll = history[-1]["test_nll"]
        report["flow_vs_conditional_gaussian_nats"] = conditional - flow_nll
        report["context_gain_gaussian"] = marginal - conditional
        report["flow_extra_gain_beyond_gaussian"] = conditional - flow_nll
    with open(os.path.join(args.output_dir, "flow_report.json"), "w") as handle:
        json.dump(report, handle, indent=2)

    print("\n=== NLL per dimension (nats, lower is better) ===")
    print(f"  marginal Gaussian     : {marginal:.4f}")
    print(f"  conditional Gaussian  : {conditional:.4f}")
    if history:
        print(f"  conditional FLOW      : {history[-1]['test_nll']:.4f}")
        print(f"\n  context gain (Gaussian)      : {marginal - conditional:+.4f}")
        print(f"  flow gain beyond Gaussian    : {conditional - history[-1]['test_nll']:+.4f}")


if __name__ == "__main__":
    main()
