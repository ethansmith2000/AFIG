"""Conditional normalizing-flow head for per-token latent prediction.

Drop-in alternative to the per-token diffusion decoder on the AR path.  The
motivation is that every obstacle this project has hit -- loss weighting being
unable to reproduce SNR dynamics, whitening and conditioning order failing to
decouple, normalization interacting with the noise schedule, MSE imposing a
metric the data's geometry may not want -- is a property of the *diffusion
formulation*, not of the data.  Exact likelihood removes them at once:

  * no timesteps, no forward process, so no SNR schedule to mis-specify
  * density estimation picks no metric, unlike MSE
  * change of variables handles reparameterization exactly, so a change of
    normalization shifts the log-likelihood by a tractable log-det rather than
    silently altering the objective

Directions (they are the same flow used opposite ways):
  training  data -> base:  z = f(x | c),  maximize log N(z;0,I) + log|det df/dx|
  sampling  base -> data:  z ~ N(0,I),    x = f^-1(z | c)

Coordinates stay Cartesian.  A polar parameterization would need a circular base
distribution for the angle, reintroducing the manifold complication this is meant
to avoid.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class FlowDecoderConfig:
    token_dim: int = 64
    context_dim: int = 768
    num_layers: int = 8
    hidden_width: int = 512
    hidden_depth: int = 2
    # Per-layer bound on |log scale|.  With num_layers stacked, the worst-case
    # Jacobian range is exp(scale_cap * num_layers), so 3.0 across 8 layers would
    # permit exp(24) ~ 3e10.  2.0 still allows ample expressiveness for
    # unit-variance latents while keeping that bound sane, and zero-init means
    # scales only grow as the data requires.
    scale_cap: float = 2.0

    def fingerprint(self) -> Dict[str, Any]:
        return asdict(self)


def _mlp(in_dim: int, hidden: int, depth: int, out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(max(depth - 1, 0)):
        layers += [nn.Linear(hidden, hidden), nn.SiLU()]
    final = nn.Linear(hidden, out_dim)
    # Zero-init the last layer so every coupling starts as the identity map and
    # the flow begins as an exact standard normal.
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


class ConditionalCoupling(nn.Module):
    """Affine coupling: the masked half is passed through and conditions the rest.

    y = x*m + (1-m) * (x * exp(s) + t),  with (s, t) = net([x*m, context]).
    log|det| = sum over the transformed half of s.  The scale is tanh-bounded so
    a single layer cannot blow up the Jacobian.
    """

    def __init__(self, config: FlowDecoderConfig, mask: torch.Tensor):
        super().__init__()
        self.config = config
        self.register_buffer("mask", mask, persistent=True)
        self.net = _mlp(
            config.token_dim + config.context_dim,
            config.hidden_width,
            config.hidden_depth,
            2 * config.token_dim,
        )

    def _params(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.mask.to(x.dtype)
        raw = self.net(torch.cat([x * mask, context], dim=-1))
        raw_scale, shift = raw.chunk(2, dim=-1)
        scale = self.config.scale_cap * torch.tanh(raw_scale)
        return scale * (1.0 - mask), shift * (1.0 - mask)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self._params(x, context)
        y = x * self.mask.to(x.dtype) + (1.0 - self.mask.to(x.dtype)) * (
            x * torch.exp(scale) + shift
        )
        return y, scale.sum(dim=-1)

    def inverse(self, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # The masked half is untouched by forward(), so the same net input is
        # recoverable from y directly -- this is what makes coupling invertible.
        scale, shift = self._params(y, context)
        mask = self.mask.to(y.dtype)
        return y * mask + (1.0 - mask) * ((y - shift) * torch.exp(-scale))


class ConditionalFlowDecoder(nn.Module):
    """Stack of conditional couplings with fixed permutations between them."""

    def __init__(self, config: Optional[FlowDecoderConfig] = None):
        super().__init__()
        self.config = config or FlowDecoderConfig()
        cfg = self.config
        if cfg.token_dim % 2:
            raise ValueError("token_dim must be even for alternating masks")

        base = torch.zeros(cfg.token_dim)
        base[: cfg.token_dim // 2] = 1.0
        self.couplings = nn.ModuleList()
        generator = torch.Generator().manual_seed(0)
        permutations = []
        for index in range(cfg.num_layers):
            mask = base if index % 2 == 0 else 1.0 - base
            self.couplings.append(ConditionalCoupling(cfg, mask.clone()))
            permutations.append(torch.randperm(cfg.token_dim, generator=generator))
        self.register_buffer("permutations", torch.stack(permutations), persistent=True)
        inverse = torch.zeros_like(self.permutations)
        for index, permutation in enumerate(permutations):
            inverse[index, permutation] = torch.arange(cfg.token_dim)
        self.register_buffer("inverse_permutations", inverse, persistent=True)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """data -> base.  Returns z and the accumulated log|det df/dx|."""
        log_det = torch.zeros(x.shape[:-1], device=x.device, dtype=torch.float32)
        for index, coupling in enumerate(self.couplings):
            x, step = coupling(x, context)
            log_det = log_det + step.float()
            x = x[..., self.permutations[index]]
        return x, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """base -> data."""
        for index in reversed(range(len(self.couplings))):
            z = z[..., self.inverse_permutations[index]]
            z = self.couplings[index].inverse(z, context)
        return z

    def log_prob(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Exact conditional log-density of x given context."""
        z, log_det = self.forward(x, context)
        dim = x.shape[-1]
        base = -0.5 * (z.float() ** 2).sum(dim=-1) - 0.5 * dim * math.log(2.0 * math.pi)
        return base + log_det

    def loss(self, x: torch.Tensor, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        log_prob = self.log_prob(x, context)
        dim = x.shape[-1]
        return {
            # Negative log-likelihood in nats per dimension, so the number is
            # comparable across token widths.
            "loss": -log_prob.mean() / dim,
            "log_prob": log_prob.mean().detach(),
            "bits_per_dim": (-log_prob.mean() / dim / math.log(2.0)).detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        shape = context.shape[:-1] + (self.config.token_dim,)
        z = torch.randn(
            shape, device=context.device, dtype=torch.float32, generator=generator
        )
        return self.inverse(z * temperature, context)
