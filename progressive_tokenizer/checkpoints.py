"""Checkpoint loading helpers shared by tokenizer and prior tools."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from .model import ProgressiveTokenizer, TokenizerConfig


def load_tokenizer_checkpoint(
    path: Union[str, Path], *, map_location: Union[str, torch.device] = "cpu"
) -> tuple[ProgressiveTokenizer, dict]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if "model_config" not in payload or "model" not in payload:
        raise ValueError("not a progressive-tokenizer checkpoint")
    config_values = dict(payload["model_config"])
    # Checkpoints created before the RMSNorm-QK migration used normalized Q/K
    # vectors with one learned log-temperature per head.
    config_values.setdefault("qk_norm", "l2_temperature")
    config_values.setdefault("pool_type", "residual")
    config_values.setdefault("cross_attention_bias", True)
    config = TokenizerConfig(**config_values)
    model = ProgressiveTokenizer(config)
    model.load_state_dict(payload["model"])
    return model, payload
