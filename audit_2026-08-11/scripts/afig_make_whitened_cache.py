#!/usr/bin/env python3
"""Per-coordinate (slot x channel) standardized copy of the latent cache."""
import sys
import torch

SRC = "/workspace/AFIG/tokenizer_runs/v2-cross-n16-d64-s1/latents_final_original_flip.pt"
cache = torch.load(SRC, map_location="cpu", weights_only=False)
train = cache["train_latents"].float()
test = cache["test_latents"].float()
pc_mean = train.mean(dim=0)                      # [16,64]
pc_std = train.std(dim=0, unbiased=False).clamp_min(1e-6)
wt = (train - pc_mean) / pc_std
wv = (test - pc_mean) / pc_std
out = dict(cache)
out["train_latents"] = wt.half()
out["test_latents"] = wv.half()
out["whitening"] = {"pc_mean": pc_mean, "pc_std": pc_std, "type": "per_coordinate"}
stats = dict(cache["statistics"])
stats["global_mean"] = wt.flatten().mean()
stats["global_std"] = wt.flatten().std(unbiased=False)
out["statistics"] = stats
torch.save(out, sys.argv[1])
print("wrote", sys.argv[1], "gmean", float(stats["global_mean"]), "gstd", float(stats["global_std"]))
