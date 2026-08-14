#!/usr/bin/env python3
"""Uniform-average joint-flow checkpoints 12500..20000 as a cheap EMA proxy."""
import sys
import torch

paths = [f"/workspace/AFIG/prior_runs/v2-joint-cross-n16-d64-s1/checkpoint_{s:06d}.pt"
         for s in (12500, 15000, 17500, 20000)]
payloads = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
avg = {}
for key in payloads[0]["model"]:
    stacked = torch.stack([p["model"][key].float() for p in payloads])
    avg[key] = stacked.mean(0).to(payloads[0]["model"][key].dtype)
out = dict(payloads[-1])
out.pop("optimizer", None)
out["model"] = avg
out["step"] = 20000
torch.save(out, sys.argv[1])
print("wrote", sys.argv[1])
