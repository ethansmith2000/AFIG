#!/usr/bin/env python3
"""Zero-training AR sampling-temperature sweep: scale the head's initial noise."""
import json, sys
sys.path.insert(0, "/workspace/AFIG")
import torch
from live_evaluation import InceptionFeatures, StreamingMoments, _fid, _kid
from progressive_tokenizer import AutoregressiveFlowConfig, AutoregressiveRectifiedFlow
from progressive_tokenizer.checkpoints import load_tokenizer_checkpoint

OUT = "/tmp/claude-0/-workspace/d4e8f99c-1817-41a7-8b75-8aa285967645/scratchpad/afig_ar_temperature.json"
CKPT = "/workspace/AFIG/prior_runs/v2-ar-cross-n16-d64-s1/checkpoint_final.pt"
N, BATCH, STEPS = 5000, 250, 50

device = torch.device("cuda")
payload = torch.load(CKPT, map_location="cpu", weights_only=False)
model = AutoregressiveRectifiedFlow(AutoregressiveFlowConfig(**payload["model_config"]))
model.load_state_dict(payload["model"]); model = model.to(device).eval()
mean = payload["normalization"]["mean"].float().to(device)
scale = payload["normalization"]["scale"].float().to(device)
tok, _ = load_tokenizer_checkpoint(payload["tokenizer_checkpoint"])
tok = tok.to(device).eval().requires_grad_(False)
reference = torch.load("/workspace/AFIG/data/cifar10_test_inception.pt", map_location="cpu", weights_only=False)
extractor = InceptionFeatures(device)
L = model.config.sequence_length


@torch.no_grad()
def head_sample_temp(condition, tau, gen):
    values = torch.randn(condition.shape[0], model.config.token_dim,
                         device=device, generator=gen) * tau
    dt = 1.0 / STEPS
    for index in range(STEPS):
        t = torch.full((values.shape[0],), index / STEPS, device=device, dtype=torch.float32)
        v = model.head.predict_velocity(values, t, condition)
        if index + 1 < STEPS:
            prop = values + dt * v
            nt = torch.full_like(t, (index + 1) / STEPS)
            nv = model.head.predict_velocity(prop, nt, condition)
            values = values + 0.5 * dt * (v + nv)
        else:
            values = values + dt * v
    return values


@torch.no_grad()
def run(tau, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    moments = StreamingMoments(2048); kid_feats = []
    for i in range(0, N, BATCH):
        tokens = torch.zeros(BATCH, L, model.config.token_dim, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for index in range(L):
                condition = model.trunk(tokens)[:, index]
                tokens[:, index] = head_sample_temp(condition, tau, gen)
            raw = tokens.float() * scale + mean
            img = tok.decode(raw).float().add(1).div(2)
        f = extractor(img)
        moments.update(f)
        if sum(t.shape[0] for t in kid_feats) < 5000:
            kid_feats.append(f[:5000 - sum(t.shape[0] for t in kid_feats)].cpu())
    gm, gc = moments.compute()
    return {"fid": _fid(reference["feature_mean"], reference["feature_covariance"], gm, gc),
            "kid": _kid(reference["kid_features"], torch.cat(kid_feats))}

results = {}
for tau in (0.9, 0.8):
    results[f"tau_{tau}"] = run(tau, 4321)
    print("tau", tau, results[f"tau_{tau}"], flush=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=1, sort_keys=True)
print("WROTE", OUT, flush=True)
