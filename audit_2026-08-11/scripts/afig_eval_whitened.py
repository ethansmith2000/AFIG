#!/usr/bin/env python3
"""Evaluate a joint-flow checkpoint trained on the per-coordinate whitened cache."""
import json, sys
sys.path.insert(0, "/workspace/AFIG")
import torch
from live_evaluation import InceptionFeatures, StreamingMoments, _fid, _kid
from progressive_tokenizer import JointFlowConfig, JointRectifiedFlow
from progressive_tokenizer.checkpoints import load_tokenizer_checkpoint

ckpt_path, cache_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
device = torch.device("cuda")
payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = JointRectifiedFlow(JointFlowConfig(**payload["model_config"]))
model.load_state_dict(payload["model"]); model = model.to(device).eval()
mean = payload["normalization"]["mean"].float().to(device)
scale = payload["normalization"]["scale"].float().to(device)
cache = torch.load(cache_path, map_location="cpu", weights_only=False)
pc_mean = cache["whitening"]["pc_mean"].to(device)
pc_std = cache["whitening"]["pc_std"].to(device)
tok, _ = load_tokenizer_checkpoint(payload["tokenizer_checkpoint"])
tok = tok.to(device).eval().requires_grad_(False)
reference = torch.load("/workspace/AFIG/data/cifar10_test_inception.pt", map_location="cpu", weights_only=False)
extractor = InceptionFeatures(device)

moments = StreamingMoments(2048); kid_feats = []
gen = torch.Generator(device=device).manual_seed(54321)
N, B = 5000, 256
done = 0
with torch.no_grad():
    while done < N:
        cur = min(B, N - done)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            std_lat = model.sample(cur, steps=50, solver="heun", generator=gen)
            whitened = std_lat.float() * scale + mean
            raw = whitened * pc_std + pc_mean
            img = tok.decode(raw).float().add(1).div(2)
        f = extractor(img)
        moments.update(f)
        if sum(t.shape[0] for t in kid_feats) < 5000:
            kid_feats.append(f[:5000 - sum(t.shape[0] for t in kid_feats)].cpu())
        done += cur
gm, gc = moments.compute()
res = {"step": int(payload["step"]),
       "fid": _fid(reference["feature_mean"], reference["feature_covariance"], gm, gc),
       "kid": _kid(reference["kid_features"], torch.cat(kid_feats))}
print(json.dumps(res), flush=True)
with open(out_path, "a") as f:
    f.write(json.dumps(res) + "\n")
