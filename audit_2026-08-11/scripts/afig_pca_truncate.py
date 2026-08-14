#!/usr/bin/env python3
"""Decode-PSNR and FID of PCA-truncated test latents through the frozen decoder."""
import json, sys
sys.path.insert(0, "/workspace/AFIG")
import torch
import torchvision
from torchvision import transforms
from live_evaluation import InceptionFeatures, StreamingMoments, _fid, _kid
from progressive_tokenizer.checkpoints import load_tokenizer_checkpoint

OUT = "/tmp/claude-0/-workspace/d4e8f99c-1817-41a7-8b75-8aa285967645/scratchpad/afig_pca_truncate.json"
device = torch.device("cuda")
cache = torch.load("/workspace/AFIG/tokenizer_runs/v2-cross-n16-d64-s1/latents_final_original_flip.pt",
                   map_location="cpu", weights_only=False)
train = cache["train_latents"].float().reshape(100000, -1).double()
test = cache["test_latents"].float().reshape(10000, -1).double()
mu = train.mean(0)
cov = ((train - mu).T @ (train - mu)) / (train.shape[0] - 1)
evals, evecs = torch.linalg.eigh(cov)
order = torch.argsort(evals, descending=True)
V = evecs[:, order]
tok, _ = load_tokenizer_checkpoint(cache["tokenizer_checkpoint"])
tok = tok.to(device).eval().requires_grad_(False)
reference = torch.load("/workspace/AFIG/data/cifar10_test_inception.pt", map_location="cpu", weights_only=False)
extractor = InceptionFeatures(device)
testset = torchvision.datasets.CIFAR10(root="/workspace/AFIG/data", train=False, download=False,
                                       transform=transforms.ToTensor())
real_pm1 = torch.stack([testset[i][0] for i in range(10000)]).mul(2).sub(1)

@torch.no_grad()
def run(latents_flat):
    z = latents_flat.float().reshape(-1, 16, 64)
    moments = StreamingMoments(2048); kid_feats = []
    sse = 0.0; nel = 0
    for i in range(0, z.shape[0], 500):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = tok.decode(z[i:i+500].to(device)).float()
        ref = real_pm1[i:i+500].to(device)
        sse += float((x - ref).square().sum()); nel += ref.numel()
        f = extractor(x.add(1).div(2)); moments.update(f)
        if sum(t.shape[0] for t in kid_feats) < 5000:
            kid_feats.append(f[:5000 - sum(t.shape[0] for t in kid_feats)].cpu())
    gm, gc = moments.compute()
    return {"psnr_db": float(10.0 * torch.log10(torch.tensor(4.0 * nel / sse / 1.0))) if False else 10.0 * torch.log10(torch.tensor(4.0 / (sse / nel))).item(),
            "fid": _fid(reference["feature_mean"], reference["feature_covariance"], gm, gc),
            "kid": _kid(reference["kid_features"], torch.cat(kid_feats))}

results = {}
tc = test - mu
for K in (64, 128, 256, 512):
    zt = mu + (tc @ V[:, :K]) @ V[:, :K].T
    results[f"pca_{K}"] = run(zt)
    print(K, results[f"pca_{K}"], flush=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=1, sort_keys=True)
print("WROTE", OUT, flush=True)
