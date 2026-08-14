#!/usr/bin/env python3
"""Tokenizer-side diagnostics: reconstruction FID, Gaussian-fit baseline FID,
decoder noise sensitivity, and latent geometry. Read-only w.r.t. the repo."""
import json, sys, os
sys.path.insert(0, "/workspace/AFIG")
import torch
import torchvision
from torchvision import transforms
from live_evaluation import InceptionFeatures, StreamingMoments, _fid, _kid
from progressive_tokenizer.checkpoints import load_tokenizer_checkpoint

OUT = "/tmp/claude-0/-workspace/d4e8f99c-1817-41a7-8b75-8aa285967645/scratchpad/afig_diag_tokenizer.json"
CACHE = "/workspace/AFIG/tokenizer_runs/v2-cross-n16-d64-s1/latents_final_original_flip.pt"
REF = "/workspace/AFIG/data/cifar10_test_inception.pt"

device = torch.device("cuda")
cache = torch.load(CACHE, map_location="cpu", weights_only=False)
mean = float(cache["statistics"]["global_mean"]); std = float(cache["statistics"]["global_std"])
test = cache["test_latents"].float()          # [10000,16,64] raw units
train = cache["train_latents"].float()        # [100000,16,64]
tok, tok_payload = load_tokenizer_checkpoint(cache["tokenizer_checkpoint"])
tok = tok.to(device).eval().requires_grad_(False)
reference = torch.load(REF, map_location="cpu", weights_only=False)
extractor = InceptionFeatures(device)

testset = torchvision.datasets.CIFAR10(root="/workspace/AFIG/data", train=False, download=False,
                                       transform=transforms.ToTensor())
real01 = torch.stack([testset[i][0] for i in range(len(testset))])  # [10000,3,32,32] in [0,1]
real_pm1 = real01.mul(2).sub(1)

@torch.no_grad()
def decode_stats(latents_raw, n_fid=None, psnr_vs=None, batch=500):
    """Decode raw-unit latents; return FID/KID vs reference and PSNR vs psnr_vs."""
    moments = StreamingMoments(2048); kid_feats = []
    sse = 0.0; nel = 0; n = latents_raw.shape[0]
    for i in range(0, n, batch):
        z = latents_raw[i:i+batch].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = tok.decode(z).float()
        if psnr_vs is not None:
            ref = psnr_vs[i:i+batch].to(device)
            sse += float((x - ref).square().sum()); nel += ref.numel()
        img = x.add(1).div(2)
        f = extractor(img)
        moments.update(f)
        if sum(t.shape[0] for t in kid_feats) < 5000:
            kid_feats.append(f[:5000 - sum(t.shape[0] for t in kid_feats)].cpu())
    gm, gc = moments.compute()
    out = {"fid": _fid(reference["feature_mean"], reference["feature_covariance"], gm, gc),
           "kid": _kid(reference["kid_features"], torch.cat(kid_feats))}
    if psnr_vs is not None:
        mse = sse / nel
        out["psnr_db"] = 10.0 * torch.log10(torch.tensor(4.0 / mse)).item()
    return out

results = {}

# 1. Reconstruction FID/KID + PSNR on the full 10k test cache
results["reconstruction_full16"] = decode_stats(test, psnr_vs=real_pm1)
print("recon:", results["reconstruction_full16"], flush=True)

# 2. Decoder sensitivity: isotropic noise in standardized units
gen = torch.Generator().manual_seed(0)
for sig in (0.05, 0.1, 0.2, 0.4):
    noisy = test + torch.randn(test.shape, generator=gen) * (sig * std)
    results[f"noise_iso_{sig}"] = decode_stats(noisy, psnr_vs=real_pm1)
    print("noise", sig, results[f"noise_iso_{sig}"], flush=True)
# slot-resolved: same total budget on token 1 only vs tokens 9..16 only
for name, slots in (("tok1", [0]), ("tail8", list(range(8, 16)))):
    noisy = test.clone()
    noisy[:, slots] += torch.randn(noisy[:, slots].shape, generator=gen) * (0.2 * std)
    results[f"noise_{name}_0.2"] = decode_stats(noisy, psnr_vs=real_pm1)
    print("noise", name, results[f"noise_{name}_0.2"], flush=True)

# 3. Gaussian-fit baseline: full 1024-D moment match on the train cache
flat = train.reshape(train.shape[0], -1).double()
mu = flat.mean(0)
xc = flat - mu
cov = (xc.T @ xc) / (flat.shape[0] - 1)
evals, evecs = torch.linalg.eigh(cov)
evals = evals.clamp_min(0)
p = evals / evals.sum()
eff_rank_1024 = float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum()))
L = evecs * evals.sqrt()
gs = torch.Generator().manual_seed(7)
eps = torch.randn(5000, flat.shape[1], generator=gs, dtype=torch.float64)
gauss = (mu + eps @ L.T).float().reshape(5000, 16, 64)
results["gaussian_fit_5k"] = decode_stats(gauss)
print("gauss:", results["gaussian_fit_5k"], flush=True)

# 4. Latent geometry on standardized train latents
zs = ((flat - flat.mean()) / flat.std())  # tensor-wide standardization, fp64
cov_s = ((zs - zs.mean(0)).T @ (zs - zs.mean(0))) / (zs.shape[0] - 1)
coord_std = zs.std(0)
k4 = ((zs - zs.mean(0)) ** 4).mean(0) / zs.var(0) ** 2
top = torch.linalg.eigvalsh(cov_s).flip(0)
share = (top / top.sum()).cumsum(0)
geom = {
    "effective_rank_1024": eff_rank_1024,
    "coord_std_min_med_max_standardized": [float(coord_std.min()), float(coord_std.median()), float(coord_std.max())],
    "kurtosis_med_p95_max": [float(k4.median()), float(k4.quantile(0.95)), float(k4.max())],
    "abs_max_sigma": float(zs.abs().max()),
    "evr_top_8_32_128_256": [float(share[7]), float(share[31]), float(share[127]), float(share[255])],
}
# slot-wise linear predictability from the exact joint covariance (needs full second moments)
D = 64
full_mu = flat.mean(0)
fc = flat - full_mu
C = (fc.T @ fc) / (flat.shape[0] - 1)
resid_frac = []
for k in range(1, 16):
    idx_p = torch.arange(0, k * D); idx_k = torch.arange(k * D, (k + 1) * D)
    Cpp = C[idx_p][:, idx_p]; Cpk = C[idx_p][:, idx_k]; Ckk = C[idx_k][:, idx_k]
    sol = torch.linalg.solve(Cpp + 1e-8 * torch.eye(len(idx_p), dtype=torch.float64), Cpk)
    schur = Ckk - Cpk.T @ sol
    resid_frac.append(float(torch.trace(schur) / torch.trace(Ckk)))
geom["slot_linear_residual_fraction"] = [1.0] + [round(v, 4) for v in resid_frac]
results["geometry"] = geom
print(json.dumps(geom, indent=1), flush=True)

with open(OUT, "w") as f:
    json.dump(results, f, indent=1, sort_keys=True)
print("WROTE", OUT, flush=True)
