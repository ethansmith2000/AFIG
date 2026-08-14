#!/usr/bin/env python3
"""AR attribution: prefix-replacement rollouts and slot-1 marginal diagnostics.

For j in {0,1,2,4,8}: feed ground-truth (test-cache) tokens for slots < j, generate
the remaining slots with the AR head, decode all 16, compute FID/KID vs the real
test reference. j=0 reproduces the standard unconditional generation.
"""
import json, sys
sys.path.insert(0, "/workspace/AFIG")
import torch
from live_evaluation import InceptionFeatures, StreamingMoments, _fid, _kid
from progressive_tokenizer import AutoregressiveFlowConfig, AutoregressiveRectifiedFlow
from progressive_tokenizer.checkpoints import load_tokenizer_checkpoint

OUT = "/tmp/claude-0/-workspace/d4e8f99c-1817-41a7-8b75-8aa285967645/scratchpad/afig_diag_prefix_rollout.json"
CKPT = "/workspace/AFIG/prior_runs/v2-ar-cross-n16-d64-s1/checkpoint_final.pt"
CACHE = "/workspace/AFIG/tokenizer_runs/v2-cross-n16-d64-s1/latents_final_original_flip.pt"
REF = "/workspace/AFIG/data/cifar10_test_inception.pt"
N, BATCH, STEPS = 5000, 250, 50

device = torch.device("cuda")
payload = torch.load(CKPT, map_location="cpu", weights_only=False)
model = AutoregressiveRectifiedFlow(AutoregressiveFlowConfig(**payload["model_config"]))
model.load_state_dict(payload["model"]); model = model.to(device).eval()
mean = payload["normalization"]["mean"].float().to(device)
scale = payload["normalization"]["scale"].float().to(device)
cache = torch.load(CACHE, map_location="cpu", weights_only=False)
test_std = ((cache["test_latents"].float() - mean.cpu()) / scale.cpu())[:N]
tok, _ = load_tokenizer_checkpoint(payload["tokenizer_checkpoint"])
tok = tok.to(device).eval().requires_grad_(False)
reference = torch.load(REF, map_location="cpu", weights_only=False)
extractor = InceptionFeatures(device)
L = model.config.sequence_length

@torch.no_grad()
def rollout(j, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    moments = StreamingMoments(2048); kid_feats = []
    slot_tokens = []
    for i in range(0, N, BATCH):
        gt = test_std[i:i+BATCH].to(device)
        tokens = torch.zeros(gt.shape[0], L, model.config.token_dim, device=device)
        tokens[:, :j] = gt[:, :j]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for index in range(j, L):
                condition = model.trunk(tokens)[:, index]
                tokens[:, index] = model.head.sample(condition, steps=STEPS, generator=gen)
            raw = tokens.float() * scale + mean
            img = tok.decode(raw).float().add(1).div(2)
        f = extractor(img)
        moments.update(f)
        if sum(t.shape[0] for t in kid_feats) < 5000:
            kid_feats.append(f[:5000 - sum(t.shape[0] for t in kid_feats)].cpu())
        slot_tokens.append(tokens[:, min(j, L - 1)].cpu())
    gm, gc = moments.compute()
    return {"fid": _fid(reference["feature_mean"], reference["feature_covariance"], gm, gc),
            "kid": _kid(reference["kid_features"], torch.cat(kid_feats))}, torch.cat(slot_tokens)

results = {}
gen_tok1 = None
for j in (0, 1, 2, 4, 8):
    res, first_gen_slot = rollout(j, 1000 + j)
    results[f"prefix_{j}"] = res
    if j == 0:
        gen_tok1 = first_gen_slot  # generated slot-1 tokens from unconditional run
    print("prefix", j, res, flush=True)

# slot-1 marginal: generated (j=0) vs real test slot-1, standardized units
real1 = test_std[:, 0].double(); fake1 = gen_tok1.double()
mu_r, mu_f = real1.mean(0), fake1.mean(0)
cr = torch.cov(real1.T); cf = torch.cov(fake1.T)
proj = torch.randn(64, 256, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
proj = proj / proj.norm(dim=0, keepdim=True)
sw = []
for c in range(256):
    a = (real1 @ proj[:, c]).sort().values
    b = (fake1 @ proj[:, c]).sort().values
    m = min(len(a), len(b)); a = a[torch.linspace(0, len(a) - 1, m).long()]; b = b[torch.linspace(0, len(b) - 1, m).long()]
    sw.append(float(((a - b) ** 2).mean()))
results["slot1_marginal"] = {
    "mean_l2_gap": float((mu_r - mu_f).norm()),
    "real_mean_norm": float(mu_r.norm()),
    "cov_frobenius_gap_rel": float((cr - cf).norm() / cr.norm()),
    "real_cov_trace": float(torch.trace(cr)), "fake_cov_trace": float(torch.trace(cf)),
    "sliced_w2_mean": sum(sw) / len(sw),
}
print(json.dumps(results["slot1_marginal"], indent=1), flush=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=1, sort_keys=True)
print("WROTE", OUT, flush=True)
