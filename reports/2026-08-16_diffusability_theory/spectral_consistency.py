"""Per-sample vs population spectral consistency on CIFAR-10.

Questions from Ethan's notes:
1. How consistent is the per-image radial power spectrum vs the population
   average (does the resolving ORDER of frequency bands vary per sample)?
2. Are nearby frequency bands' energies correlated per-image?
Baseline: white noise with matched per-pixel variance.
"""

import io

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

pf = pq.read_table(
    "/workspace/.hf_home/hub/datasets--uoft-cs--cifar10/snapshots/"
    "0b2714987fa478483af9968de7c934580d0bb9a2/plain_text/train-00000-of-00001.parquet"
)
imgs = [np.array(Image.open(io.BytesIO(b["bytes"]))) for b in pf.column("img").to_pylist()]
x = torch.from_numpy(np.stack(imgs)).float() / 127.5 - 1.0  # [50000,32,32,3]
# luma channel for the spectrum
luma = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])
luma = luma - luma.mean(dim=(1, 2), keepdim=True)

noise = torch.randn_like(luma) * luma.std()

freq = torch.fft.fftfreq(32) * 32
fy, fx = torch.meshgrid(freq, freq, indexing="ij")
radius = (fy ** 2 + fx ** 2).sqrt()

bands = [(0.5, 2.5), (2.5, 4.5), (4.5, 6.5), (6.5, 8.5), (8.5, 12.5), (12.5, 16.5)]
labels = ["r1-2", "r3-4", "r5-6", "r7-8", "r9-12", "r13-16"]
masks = [(radius > lo) & (radius <= hi) for lo, hi in bands]


def band_stats(images, name):
    F = torch.fft.fft2(images)
    power = F.abs() ** 2  # [N,32,32]
    # mean power per FFT bin within each band (area-normalized)
    bp = torch.stack([power[:, m].mean(dim=1) for m in masks], dim=1)  # [N, B]
    logp = bp.clamp_min(1e-12).log10()
    pop_mean = logp.mean(dim=0)
    pop_std = logp.std(dim=0)
    # adjacent-band ordering consistency per image
    order = [(bp[:, k] > bp[:, k + 1]).float().mean().item() for k in range(len(bands) - 1)]
    # per-image spectral slope over r in [1, 12]
    rmask = (radius > 0.5) & (radius <= 12.5)
    r = radius[rmask].log10()
    p = power[:, rmask].clamp_min(1e-12).log10()
    rc = r - r.mean()
    slope = (p * rc).sum(dim=1) / (rc ** 2).sum()
    # cross-band correlation of log-energies
    corr = torch.corrcoef(logp.T)
    print(f"\n=== {name} ===")
    print("band        :", "  ".join(f"{l:>7}" for l in labels))
    print("mean log10 P:", "  ".join(f"{v:7.3f}" for v in pop_mean))
    print("std  log10 P:", "  ".join(f"{v:7.3f}" for v in pop_std))
    print("P(band_k > band_k+1) per image:", "  ".join(f"{v:.3f}" for v in order))
    print(f"spectral slope: mean {slope.mean():.3f}  std {slope.std():.3f}")
    print("cross-band correlation of log-energy:")
    for i, l in enumerate(labels):
        print(f"  {l:>6}:", "  ".join(f"{corr[i, j]:6.3f}" for j in range(len(labels))))


band_stats(luma, "CIFAR-10 (luma, 50k)")
band_stats(noise, "white noise (matched var)")
