"""Does per-orbit mean subtraction corrupt phase?

Power lives in the norm of a complex coefficient and phase in its angle, so a
subtractive normalization can rotate the phase arbitrarily.  For z = 0.25+0.25i a
mean of 0.1 barely moves the angle, but a mean of 0.5 inverts it.  Whether this
matters in practice depends entirely on |mean| relative to the coefficient
spread, which is measurable.

Prediction worth testing: natural images have no systematic alignment, so every
non-DC orbit should have a near-zero mean and centering should be nearly
phase-neutral there, while DC carries mean brightness and a large mean -- but DC
is self-conjugate and purely real, so it has no phase to corrupt.

Reports per ring: |mean|/std, the median absolute phase rotation induced by
centering, and the fraction of coefficients whose phase moves more than 90
degrees.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
import torchvision
from torchvision import transforms

from frequency import FrequencyCodec, FrequencyCodecConfig

NUM_CHANNELS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec_stats", default="autoencoder_runs/codec_stats_32.pt")
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_images", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    payload = torch.load(args.codec_stats, map_location="cpu", weights_only=False)
    config = FrequencyCodecConfig(**payload["config"])
    codec = FrequencyCodec(config)
    codec.load_exported(payload)
    codec = codec.to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        args.data_root, train=True, download=False, transform=transform
    )
    images = torch.stack(
        [dataset[i][0] for i in range(args.num_images)], dim=0
    ).to(device)

    raw = codec.encode_raw(images)  # [B, L, 6] = 3 real then 3 imag
    mean = codec.orbit_mean.to(device)  # [L, 6]
    std = codec.orbit_std.to(device).clamp_min(1e-12)
    radius_bin = codec.radius_bin.to(device)
    self_conjugate = codec.is_self_conjugate.to(device)

    rows: List[Dict[str, float]] = []
    for radius in range(int(radius_bin.max()) + 1):
        selector = radius_bin == radius
        if not bool(selector.any()):
            continue
        # Complex coefficient per colour channel, before and after centering.
        before_angles = []
        after_angles = []
        for channel in range(NUM_CHANNELS):
            real = raw[:, selector, channel]
            imag = raw[:, selector, channel + NUM_CHANNELS]
            mean_real = mean[selector, channel]
            mean_imag = mean[selector, channel + NUM_CHANNELS]
            before = torch.complex(real, imag)
            after = torch.complex(real - mean_real, imag - mean_imag)
            before_angles.append(torch.angle(before))
            after_angles.append(torch.angle(after))
        before_angle = torch.cat(before_angles, dim=-1)
        after_angle = torch.cat(after_angles, dim=-1)
        delta = torch.atan2(
            torch.sin(after_angle - before_angle),
            torch.cos(after_angle - before_angle),
        ).abs()

        ratio = (
            mean[selector].abs() / std[selector]
        ).mean()
        rows.append(
            {
                "radius": radius,
                "orbits": int(selector.sum()),
                "self_conjugate": bool(self_conjugate[selector].all()),
                "mean_over_std": float(ratio),
                "median_phase_shift_deg": float(delta.median() * 180.0 / torch.pi),
                "p95_phase_shift_deg": float(
                    delta.flatten().quantile(0.95) * 180.0 / torch.pi
                ),
                "frac_phase_shift_gt_90deg": float((delta > torch.pi / 2).float().mean()),
            }
        )

    report = {"num_images": int(images.shape[0]), "rings": rows}
    path = os.path.join(args.output_dir, "phase_centering_report.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"{'ring':>5} {'orbits':>7} {'selfconj':>9} {'|mean|/std':>11}"
          f" {'med dphase':>11} {'p95 dphase':>11} {'frac>90deg':>11}")
    for row in rows:
        print(
            f"{row['radius']:>5d} {row['orbits']:>7d} {str(row['self_conjugate']):>9}"
            f" {row['mean_over_std']:>11.4f} {row['median_phase_shift_deg']:>10.2f}d"
            f" {row['p95_phase_shift_deg']:>10.2f}d {row['frac_phase_shift_gt_90deg']:>11.4f}"
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
