"""Quantify the geometry hidden by the phrase "frequency token".

The matched joint controls use two materially different 48-D token layouts:

* FFT: eight complex RGB Hermitian orbits (8 * 6 real values), ordered by radius.
* DCT/Hartley: a contiguous 4x4 tile of a real RGB frequency plane
  (16 * 3 real values).

This script compares their within-token frequency locality.  FFT orbit distance
is measured on the frequency torus modulo conjugacy, because k and -k name the
same Hermitian orbit.  The final padded FFT group is excluded.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

from frequency import build_orbit_table


Point = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--orbits_per_token", type=int, default=8)
    parser.add_argument("--tile", type=int, default=4)
    parser.add_argument(
        "--output", default="diagnostics/token_composition.json"
    )
    return parser.parse_args()


def torus_distance(a: Point, b: Point, size: int) -> float:
    dy = abs(a[0] - b[0])
    dx = abs(a[1] - b[1])
    dy = min(dy, size - dy)
    dx = min(dx, size - dx)
    return math.hypot(dy, dx)


def orbit_distance(a: Point, b: Point, size: int) -> float:
    partner = ((-b[0]) % size, (-b[1]) % size)
    return min(torus_distance(a, b, size), torus_distance(a, partner, size))


def radius(point: Point, size: int) -> float:
    y = min(point[0], size - point[0])
    x = min(point[1], size - point[1])
    return math.hypot(y, x)


def summarize(groups: list[list[Point]], size: int, *, modulo_conjugacy: bool) -> dict:
    metric = orbit_distance if modulo_conjugacy else torus_distance
    pair_means: list[float] = []
    diameters: list[float] = []
    local_fractions: list[float] = []
    radial_spreads: list[float] = []
    local_threshold = math.sqrt(2.0) + 1e-9
    for group in groups:
        distances = [
            metric(a, b, size) for a, b in itertools.combinations(group, 2)
        ]
        radii = [radius(point, size) for point in group]
        pair_means.append(sum(distances) / len(distances))
        diameters.append(max(distances))
        local_fractions.append(
            sum(distance <= local_threshold for distance in distances)
            / len(distances)
        )
        radial_spreads.append(max(radii) - min(radii))
    return {
        "tokens": len(groups),
        "points_per_token": len(groups[0]),
        "mean_pair_distance": sum(pair_means) / len(pair_means),
        "mean_diameter": sum(diameters) / len(diameters),
        "mean_local_pair_fraction": sum(local_fractions) / len(local_fractions),
        "mean_radial_spread": sum(radial_spreads) / len(radial_spreads),
    }


def main() -> None:
    args = parse_args()
    if args.size % args.tile:
        raise ValueError("size must be divisible by tile")

    table = build_orbit_table(args.size, args.size, ordering="radial")
    representatives = [
        (int(y), int(x)) for y, x in zip(table["ky"], table["kx"])
    ]
    fft_groups = [
        representatives[start : start + args.orbits_per_token]
        for start in range(0, len(representatives), args.orbits_per_token)
        if len(representatives[start : start + args.orbits_per_token])
        == args.orbits_per_token
    ]
    grid_groups = [
        [
            (grid_y + local_y, grid_x + local_x)
            for local_y in range(args.tile)
            for local_x in range(args.tile)
        ]
        for grid_y in range(0, args.size, args.tile)
        for grid_x in range(0, args.size, args.tile)
    ]

    result = {
        "size": args.size,
        "fft_radial_orbit_tokens": summarize(
            fft_groups, args.size, modulo_conjugacy=True
        ),
        "frequency_grid_tiles": summarize(
            grid_groups, args.size, modulo_conjugacy=False
        ),
        "interpretation": (
            "Radial FFT tokens preserve a narrow frequency-magnitude band but "
            "mix directions; grid tiles sacrifice radial homogeneity for local "
            "2-D frequency neighborhoods."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
