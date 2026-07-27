#!/usr/bin/env python3
"""Build matched-step AFIG architecture scorecards from W&B runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Sequence

import wandb


CORE_METRICS = (
    "spectral/normalized_active_mse",
    "spectral/physical_complex_nrmse",
    "spectral/log_amplitude_mae",
    "spectral/log_amplitude_bias",
    "spectral/phase_circular_error",
    "spectral/phase_coherence",
    "spectral/radial_power_relative_error",
)
TIMESTEP_METRICS = tuple(
    f"spectral/timestep/{step}/{metric}"
    for step in (0, 533, 999)
    for metric in ("physical_complex_nrmse", "phase_circular_error")
)
ROBUSTNESS_METRICS = (
    "robustness/gaussian/physical_complex_nrmse",
    "robustness/gaussian/phase_circular_error",
    "robustness/gaussian_delta/physical_complex_nrmse",
    "robustness/gaussian_delta/phase_circular_error",
    "robustness/condition_cosine",
    "robustness/condition_relative_rms",
)
OPTIMIZATION_METRICS = (
    "grad_norm",
    "projection_grad_rms/token_proj",
    "projection_grad_rms/diffusion_input",
    "projection_grad_rms/diffusion_output",
    "output_gain/mean",
    "output_gain/min",
    "output_gain/max",
)
PROFILES = {
    "core": CORE_METRICS,
    "timestep": TIMESTEP_METRICS,
    "robustness": ROBUSTNESS_METRICS,
    "optimization": OPTIMIZATION_METRICS,
}

HIGHER_IS_BETTER = {
    "spectral/phase_coherence",
    "robustness/condition_cosine",
}

CONFIG_KEYS = (
    "seed",
    "max_train_steps",
    "train_batch_size",
    "backbone_position_mode",
    "adam_beta1",
    "adam_beta2",
    "history_cartesian_features",
    "history_polar_features",
    "centering",
    "input_timestep_conditioning",
    "input_projection_init",
    "loss_metric",
    "orbit_scale_exponent",
    "learned_output_gain",
    "history_corruption",
)

ARM_PATTERN = re.compile(
    r"^arch-(?P<arm>.+)-s(?P<seed>\d+)-b(?P<batch>\d+)-n(?P<steps>\d+)$"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--project", default="afig-continuous")
    parser.add_argument("--group", default="afig-coefficient-architecture-gates")
    parser.add_argument("--name-regex", default=r"^arch-")
    parser.add_argument(
        "--steps",
        default="5000,30000,100000",
        help="Comma-separated optimizer steps. Sub-30k rows are marked exploratory.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=sorted(PROFILES),
        default=[],
        help="Metric family; repeat as needed. Defaults to all profiles.",
    )
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--step-tolerance", type=int, default=1000)
    parser.add_argument(
        "--attempt-policy",
        choices=("newest", "all"),
        default="newest",
        help="Resolve duplicate run names created by reruns.",
    )
    parser.add_argument("--state", default="finished")
    parser.add_argument("--max-runs", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        default="analysis/architecture_gates",
    )
    return parser.parse_args(argv)


def parse_steps(text: str) -> list[int]:
    steps = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not steps or any(step <= 0 for step in steps):
        raise ValueError("--steps must contain positive integers")
    return steps


def metric_keys(args: argparse.Namespace) -> list[str]:
    profiles = args.profile or list(PROFILES)
    keys = {
        metric
        for profile in profiles
        for metric in PROFILES[profile]
    }
    for item in args.metric:
        keys.update(part.strip() for part in item.split(",") if part.strip())
    return sorted(keys)


def parse_arm(run_name: str) -> dict[str, Any]:
    match = ARM_PATTERN.match(run_name)
    if match is None:
        raise ValueError(f"Run name does not follow architecture convention: {run_name}")
    values = match.groupdict()
    return {
        "arm": values["arm"].lower(),
        "seed": int(values["seed"]),
        "batch": int(values["batch"]),
        "budget_steps": int(values["steps"]),
    }


def evidence_tier(step: int) -> str:
    if step < 30000:
        return "exploratory"
    if step < 100000:
        return "medium"
    return "confirmation"


def control_arm(arm: str) -> str | None:
    if arm in ("p1", "p2"):
        return "p0"
    if arm.startswith("b-") and arm != "b-default":
        return "b-default"
    if arm in ("r1", "r2"):
        return "r0"
    if arm in ("n1", "n2"):
        return "n0"
    if arm == "s1":
        return "s0"
    if arm in ("f-alpha02", "f-alpha1"):
        return "f-alpha0"
    if arm == "f-gain":
        return "f-alpha02"
    if arm == "g-noise":
        return "g-clean"
    if arm in ("h-finalist1", "h-finalist2"):
        return "h-anchor"
    return None


def resolve_attempts(runs: Iterable[Any], policy: str) -> list[Any]:
    ordered = sorted(runs, key=lambda run: run.created_at or "", reverse=True)
    if policy == "all":
        return ordered
    selected = []
    seen = set()
    for run in ordered:
        if run.name in seen:
            continue
        seen.add(run.name)
        selected.append(run)
    return selected


def nearest_row(
    history: Sequence[dict[str, Any]],
    requested_step: int,
    tolerance: int,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in history
        if "_step" in row and int(row["_step"]) <= requested_step
    ]
    if not candidates:
        return None
    matched = max(candidates, key=lambda row: int(row["_step"]))
    if requested_step - int(matched["_step"]) > tolerance:
        return None
    return matched


def is_higher_better(metric: str) -> bool:
    return metric in HIGHER_IS_BETTER or metric.endswith("/phase_coherence")


def metric_improvement(metric: str, value: float, baseline: float) -> float | None:
    if metric == "spectral/log_amplitude_bias":
        return abs(baseline) - abs(value)
    if metric == "grad_norm" or metric.startswith(
        ("projection_grad_rms/", "output_gain/")
    ):
        return None
    if is_higher_better(metric):
        return value - baseline
    return baseline - value


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def add_control_deltas(
    rows: list[dict[str, Any]],
    metrics: Sequence[str],
) -> None:
    index = {
        (row["arm"], row["seed"], row["requested_step"]): row
        for row in rows
    }
    for row in rows:
        control = control_arm(row["arm"])
        if control is None:
            continue
        baseline = index.get((control, row["seed"], row["requested_step"]))
        if baseline is None:
            continue
        row["control_arm"] = control
        for metric in metrics:
            value = finite_number(row.get(metric))
            base = finite_number(baseline.get(metric))
            if value is None or base is None:
                continue
            delta = value - base
            row[f"delta/{metric}"] = delta
            improvement = metric_improvement(metric, value, base)
            if improvement is not None:
                row[f"improvement/{metric}"] = improvement


def aggregate_rows(
    rows: Sequence[dict[str, Any]],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["requested_step"])].append(row)
    output = []
    for (arm, requested_step), members in sorted(groups.items()):
        result: dict[str, Any] = {
            "arm": arm,
            "requested_step": requested_step,
            "evidence_tier": evidence_tier(requested_step),
            "num_seeds": len({member["seed"] for member in members}),
            "run_ids": ",".join(member["run_id"] for member in members),
        }
        for key in (
            *metrics,
            *(f"improvement/{metric}" for metric in metrics),
        ):
            values = [
                number
                for member in members
                if (number := finite_number(member.get(key))) is not None
            ]
            if not values:
                continue
            result[f"mean/{key}"] = mean(values)
            result[f"std/{key}"] = stdev(values) if len(values) > 1 else None
            result[f"min/{key}"] = min(values)
            result[f"max/{key}"] = max(values)
        output.append(result)
    return output


def select_runs(api: wandb.Api, args: argparse.Namespace) -> list[Any]:
    entity = args.entity or getattr(api, "default_entity", None)
    if not entity:
        raise ValueError("Set --entity or WANDB_ENTITY")
    filters: dict[str, Any] = {"group": args.group}
    if args.state:
        filters["state"] = args.state
    pattern = re.compile(args.name_regex)
    candidates = api.runs(
        f"{entity}/{args.project}",
        filters=filters,
        order="-created_at",
    )
    matching = []
    for run in candidates:
        if pattern.search(run.name or ""):
            matching.append(run)
        if len(matching) >= args.max_runs:
            break
    return resolve_attempts(matching, args.attempt_policy)


def run_rows(
    runs: Sequence[Any],
    requested_steps: Sequence[int],
    metrics: Sequence[str],
    tolerance: int,
) -> list[dict[str, Any]]:
    output = []
    for run in runs:
        identity = parse_arm(run.name)
        available = [metric for metric in metrics if metric in run.summary]
        if not available:
            continue
        max_requested = min(max(requested_steps), identity["budget_steps"])
        history = list(
            run.scan_history(
                keys=["_step", *available],
                min_step=max(0, min(requested_steps) - tolerance),
                # W&B scan_history treats max_step as an exclusive bound.
                max_step=max_requested + 1,
                page_size=1000,
            )
        )
        for requested_step in requested_steps:
            if requested_step > identity["budget_steps"]:
                continue
            matched = nearest_row(history, requested_step, tolerance)
            if matched is None:
                continue
            row = {
                "run_id": run.id,
                "run_name": run.name,
                "url": run.url,
                "created_at": run.created_at,
                **identity,
                "requested_step": requested_step,
                "matched_step": int(matched["_step"]),
                "step_lag": requested_step - int(matched["_step"]),
                "evidence_tier": evidence_tier(requested_step),
            }
            for key in CONFIG_KEYS:
                row[f"config/{key}"] = run.config.get(key)
            for metric in available:
                value = finite_number(matched.get(metric))
                if value is not None:
                    row[metric] = value
            output.append(row)
    add_control_deltas(output, metrics)
    return output


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(rows, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_report(
    path: Path,
    aggregate: Sequence[dict[str, Any]],
) -> None:
    columns = (
        "mean/spectral/physical_complex_nrmse",
        "mean/spectral/log_amplitude_mae",
        "mean/spectral/phase_circular_error",
        "mean/spectral/radial_power_relative_error",
    )
    with path.open("w") as stream:
        stream.write("# AFIG architecture gate scorecard\n\n")
        stream.write(
            "Optimizer steps are the comparison axis. Results below 30k steps "
            "are exploratory and must not determine a final promotion alone.\n\n"
        )
        stream.write(
            "| Arm | Step | Tier | Seeds | Physical NRMSE | Log-amp MAE | "
            "Phase error | Radial error |\n"
        )
        stream.write("|---|---:|---|---:|---:|---:|---:|---:|\n")
        for row in aggregate:
            values = [
                (
                    f"{row.get(column):.6f}"
                    if finite_number(row.get(column)) is not None
                    else ""
                )
                for column in columns
            ]
            stream.write(
                f"| {row['arm']} | {row['requested_step']} | "
                f"{row['evidence_tier']} | {row['num_seeds']} | "
                + " | ".join(values)
                + " |\n"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.step_tolerance < 0:
        raise ValueError("--step-tolerance must be non-negative")
    steps = parse_steps(args.steps)
    metrics = metric_keys(args)
    runs = select_runs(wandb.Api(), args)
    rows = run_rows(runs, steps, metrics, args.step_tolerance)
    aggregate = aggregate_rows(rows, metrics)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "runs.csv", rows)
    write_csv(output_dir / "scorecard.csv", aggregate)
    write_json(output_dir / "scorecard.json", aggregate)
    write_report(output_dir / "scorecard.md", aggregate)
    print(
        f"Wrote {len(rows)} matched run rows and {len(aggregate)} scorecard rows "
        f"to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
