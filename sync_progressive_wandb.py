#!/usr/bin/env python3
"""Backfill and follow a local progressive-token training log into W&B."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from progressive_tokenizer.tracking import WandbTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--log_file", type=Path, required=True)
    parser.add_argument("--project", default="afig-progressive-tokenizer")
    parser.add_argument("--group", default="tokenizer-v2")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--poll_seconds", type=float, default=10.0)
    parser.add_argument("--follow", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def log_record(tracker: WandbTracker, record: dict) -> None:
    if "evaluation" in record:
        metrics = record["evaluation"]
        tracker.log(metrics, step=int(metrics["step"]), prefix="eval")
    elif "preview" in record:
        metrics = record["preview"]
        tracker.log(metrics, step=int(metrics["step"]), prefix="preview")
    elif "final" in record:
        metrics = record["final"]
        tracker.log(metrics, step=int(metrics["step"]), prefix="eval/final")
    elif "step" in record and "loss" in record:
        tracker.log(record, step=int(record["step"]), prefix="train")


def image_step(path: Path) -> int | None:
    stem = path.stem
    suffix = stem.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    tracker = WandbTracker(
        enabled=True,
        output_dir=args.run_dir,
        project=args.project,
        name=args.run_name or args.run_dir.name,
        group=args.group,
        config=config,
    )
    if tracker.run is None:
        raise RuntimeError("W&B initialization failed; see warning above")

    state_path = args.run_dir / "wandb_sync_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {"line_count": 0, "images": []}
    uploaded_images = set(state.get("images", []))

    while True:
        lines = args.log_file.read_text().splitlines() if args.log_file.exists() else []
        start = min(int(state.get("line_count", 0)), len(lines))
        for line in lines[start:]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                log_record(tracker, record)
        state["line_count"] = len(lines)

        image_paths = sorted(args.run_dir.glob("reconstruction_*.png"))
        image_paths.extend(sorted(args.run_dir.glob("samples_*.png")))
        for path in image_paths:
            relative = path.name
            if relative in uploaded_images:
                continue
            step = image_step(path)
            if step is None:
                continue
            key = "eval/reconstruction" if path.name.startswith("reconstruction") else "preview/samples"
            tracker.log_image(path, step=step, key=key)
            uploaded_images.add(relative)

        final_image = args.run_dir / "reconstruction_final.png"
        final_metrics = args.run_dir / "metrics_final.json"
        if final_image.exists() and final_image.name not in uploaded_images:
            metrics = json.loads(final_metrics.read_text()) if final_metrics.exists() else {}
            step = int(metrics.get("step", 0))
            tracker.log_image(
                final_image, step=step, key="eval/final_reconstruction"
            )
            uploaded_images.add(final_image.name)

        state["images"] = sorted(uploaded_images)
        atomic_json(state_path, state)
        if not args.follow:
            break
        if final_metrics.exists() and final_image.exists():
            break
        time.sleep(args.poll_seconds)

    tracker.finish()


if __name__ == "__main__":
    main()
