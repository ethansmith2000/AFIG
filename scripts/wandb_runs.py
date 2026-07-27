#!/usr/bin/env python3
"""Query AFIG W&B runs and export exact metric history without modifying runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import wandb


def _json_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _parse_key_value(items: Iterable[str]) -> dict[str, Any]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        result[key] = _json_value(value)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--project", default="afig-continuous")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run ID or exact run name; repeat to select multiple runs.",
    )
    parser.add_argument("--name-regex", help="Regex applied to run names.")
    parser.add_argument("--group")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--state", choices=["running", "finished", "failed", "crashed"])
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Match a run config value; VALUE accepts JSON syntax.",
    )
    parser.add_argument("--max-runs", type=int, default=50)
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Metric key or comma-separated keys; repeat as needed.",
    )
    parser.add_argument(
        "--metric-regex",
        help="Select metric keys matching this regex from run summaries.",
    )
    parser.add_argument("--config-key", action="append", default=[])
    parser.add_argument("--min-step", type=int)
    parser.add_argument("--max-step", type=int)
    parser.add_argument(
        "--step-interval",
        type=int,
        default=1,
        help="Keep history rows at least this many steps apart.",
    )
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--list", action="store_true", help="List matching runs only.")
    parser.add_argument("--format", choices=["csv", "jsonl", "json"], default="csv")
    parser.add_argument("--output", help="Output path; defaults to stdout.")
    return parser.parse_args(argv)


def _select_runs(api: wandb.Api, args: argparse.Namespace):
    entity = args.entity or getattr(api, "default_entity", None)
    if not entity:
        raise ValueError("Set --entity or WANDB_ENTITY.")
    filters: dict[str, Any] = {}
    if args.group:
        filters["group"] = args.group
    if args.state:
        filters["state"] = args.state
    if args.tag:
        filters["tags"] = {"$all": args.tag}
    for key, value in _parse_key_value(args.config).items():
        filters[f"config.{key}"] = value

    candidates = api.runs(
        f"{entity}/{args.project}",
        filters=filters or None,
        order="-created_at",
    )
    identifiers = set(args.run)
    name_pattern = re.compile(args.name_regex) if args.name_regex else None
    selected = []
    for run in candidates:
        if identifiers and run.id not in identifiers and run.name not in identifiers:
            continue
        if name_pattern is not None and not name_pattern.search(run.name or ""):
            continue
        selected.append(run)
        if len(selected) >= args.max_runs:
            break
    return selected


def _metric_keys(runs, args: argparse.Namespace) -> list[str]:
    explicit = []
    for item in args.metric:
        explicit.extend(key.strip() for key in item.split(",") if key.strip())
    keys = set(explicit)
    if args.metric_regex:
        pattern = re.compile(args.metric_regex)
        for run in runs:
            keys.update(key for key in run.summary.keys() if pattern.search(key))
    return sorted(keys)


def _run_metadata(run, config_keys: list[str]) -> dict[str, Any]:
    row = {
        "run_id": run.id,
        "run_name": run.name,
        "group": run.group,
        "state": run.state,
        "created_at": run.created_at,
        "url": run.url,
    }
    for key in config_keys:
        row[f"config/{key}"] = _scalar(run.config.get(key))
    return row


def _list_rows(runs, config_keys: list[str]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        row = _run_metadata(run, config_keys)
        row["tags"] = list(run.tags)
        rows.append(row)
    return rows


def _history_rows(
    runs,
    metric_keys: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not metric_keys:
        raise ValueError("Specify --metric or --metric-regex when exporting history.")
    rows = []
    keys = ["_step", *metric_keys]
    for run in runs:
        metadata = _run_metadata(run, args.config_key)
        last_step = None
        history = run.scan_history(
            keys=keys,
            min_step=0 if args.min_step is None else args.min_step,
            max_step=args.max_step,
            page_size=args.page_size,
        )
        for raw in history:
            step = int(raw["_step"])
            if last_step is not None and step - last_step < args.step_interval:
                continue
            row = dict(metadata)
            row["_step"] = step
            row.update({key: _scalar(raw.get(key)) for key in metric_keys})
            rows.append(row)
            last_step = step
    return rows


def _write_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    stream = open(args.output, "w", newline="") if args.output else sys.stdout
    try:
        if args.format == "json":
            json.dump(rows, stream, indent=2, allow_nan=False)
            stream.write("\n")
        elif args.format == "jsonl":
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        else:
            fieldnames = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    finally:
        if args.output:
            stream.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_runs <= 0:
        raise ValueError("--max-runs must be positive")
    if args.step_interval <= 0:
        raise ValueError("--step-interval must be positive")
    api = wandb.Api()
    runs = _select_runs(api, args)
    rows = (
        _list_rows(runs, args.config_key)
        if args.list
        else _history_rows(runs, _metric_keys(runs, args), args)
    )
    _write_rows(rows, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
