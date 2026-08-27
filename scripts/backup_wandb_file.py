#!/usr/bin/env python3
"""Best-effort upload of one checkpoint as a named W&B artifact.

Telemetry must never prevent the following scientific evaluation phase. The
training harness already falls back to local JSON/PNG when W&B is unavailable;
this helper follows the same policy and reports a structured skip while exiting
successfully.
"""

from __future__ import annotations

import json
import sys

import wandb


def main() -> None:
    path, name = sys.argv[1], sys.argv[2]
    run = None
    try:
        run = wandb.init(
            project="afig-progressive-tokenizer",
            job_type="backup",
            name=f"backup-{name}",
        )
        artifact = wandb.Artifact(name, type="checkpoints")
        artifact.add_file(path)
        run.log_artifact(artifact)
        run.finish()
        print(json.dumps({"backed_up": path, "artifact": name}), flush=True)
    except Exception as error:  # W&B is optional experiment telemetry.
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception:
                pass
        print(
            json.dumps(
                {
                    "backup_skipped": path,
                    "artifact": name,
                    "warning": f"{type(error).__name__}: {error}",
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
