"""Small failure-tolerant W&B adapter for progressive-token experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional


def flatten_metrics(
    values: Mapping[str, Any], prefix: str = ""
) -> dict[str, int | float]:
    flattened: dict[str, int | float] = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_metrics(value, name))
        elif isinstance(value, bool):
            flattened[name] = int(value)
        elif isinstance(value, (int, float)):
            flattened[name] = value
    return flattened


class WandbTracker:
    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: Path,
        project: str,
        name: str,
        group: Optional[str],
        config: Mapping[str, Any],
    ) -> None:
        self.run = None
        self.output_dir = output_dir
        if not enabled:
            return
        try:
            import wandb

            metadata_path = output_dir / "wandb_run.json"
            run_id = None
            if metadata_path.exists():
                run_id = json.loads(metadata_path.read_text()).get("id")
            if not run_id:
                run_id = wandb.util.generate_id()
            self.run = wandb.init(
                project=project,
                name=name,
                group=group,
                config=dict(config),
                id=run_id,
                resume="allow",
                dir=str(output_dir),
            )
            self.run.define_metric("global_step")
            self.run.define_metric("train/*", step_metric="global_step")
            self.run.define_metric("eval/*", step_metric="global_step")
            self.run.define_metric("preview/*", step_metric="global_step")
            metadata_path.write_text(
                json.dumps(
                    {
                        "id": self.run.id,
                        "name": self.run.name,
                        "project": self.run.project,
                        "url": self.run.url,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        except Exception as error:
            print(
                json.dumps(
                    {
                        "wandb_warning": f"{type(error).__name__}: {error}",
                        "wandb_fallback": "local_json_and_png",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            self.run = None

    def log(self, values: Mapping[str, Any], *, step: int, prefix: str) -> None:
        if self.run is None:
            return
        try:
            payload = flatten_metrics(values, prefix)
            payload["global_step"] = int(step)
            self.run.log(payload)
        except Exception as error:
            print(
                json.dumps(
                    {"wandb_log_warning": f"{type(error).__name__}: {error}"},
                    sort_keys=True,
                ),
                flush=True,
            )

    def log_image(self, path: Path, *, step: int, key: str) -> None:
        if self.run is None or not path.exists():
            return
        try:
            import wandb

            self.run.log(
                {"global_step": int(step), key: wandb.Image(str(path))}
            )
        except Exception as error:
            print(
                json.dumps(
                    {"wandb_image_warning": f"{type(error).__name__}: {error}"},
                    sort_keys=True,
                ),
                flush=True,
            )

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
