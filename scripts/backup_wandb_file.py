#!/usr/bin/env python3
"""Upload one file to W&B as a named artifact (per-artifact inline backup)."""
import sys, wandb
path, name = sys.argv[1], sys.argv[2]
run = wandb.init(project="afig-progressive-tokenizer", job_type="backup", name=f"backup-{name}")
art = wandb.Artifact(name, type="checkpoints")
art.add_file(path)
run.log_artifact(art)
run.finish()
print(f"backed up {path} as {name}")
