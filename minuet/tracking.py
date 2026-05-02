from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_wandb_config(config_path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text())
    return config.get("wandb", {})


def maybe_init_wandb(config_path: str | Path, extra_config: dict[str, Any] | None = None):
    wandb_cfg = load_wandb_config(config_path)
    if not wandb_cfg.get("enabled", False):
        return None

    import wandb

    name = os.getenv("WANDB_NAME", wandb_cfg.get("name"))
    project = os.getenv("WANDB_PROJECT", wandb_cfg.get("project"))
    tags = wandb_cfg.get("tags")
    if os.getenv("WANDB_TAGS"):
        tags = [tag.strip() for tag in os.getenv("WANDB_TAGS", "").split(",") if tag.strip()]

    run = wandb.init(
        project=project,
        name=name,
        tags=tags,
        config=extra_config or {},
    )
    return run
