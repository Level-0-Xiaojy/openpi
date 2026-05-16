"""Train Velocity Guider.

Single GPU:
    uv run python velocity_guider/train.py --config velocity_guider/configs/train_v1.yaml

Multi GPU:
    torchrun --standalone --nproc_per_node=4 velocity_guider/train.py --config velocity_guider/configs/train_v1.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from velocity_guider.model import VelocityGuiderConfig
from velocity_guider.train.trainer import DataConfig, OptimConfig, TrainConfig, WandbConfig, run_training


T = TypeVar("T")


def _update_dataclass(instance: T, values: dict[str, Any]) -> T:
    valid_fields = {f.name: f for f in fields(instance)}
    for key, value in values.items():
        if key not in valid_fields:
            raise KeyError(f"Unknown config field {key!r} for {type(instance).__name__}")
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            # YAML lists are more convenient; model config stores hidden dims as tuple.
            if key == "hidden_dims" and isinstance(value, list):
                value = tuple(value)
            setattr(instance, key, value)
    return instance


def load_train_config(path: str | Path) -> TrainConfig:
    with Path(path).open("r") as f:
        raw = yaml.safe_load(f) or {}
    cfg = TrainConfig(
        data=DataConfig(),
        model=VelocityGuiderConfig(),
        optim=OptimConfig(),
        wandb=WandbConfig(),
    )
    return _update_dataclass(cfg, raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Velocity Guider.")
    parser.add_argument("--config", type=str, required=True, help="Path to train YAML config.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override data.dataset_root.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output_dir.")
    parser.add_argument("--epochs", type=int, default=None, help="Override optim.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override data.batch_size per process.")
    parser.add_argument("--wandb-mode", type=str, default=None, help="Override wandb.mode, e.g. online/offline/disabled.")
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    if args.dataset_root is not None:
        cfg.data.dataset_root = args.dataset_root
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.epochs is not None:
        cfg.optim.epochs = args.epochs
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.wandb_mode is not None:
        cfg.wandb.mode = args.wandb_mode
        if args.wandb_mode == "disabled":
            cfg.wandb.enabled = False

    run_training(cfg)


if __name__ == "__main__":
    main()
