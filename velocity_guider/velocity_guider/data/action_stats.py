"""统计 14 维 master action 在训练集上的归一化数值（mean / std / quantile / min / max）。

这个统计只用 train 集（不污染 val），归一化时训练 / 推理共用。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .lerobot_loader import LeRobotDatasetInfo

logger = logging.getLogger(__name__)


def compute_master_action_stats(
    dataset_infos: list[LeRobotDatasetInfo],
    train_episodes_per_dataset: dict[str, list[int]],
) -> dict[str, list[float]]:
    """聚合所有 train episode 的 14 维 master action，计算统计量。

    Args:
        dataset_infos: ``LeRobotDatasetInfo`` 列表，每个对应一个 repo
        train_episodes_per_dataset: ``{repo_name: [ep_idx, ...]}`` 仅 train 部分

    Returns:
        dict 含 ``mean / std / q01 / q99 / min / max``，每个都是长度 14 的 list[float]。
    """
    chunks: list[np.ndarray] = []
    total_frames = 0
    for info in dataset_infos:
        ep_list = train_episodes_per_dataset.get(info.repo_name, [])
        for ep_idx in ep_list:
            arr = info.get_episode_master_actions(ep_idx)  # [T, 14]
            chunks.append(arr)
            total_frames += arr.shape[0]

    if not chunks:
        raise ValueError("No training frames to compute action stats from.")

    all_actions = np.concatenate(chunks, axis=0).astype(np.float64)  # [N, 14]
    logger.info(
        "Computing master action stats from %d frames across %d episodes",
        total_frames,
        sum(len(v) for v in train_episodes_per_dataset.values()),
    )

    mean = all_actions.mean(axis=0)
    std = all_actions.std(axis=0)
    q01 = np.quantile(all_actions, 0.01, axis=0)
    q99 = np.quantile(all_actions, 0.99, axis=0)
    mn = all_actions.min(axis=0)
    mx = all_actions.max(axis=0)

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "min": mn.tolist(),
        "max": mx.tolist(),
        "num_frames": int(total_frames),
        "action_dim": int(all_actions.shape[1]),
    }


def write_action_stats(stats: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Wrote action stats to %s", out_path)
