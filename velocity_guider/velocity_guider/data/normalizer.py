"""Action normalization utilities for Velocity Guider.

The dataset builder writes ``action_stats.json`` from train episodes only.
Training, validation, and online inference should reuse the same statistics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


ArrayLike = np.ndarray | torch.Tensor


@dataclass(frozen=True)
class ActionNormalizer:
    """Z-score normalizer for 14-D master action chunks.

    Args:
        mean: Per-action-dimension mean, shape ``[action_dim]``.
        std: Per-action-dimension std, shape ``[action_dim]``.
        eps: Lower bound for std to avoid division by tiny values.
        clip: Optional symmetric clipping after normalization. Leave ``None``
            for pure z-score normalization.
    """

    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6
    clip: float | None = None

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        if mean.ndim != 1 or std.ndim != 1:
            raise ValueError(f"mean/std must be 1-D, got {mean.shape=} {std.shape=}")
        if mean.shape != std.shape:
            raise ValueError(f"mean/std shape mismatch: {mean.shape} vs {std.shape}")
        safe_std = np.maximum(std, self.eps).astype(np.float32)

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", safe_std)

    @property
    def action_dim(self) -> int:
        return int(self.mean.shape[0])

    @classmethod
    def from_stats_path(
        cls,
        stats_path: str | Path,
        *,
        eps: float = 1e-6,
        clip: float | None = None,
    ) -> "ActionNormalizer":
        stats_path = Path(stats_path)
        with stats_path.open("r") as f:
            stats = json.load(f)
        return cls.from_stats_dict(stats, eps=eps, clip=clip)

    @classmethod
    def from_stats_dict(
        cls,
        stats: dict[str, Any],
        *,
        eps: float = 1e-6,
        clip: float | None = None,
    ) -> "ActionNormalizer":
        if "mean" not in stats or "std" not in stats:
            raise KeyError("Action stats must contain 'mean' and 'std'.")
        return cls(mean=np.asarray(stats["mean"]), std=np.asarray(stats["std"]), eps=eps, clip=clip)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "eps": float(self.eps),
            "clip": self.clip,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ActionNormalizer":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
            eps=float(state.get("eps", 1e-6)),
            clip=state.get("clip"),
        )

    def normalize(self, actions: ArrayLike) -> ArrayLike:
        """Normalize action arrays with shape ``[..., action_dim]``."""

        self._check_last_dim(actions)
        if isinstance(actions, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=actions.dtype, device=actions.device)
            std = torch.as_tensor(self.std, dtype=actions.dtype, device=actions.device)
            out = (actions - mean) / std
            if self.clip is not None:
                out = torch.clamp(out, -self.clip, self.clip)
            return out

        out = (actions.astype(np.float32, copy=False) - self.mean) / self.std
        if self.clip is not None:
            out = np.clip(out, -self.clip, self.clip)
        return out.astype(np.float32, copy=False)

    def denormalize(self, actions: ArrayLike) -> ArrayLike:
        """Invert :meth:`normalize` for arrays with shape ``[..., action_dim]``."""

        self._check_last_dim(actions)
        if isinstance(actions, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=actions.dtype, device=actions.device)
            std = torch.as_tensor(self.std, dtype=actions.dtype, device=actions.device)
            return actions * std + mean

        return (actions.astype(np.float32, copy=False) * self.std + self.mean).astype(np.float32, copy=False)

    def _check_last_dim(self, actions: ArrayLike) -> None:
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got shape {tuple(actions.shape)}")


def load_action_normalizer(
    dataset_root: str | Path,
    *,
    eps: float = 1e-6,
    clip: float | None = None,
) -> ActionNormalizer:
    """Load ``action_stats.json`` from a built Velocity Guider dataset root."""

    return ActionNormalizer.from_stats_path(Path(dataset_root) / "action_stats.json", eps=eps, clip=clip)
