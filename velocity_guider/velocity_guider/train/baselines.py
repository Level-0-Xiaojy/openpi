"""Simple validation baselines for Velocity Guider."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import confusion_matrix, metrics_from_confusion


def chunk_motion_score(action_chunk: torch.Tensor) -> torch.Tensor:
    """Average absolute step delta over time and action dimensions."""

    delta = action_chunk[:, 1:] - action_chunk[:, :-1]
    return delta.abs().mean(dim=(1, 2))


@dataclass
class MotionThresholdBaseline:
    """Map chunk speed to classes using two quantile thresholds.

    Smaller movement chunks predict v_mode=3/class 0, medium predicts class 1,
    and larger movement chunks predict v_mode=1/class 2.
    """

    low_threshold: float
    high_threshold: float

    @classmethod
    def fit_from_loader(
        cls,
        loader: DataLoader,
        *,
        low_q: float = 1 / 3,
        high_q: float = 2 / 3,
        max_batches: int | None = None,
    ) -> "MotionThresholdBaseline":
        scores: list[np.ndarray] = []
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            action = batch.get("raw_action_chunk", batch["action_chunk"])
            scores.append(chunk_motion_score(action).cpu().numpy())
        if not scores:
            raise ValueError("Cannot fit baseline from an empty loader.")
        all_scores = np.concatenate(scores, axis=0)
        return cls(
            low_threshold=float(np.quantile(all_scores, low_q)),
            high_threshold=float(np.quantile(all_scores, high_q)),
        )

    def predict(self, action_chunk: torch.Tensor) -> torch.Tensor:
        scores = chunk_motion_score(action_chunk)
        pred = torch.zeros_like(scores, dtype=torch.long)
        pred[scores >= self.low_threshold] = 1
        pred[scores >= self.high_threshold] = 2
        return pred


@torch.no_grad()
def evaluate_majority_baseline(loader: DataLoader, majority_class: int = 0) -> dict[str, float]:
    cm = torch.zeros((3, 3), dtype=torch.long)
    for batch in loader:
        labels = batch["label"]
        preds = torch.full_like(labels, int(majority_class))
        cm += confusion_matrix(preds, labels, num_classes=3)
    out = {f"majority/{k}": v for k, v in metrics_from_confusion(cm).items()}
    out["majority/class"] = float(majority_class)
    return out


@torch.no_grad()
def evaluate_motion_baseline(
    baseline: MotionThresholdBaseline,
    loader: DataLoader,
) -> dict[str, float]:
    cm = torch.zeros((3, 3), dtype=torch.long)
    for batch in loader:
        action = batch.get("raw_action_chunk", batch["action_chunk"])
        preds = baseline.predict(action)
        cm += confusion_matrix(preds, batch["label"], num_classes=3)
    out = {f"motion_baseline/{k}": v for k, v in metrics_from_confusion(cm).items()}
    out["motion_baseline/low_threshold"] = baseline.low_threshold
    out["motion_baseline/high_threshold"] = baseline.high_threshold
    return out
