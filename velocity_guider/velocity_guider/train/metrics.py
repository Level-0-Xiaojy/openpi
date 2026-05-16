"""Metrics for 3-class Velocity Guider training."""

from __future__ import annotations

import torch


def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3,
) -> torch.Tensor:
    preds = preds.detach().view(-1).to(torch.long).cpu()
    targets = targets.detach().view(-1).to(torch.long).cpu()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    valid = (targets >= 0) & (targets < num_classes) & (preds >= 0) & (preds < num_classes)
    for t, p in zip(targets[valid], preds[valid], strict=False):
        cm[t, p] += 1
    return cm


def metrics_from_confusion(cm: torch.Tensor) -> dict[str, float]:
    cm = cm.to(torch.float64)
    total = cm.sum().item()
    correct = cm.diag().sum().item()
    metrics: dict[str, float] = {
        "acc": float(correct / total) if total > 0 else 0.0,
    }
    for c in range(cm.shape[0]):
        support = cm[c].sum().item()
        pred_count = cm[:, c].sum().item()
        tp = cm[c, c].item()
        metrics[f"class_{c}_acc"] = float(tp / support) if support > 0 else 0.0
        metrics[f"class_{c}_precision"] = float(tp / pred_count) if pred_count > 0 else 0.0
        metrics[f"class_{c}_support"] = float(support)
    return metrics


def average_confidence(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.detach(), dim=-1)
    return float(probs.max(dim=-1).values.mean().item())
