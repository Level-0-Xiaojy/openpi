"""PyTorch dataset for the built Velocity Guider training data.

Supports two data formats:
- **v1** (no ``burst_mask`` / ``is_burst`` column): all v_mode=3/2/1 samples are
  used, as before.
- **v2** (``burst_mask`` in npz + ``is_burst`` column in parquet): v_mode=2/1
  samples are only used for burst frames.  Non-burst frames contribute only
  a single v_mode=3 sample.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .normalizer import ActionNormalizer, load_action_normalizer

logger = logging.getLogger("velocity_guider.dataset")

V_MODE_TO_LABEL: dict[int, int] = {3: 0, 2: 1, 1: 2}
LABEL_TO_V_MODE: dict[int, int] = {v: k for k, v in V_MODE_TO_LABEL.items()}
V_MODE_AXIS_ORDER: tuple[int, int, int] = (3, 2, 1)


@dataclass(frozen=True)
class VelocityGuiderSampleIndex:
    out_path: str
    sample_id_in_episode: int
    is_burst: bool


class _NpzLruCache:
    def __init__(self, max_size: int = 16) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, path: str) -> dict[str, np.ndarray]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        with np.load(path) as data:
            item = {k: data[k] for k in data.files}
        self._cache[path] = item
        self._cache.move_to_end(path)
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return item


class VelocityGuiderDataset(Dataset[dict[str, Any]]):
    """Dataset with burst-aware v-mode expansion.

    Built ``.npz`` files store chunks as ``[num_t, 3, K, 14]`` with axis-1 order
    ``[v_mode=3, v_mode=2, v_mode=1]``.  The classifier labels are:

    - ``0``: v_mode=3, slowest / smoothest
    - ``1``: v_mode=2
    - ``2``: v_mode=1, fastest / strongest

    When ``is_burst`` information is available (v2 format), non-burst frames
    only produce a v_mode=3 sample; burst frames produce all three.
    v1 format (no ``is_burst`` column) falls back to expanding all three modes
    for every frame.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        *,
        normalizer: ActionNormalizer | None = None,
        samples_file: str = "samples.parquet",
        cache_size: int = 16,
        normalize_actions: bool = True,
        include_raw_action: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.normalize_actions = normalize_actions
        self.include_raw_action = include_raw_action

        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        samples_path = self.dataset_root / samples_file
        if not samples_path.exists():
            raise FileNotFoundError(f"Missing sample index: {samples_path}")
        df = pd.read_parquet(samples_path)
        df = df[df["split"] == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows for split={split!r} in {samples_path}")

        has_burst = "is_burst" in df.columns

        self._base_indices: list[VelocityGuiderSampleIndex] = []
        for row in df.itertuples(index=False):
            burst = bool(row.is_burst) if has_burst else True
            self._base_indices.append(
                VelocityGuiderSampleIndex(
                    out_path=str(row.out_path),
                    sample_id_in_episode=int(row.sample_id_in_episode),
                    is_burst=burst,
                )
            )

        self._expanded: list[tuple[int, int]] = []
        for base_idx, si in enumerate(self._base_indices):
            for mode_axis, vm in enumerate(V_MODE_AXIS_ORDER):
                if vm == 3 or si.is_burst:
                    self._expanded.append((base_idx, mode_axis))

        self.normalizer = normalizer or load_action_normalizer(self.dataset_root)
        self.cache = _NpzLruCache(max_size=cache_size)

        n_burst = sum(1 for si in self._base_indices if si.is_burst)
        n_total = len(self._base_indices)
        logger.info(
            "VelocityGuiderDataset[%s]: %d base samples, %d burst (%d%%), "
            "%d expanded samples (v1 would be %d)",
            split, n_total, n_burst,
            int(100 * n_burst / max(n_total, 1)),
            len(self._expanded),
            n_total * len(V_MODE_AXIS_ORDER),
        )

    def __len__(self) -> int:
        return len(self._expanded)

    @property
    def num_base_samples(self) -> int:
        return len(self._base_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_idx, mode_axis = self._expanded[index]
        sample_index = self._base_indices[base_idx]

        ep = self.cache.get(sample_index.out_path)
        sid = sample_index.sample_id_in_episode

        obs_feat = ep["obs_feat"][sid].astype(np.float32, copy=False)
        raw_action = ep["chunks"][sid, mode_axis].astype(np.float32, copy=False)
        v_mode = int(ep["v_modes"][sid, mode_axis])
        label = V_MODE_TO_LABEL[v_mode]

        action = self.normalizer.normalize(raw_action) if self.normalize_actions else raw_action

        item: dict[str, Any] = {
            "obs_feat": torch.from_numpy(obs_feat),
            "action_chunk": torch.from_numpy(action),
            "label": torch.tensor(label, dtype=torch.long),
            "v_mode": torch.tensor(v_mode, dtype=torch.long),
        }
        if self.include_raw_action:
            item["raw_action_chunk"] = torch.from_numpy(raw_action)
        return item


def create_velocity_guider_datasets(
    dataset_root: str | Path,
    *,
    action_clip: float | None = None,
    cache_size: int = 16,
) -> tuple[VelocityGuiderDataset, VelocityGuiderDataset]:
    normalizer = load_action_normalizer(dataset_root, clip=action_clip)
    train_ds = VelocityGuiderDataset(
        dataset_root,
        "train",
        normalizer=normalizer,
        cache_size=cache_size,
    )
    val_ds = VelocityGuiderDataset(
        dataset_root,
        "val",
        normalizer=normalizer,
        cache_size=cache_size,
    )
    return train_ds, val_ds
