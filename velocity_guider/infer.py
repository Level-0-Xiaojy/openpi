"""Inference helper for Velocity Guider checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from velocity_guider.data.dataset import LABEL_TO_V_MODE
from velocity_guider.data.normalizer import ActionNormalizer
from velocity_guider.model import VelocityGuider, VelocityGuiderConfig


def _config_from_checkpoint(raw_cfg: dict[str, Any]) -> VelocityGuiderConfig:
    model_cfg = raw_cfg.get("model", {})
    allowed = {f.name for f in fields(VelocityGuiderConfig)}
    values = {k: v for k, v in model_cfg.items() if k in allowed}
    if isinstance(values.get("hidden_dims"), list):
        values["hidden_dims"] = tuple(values["hidden_dims"])
    return VelocityGuiderConfig(**values)


class VelocityGuiderInfer:
    """Load a trained guider and predict execution velocity mode."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cuda",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if str(device) == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        self.cfg = _config_from_checkpoint(ckpt["config"])
        self.normalizer = ActionNormalizer.from_state_dict(ckpt["action_normalizer"])
        self.model = VelocityGuider(self.cfg)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(
        self,
        obs_feat: np.ndarray | torch.Tensor,
        action_chunk: np.ndarray | torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """Predict v_mode for ``obs_feat`` and raw action chunk.

        Args:
            obs_feat: ``[obs_dim]`` or ``[B, obs_dim]``.
            action_chunk: raw unnormalized ``[K, 14]`` or ``[B, K, 14]``.
        """

        obs = self._to_tensor(obs_feat)
        action = self._to_tensor(action_chunk)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if action.ndim == 2:
            action = action.unsqueeze(0)

        action = self.normalizer.normalize(action)
        logits = self.model(obs, action)
        probs = torch.softmax(logits, dim=-1)
        labels = probs.argmax(dim=-1)
        v_modes = torch.tensor([LABEL_TO_V_MODE[int(x)] for x in labels.cpu()], dtype=torch.long)
        return {
            "label": labels.cpu().numpy(),
            "v_mode": v_modes.numpy(),
            "prob": probs.cpu().numpy(),
        }

    def _to_tensor(self, array: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(array, torch.Tensor):
            return array.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a Velocity Guider checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    infer = VelocityGuiderInfer(args.checkpoint)
    obs = np.zeros((infer.cfg.obs_dim,), dtype=np.float32)
    action = np.zeros((infer.cfg.chunk_size, infer.cfg.action_dim), dtype=np.float32)
    print(infer.predict(obs, action))


if __name__ == "__main__":
    main()
