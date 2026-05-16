"""Velocity Guider model.

Scheme A: directly classify the best velocity mode from visual features and an
action chunk. Class labels follow ``data.dataset.V_MODE_TO_LABEL``:

    0 -> v_mode=3, 1 -> v_mode=2, 2 -> v_mode=1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _make_mlp(
    in_dim: int,
    hidden_dims: list[int],
    out_dim: int,
    *,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for hidden in hidden_dims:
        layers.extend([
            nn.Linear(prev, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
        prev = hidden
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


@dataclass
class VelocityGuiderConfig:
    obs_dim: int = 6144
    chunk_size: int = 20
    action_dim: int = 14
    action_embed_dim: int = 256
    obs_embed_dim: int = 256
    motion_embed_dim: int = 64
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.1
    num_classes: int = 3


class VelocityGuider(nn.Module):
    """A lightweight MLP classifier for choosing execution velocity mode."""

    def __init__(self, cfg: VelocityGuiderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or VelocityGuiderConfig()
        flat_action_dim = self.cfg.chunk_size * self.cfg.action_dim

        self.obs_encoder = _make_mlp(
            self.cfg.obs_dim,
            [self.cfg.obs_embed_dim],
            self.cfg.obs_embed_dim,
            dropout=self.cfg.dropout,
        )
        self.action_encoder = _make_mlp(
            flat_action_dim,
            [self.cfg.action_embed_dim],
            self.cfg.action_embed_dim,
            dropout=self.cfg.dropout,
        )

        # Motion summary gives the classifier direct access to scale/velocity
        # cues while keeping the flattened action path unchanged.
        motion_dim = self.cfg.action_dim * 4
        self.motion_encoder = _make_mlp(
            motion_dim,
            [self.cfg.motion_embed_dim],
            self.cfg.motion_embed_dim,
            dropout=self.cfg.dropout,
        )

        fused_dim = self.cfg.obs_embed_dim + self.cfg.action_embed_dim + self.cfg.motion_embed_dim
        self.classifier = _make_mlp(
            fused_dim,
            list(self.cfg.hidden_dims),
            self.cfg.num_classes,
            dropout=self.cfg.dropout,
        )

    def forward(self, obs_feat: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape ``[B, 3]``.

        Args:
            obs_feat: ``[B, obs_dim]`` visual features.
            action_chunk: normalized action chunk, ``[B, K, action_dim]``.
        """

        if obs_feat.ndim != 2:
            raise ValueError(f"obs_feat must be [B, obs_dim], got {tuple(obs_feat.shape)}")
        if action_chunk.ndim != 3:
            raise ValueError(f"action_chunk must be [B, K, action_dim], got {tuple(action_chunk.shape)}")
        if action_chunk.shape[1:] != (self.cfg.chunk_size, self.cfg.action_dim):
            raise ValueError(
                "action_chunk shape mismatch: "
                f"expected [B, {self.cfg.chunk_size}, {self.cfg.action_dim}], got {tuple(action_chunk.shape)}"
            )

        obs_emb = self.obs_encoder(obs_feat)
        action_emb = self.action_encoder(action_chunk.flatten(start_dim=1))
        motion_emb = self.motion_encoder(self._motion_features(action_chunk))
        return self.classifier(torch.cat([obs_emb, action_emb, motion_emb], dim=-1))

    def predict_proba(self, obs_feat: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(obs_feat, action_chunk), dim=-1)

    @staticmethod
    def _motion_features(action_chunk: torch.Tensor) -> torch.Tensor:
        delta = action_chunk[:, 1:] - action_chunk[:, :-1]
        start_to_end = action_chunk[:, -1] - action_chunk[:, 0]
        mean_abs_delta = delta.abs().mean(dim=1)
        max_abs_delta = delta.abs().amax(dim=1)
        rms_delta = torch.sqrt((delta.square()).mean(dim=1) + 1e-8)
        return torch.cat([start_to_end, mean_abs_delta, max_abs_delta, rms_delta], dim=-1)
