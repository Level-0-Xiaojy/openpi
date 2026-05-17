"""封装 pi0 (PyTorch) vision tower，用作 Velocity Guider 的特征提取器。

设计要点：
- 复用 openpi 的 PaliGemmaWithExpertModel，并用和 ``scripts/train_pytorch.py`` 同款
  ``safetensors.torch.load_model`` 加载 ``paligemma_with_expert.*`` 权重。
  这里不直接实例化 ``PI0Pytorch``，因为它会检查本地 transformers_replace 安装状态；
  数据构造只需要 vision tower，不需要整套 action model。
- 推理时只调用 ``paligemma_with_expert.embed_image()`` → ``[B, num_patches, width]``，
  再 mean-pool patch 维度，得到每视图一个 ``[B, width]`` 向量，三视图 concat → ``[B, 3*width]``。
- 输入是 ``uint8 [B, H, W, C]`` （和 lerobot 数据集存储一致）；内部做归一化到 ``[-1, 1]`` +
  ``resize_with_pad_torch`` 到 224x224，匹配 openpi 的 ``preprocess_observation_pytorch`` eval 路径。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
from torch import nn

from openpi.models import gemma as _gemma
from openpi.models import pi0_config
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.shared import image_tools

logger = logging.getLogger(__name__)


IMAGE_RESOLUTION: tuple[int, int] = (224, 224)


class _PaliGemmaWeightWrapper(nn.Module):
    """保持 checkpoint key 前缀兼容的轻量 wrapper。

    pi0 PyTorch checkpoint 中相关权重名形如：
    ``paligemma_with_expert.paligemma.model.vision_tower...``。
    直接把 PaliGemmaWithExpertModel 挂在同名 attribute 下，就可以用
    ``safetensors.torch.load_model(..., strict=False)`` 加载 subset 权重。
    """

    def __init__(self, paligemma_variant: str, action_expert_variant: str, dtype: str):
        super().__init__()
        paligemma_config = _gemma.get_config(paligemma_variant)
        action_expert_config = _gemma.get_config(action_expert_variant)
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, False],
            precision=dtype,
        )


def _normalize_uint8_to_pm1_bhwc(img_uint8: torch.Tensor) -> torch.Tensor:
    """``uint8 [B, H, W, C]`` -> ``float32 [B, H, W, C]`` in ``[-1, 1]``。"""
    if img_uint8.dtype != torch.uint8:
        raise ValueError(f"expected uint8 input, got {img_uint8.dtype}")
    return img_uint8.to(torch.float32) / 255.0 * 2.0 - 1.0


class VisionEncoder:
    """加载 pi0 PyTorch ckpt 中的 PaliGemma vision tower，提供 batch 特征提取接口。"""

    def __init__(
        self,
        checkpoint_path: str | os.PathLike,
        action_horizon: int = 20,
        device: str | torch.device = "cuda:0",
        dtype: str = "bfloat16",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.dtype = self._parse_dtype(dtype)

        model_cfg = pi0_config.Pi0Config(action_horizon=action_horizon, dtype=dtype)
        self.action_horizon = action_horizon
        self.paligemma_variant = model_cfg.paligemma_variant
        self.hidden_width = self._infer_paligemma_width(self.paligemma_variant)

        logger.info(
            "Building PaliGemmaWithExpertModel wrapper (paligemma_variant=%s, action_expert=%s) for vision-tower-only inference",
            model_cfg.paligemma_variant,
            model_cfg.action_expert_variant,
        )
        self.model = _PaliGemmaWeightWrapper(
            paligemma_variant=model_cfg.paligemma_variant,
            action_expert_variant=model_cfg.action_expert_variant,
            dtype=dtype,
        )

        weights_path = self.checkpoint_path / "model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"model.safetensors not found under {self.checkpoint_path}")
        logger.info("Loading weights from %s", weights_path)
        missing, unexpected = safetensors.torch.load_model(self.model, str(weights_path), strict=False)
        if missing:
            logger.warning("safetensors load_model missing keys (truncated 10): %s", missing[:10])
        if unexpected:
            logger.warning("safetensors load_model unexpected keys (truncated 10): %s", unexpected[:10])

        # 只保留 vision tower 这条链路；其他参数也保留在显存上没事，反正不会反传
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # 让 vision tower 用目标 dtype（embed_image 内部走 PaliGemma vision tower）
        # PI0Pytorch 内部已经在 init 时对部分 layer 强制 bf16/fp32，这里不再额外转换。

    @staticmethod
    def _parse_dtype(dtype: str) -> torch.dtype:
        return {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
        }[dtype]

    @staticmethod
    def _infer_paligemma_width(variant: str) -> int:
        # 与 openpi.models.gemma.get_config(variant) 对齐
        from openpi.models import gemma as _gemma

        return _gemma.get_config(variant).width

    @property
    def obs_feat_dim(self) -> int:
        """3 视图 × paligemma width，例如 gemma_2b → 3 × 2048 = 6144。"""
        return 3 * self.hidden_width

    @torch.no_grad()
    def encode(self, images_uint8: dict[str, np.ndarray]) -> np.ndarray:
        """对一个 batch 的三视图提取特征。

        Args:
            images_uint8: ``{camera_key: [B, H, W, C] uint8}``，必须含
                ``face_view, left_wrist_view, right_wrist_view``。

        Returns:
            ``[B, 3 * width] float32`` numpy 数组，按
            ``face_view -> left_wrist_view -> right_wrist_view`` 顺序拼接。
        """
        # lerobot dataset key → openpi 内部 key
        # face_view -> base_0_rgb
        # left_wrist_view -> left_wrist_0_rgb
        # right_wrist_view -> right_wrist_0_rgb
        # 但提取特征本身只是把图过 vision tower，所以这里直接按统一顺序处理就好。
        ordered_keys = ("face_view", "left_wrist_view", "right_wrist_view")
        feats: list[torch.Tensor] = []
        batch_size: int | None = None

        for cam in ordered_keys:
            if cam not in images_uint8:
                raise KeyError(f"missing camera key '{cam}' in images_uint8")
            arr = images_uint8[cam]
            if arr.ndim != 4 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
                raise ValueError(
                    f"camera '{cam}' expects uint8 [B, H, W, 3], got dtype={arr.dtype}, shape={arr.shape}"
                )
            if batch_size is None:
                batch_size = arr.shape[0]
            elif arr.shape[0] != batch_size:
                raise ValueError(
                    f"camera '{cam}' batch size {arr.shape[0]} != {batch_size} (expected aligned across views)"
                )

            t = torch.from_numpy(arr).to(self.device, non_blocking=True)  # uint8 BHWC
            t = _normalize_uint8_to_pm1_bhwc(t)
            # SigLIP / HF vision tower expects BCHW. 先转 BCHW 再 resize，可以避开
            # image_tools.resize_with_pad_torch 在 BHWC 且 B=1 时 squeeze batch 维的行为。
            t = t.permute(0, 3, 1, 2).contiguous()  # BCHW
            if t.shape[2:4] != IMAGE_RESOLUTION:
                t = image_tools.resize_with_pad_torch(t, *IMAGE_RESOLUTION)
            # Keep image input fp32. openpi 的 PaliGemmaWithExpertModel 会把部分 vision
            # layers（如 LayerNorm）保持 fp32；若这里强转 bf16，会在这些层触发 dtype mismatch。

            # embed_image 返回 [B, num_patches, width]，patch 数随分辨率而定（224/14=16 → 256）
            img_emb = self.model.paligemma_with_expert.embed_image(t)
            # mean-pool over patches → [B, width]
            pooled = img_emb.mean(dim=1)
            feats.append(pooled.to(torch.float32))

        out = torch.cat(feats, dim=-1)  # [B, 3 * width]
        return out.detach().cpu().numpy().astype(np.float32, copy=False)
