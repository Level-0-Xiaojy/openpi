import dataclasses
import logging
import os
import struct
import socket
import sys
import time
import types
from collections import deque
from pathlib import Path
from typing import Literal

import tyro
import json
import cv2
import numpy as np
import torch
import jax

from openpi.models import model as _model
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "velocity_guider"))

# ─────────────────────────────────────────────────────────────────────────────
# 合并脚本：一个脚本同时支持
#   1. BC 模型推理（等价于 x2robot_infer_seq.py，含 Velocity Guider / actions_factor）
#   2. CFGRL 模型推理（等价于 x2robot_infer_seq_lzh.py，含 Classifier-Free Guidance）
#   3. rollout 过程中人工接管再切回（等价于 x2robot_infer_seq_zyx.py 的断线重连机制）
#
# 用法示例：
#   # BC 模型（断线可重连 = 人工接管后可切回 rollout）
#   uv run scripts/x2robot_infer_seq_qiuyi.py \
#       --policy-config open_giftbox_sm2sm \
#       --policy-dir .../open_giftbox_xpc_0623062606270628_sm2sm/29999
#
#   # BC 模型 + Velocity Guider
#   uv run scripts/x2robot_infer_seq_qiuyi.py \
#       --policy-config open_giftbox_sm2sm --policy-dir .../29999 \
#       --guider-checkpoint .../best.pt
#
#   # CFGRL 模型
#   uv run scripts/x2robot_infer_seq_qiuyi.py \
#       --policy-config restock_cola_sm2sm --policy-mode sm2sm \
#       --policy-dir .../restock_cola_steam_dagger/global_step_30000 \
#       --prompt "Restock the goods onto the shelf." \
#       --cfg-enable --cfg-guidance-scale 2.5 --cfg-guidance-type positive --profile-infer
#
# 人工接管：机器人客户端断开连接即可切到人工接管；重新连接即切回 rollout，
# 服务端在每次新连接时重置 master_queue，不会携带接管前的旧动作历史。
# ─────────────────────────────────────────────────────────────────────────────

# bagging_4sku_sm2sm 可选 assets（与 config.py 中注释标签一致）
AssetPreset = Literal[
    "1121",
    "v0520",
    "v0525",
    "v0601",
    "v0602",
    "v0604_pi05",  # bagging_4sku_zyx_xpc_pi05_sm2sm_h3f2_a20_dm10dh50df50po20
    "v0604_pi0",   # bagging_4sku_zyx_ny_xpc_sm2sm_h3f2_a20_dm10dh50df50po20
    "v0630",
]
BAGGING_4SKU_ASSET_PRESETS: dict[str, str] = {
    "1121": "bagging_4sku_sm2sm_multi_bd90ba7812",
    "v0520": "bagging_4sku_sm2sm_multi_3617113b35",
    "v0525": "bagging_4sku_sm2sm_multi_b968c1739d",
    "v0601": "bagging_4sku_sm2sm_multi_1df01fc672",
    "v0602": "bagging_4sku_sm2sm_multi_1df01fc672",
    "v0604_pi05": "bagging_4sku_sm2sm_multi_067a46021d",
    "v0604_pi0": "bagging_4sku_sm2sm_multi_1df01fc672",
    "v0630": "bagging_4sku_sm2sm_multi_0602",
}


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    policy_config: str = "throw_sm2m"
    policy_dir: str = "checkpoints/throw_sm2m/throw_0113_sm2m_h5f3/29999"
    policy_mode: Literal["s2s", "s2m", "sm2m", "sm2sm"] | None = None
    host: str = "192.168.120.153"
    port: int = 57770
    log_replay: bool = False
    state_history_size: int = None
    state_future_size: int = None
    state_step: int = None
    move_steps: int = 15
    only_right_arm: bool = False
    latency_step: int = None
    # Natural-language task instruction for pi0 (match training style / language).
    # Empty keeps previous behavior.
    prompt: str = ""
    # 覆盖 config.py 中 assets；不传则使用 config 里当前生效的 asset_id
    asset_preset: AssetPreset | None = None

    # ─── Velocity Guider (BC path) ───────────────────────────────────────
    guider_checkpoint: str | None = None
    """Path to Velocity Guider best.pt. If None, v_mode prediction is disabled
    and actions_factor is always 3."""

    # ─── Profiling ───────────────────────────────────────────────────────
    profile_infer: bool = False
    profile_warmup: int = 3
    profile_report_every: int = 1

    # If set, dump each step's obs (images/state/prompt) and predicted actions to
    # this dir for offline visualization/replay.
    save_data_dir: str | None = None

    # If >0, smooth the chunk-boundary discontinuity by blending the first W
    # newly-predicted actions of each chunk with a linear extrapolation of the
    # previous chunk's tail velocity. W=0 disables. Recommended: 3.
    blend_steps: int = 0
    # Action dims to exclude from blending (passed through as raw model output).
    # Defaults to Aloha gripper dims (6=left gripper, 13=right gripper) so that
    # grasp/release timing is not slowed by smoothing.
    blend_skip_dims: tuple[int, ...] = (6, 13)

    # ─── CFG (Classifier-Free Guidance) ──────────────────────────────────
    # Only meaningful for ckpts trained with cfg-sft (Advantage-tag routing).
    # Off by default; original single-prompt path is preserved when disabled.
    cfg_enable: bool = False
    # 0.0  → pure unconditional (base prompt, no Advantage tag)
    # 1.0  → pure conditional (equivalent to appending \nAdvantage: positive)
    # >1.0 → standard CFG amplification (commonly 1.5 / 2.0 / 3.0)
    cfg_guidance_scale: float = 1.0
    # positive: cond pass uses "{prompt}\n{cfg_pos_suffix}"
    # negative: cond pass uses "{prompt}\n{cfg_neg_suffix}"
    # no_guide: skip cond pass; equivalent to scale=0
    cfg_guidance_type: Literal["positive", "negative", "no_guide"] = "positive"
    # Defaults match RLinf training-side template
    # (rlinf/data/datasets/cfg/__init__.py:116-117).
    cfg_pos_suffix: str = "Advantage: positive"
    cfg_neg_suffix: str = "Advantage: negative"
    # Override denoise step count; None → reuse policy._sample_kwargs (typically 10).
    cfg_num_steps: int | None = None


_CFG_DEBUG_BUDGET = int(os.environ.get("CFG_DEBUG_BUDGET", "1"))


# ═════════════════════════════════════════════════════════════════════════════
# Velocity Guider (BC path)
# ═════════════════════════════════════════════════════════════════════════════
def _extract_obs_feat_from_policy(policy: _policy.Policy) -> np.ndarray | None:
    """Extract obs_feat from pi0's cached image embeddings after policy.infer().

    The PI0Pytorch model caches per-camera image embeddings in
    ``_cached_image_embeds`` during ``embed_prefix``.  We mean-pool the
    patch dimension and concatenate across cameras to get ``[1, 3*width]``.

    Returns None if the model is not PyTorch or the cache is empty.
    """
    if not policy._is_pytorch_model:
        return None
    model = policy._model
    cache = getattr(model, "_cached_image_embeds", None)
    if not cache:
        return None
    pooled = []
    for emb in cache:
        pooled.append(emb.mean(dim=1).to(torch.float32))
    obs_feat = torch.cat(pooled, dim=-1)
    return obs_feat.cpu().numpy().astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# CFG (Classifier-Free Guidance) — CFGRL path
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def _cfg_sample_actions_method(
    self,
    observation_uncond,
    observation_cond,
    *,
    num_steps: int,
    guidance_scale: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """torch.compile-friendly CFG sampler bound onto a PI0Pytorch instance.

    Companion of `_cfg_sample_actions` but designed to be wrapped by
    `torch.compile(mode="max-autotune")` and bound as an instance method on
    `policy._model` (see main()). To keep dynamo's trace single-path:
      - no `pure_cond` short-circuit (always run both uncond and cond denoise)
      - no `use_cond` branching: caller must provide a real `observation_cond`
        (the no_guide path is handled in `_cfg_infer` by falling through to
        `policy.infer`, which uses the already-compiled `sample_actions`)
      - `guidance_scale` is passed as a torch.Tensor to avoid Python-scalar
        guards triggering recompiles per sweep value.
    """
    bsize = observation_uncond.state.shape[0]
    device = observation_uncond.state.device
    if noise is None:
        actions_shape = (
            bsize,
            self.config.action_horizon,
            self.config.action_dim,
        )
        noise = self.sample_noise(actions_shape, device)

    # uncond prefix forward
    images_u, img_masks_u, lang_tokens_u, lang_masks_u, state = (
        self._preprocess_observation(observation_uncond, train=False)
    )
    prefix_embs_u, prefix_pad_masks_u, prefix_att_masks_u = self.embed_prefix(
        images_u, img_masks_u, lang_tokens_u, lang_masks_u
    )
    prefix_att_2d_masks_u = make_att_2d_masks(prefix_pad_masks_u, prefix_att_masks_u)
    prefix_position_ids_u = torch.cumsum(prefix_pad_masks_u, dim=1) - 1
    prefix_att_2d_masks_4d_u = self._prepare_attention_masks_4d(prefix_att_2d_masks_u)
    self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, past_kv_u = self.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d_u,
        position_ids=prefix_position_ids_u,
        past_key_values=None,
        inputs_embeds=[prefix_embs_u, None],
        use_cache=True,
    )

    # cond prefix forward
    images_c, img_masks_c, lang_tokens_c, lang_masks_c, _ = (
        self._preprocess_observation(observation_cond, train=False)
    )
    prefix_embs_c, prefix_pad_masks_c, prefix_att_masks_c = self.embed_prefix(
        images_c, img_masks_c, lang_tokens_c, lang_masks_c
    )
    prefix_att_2d_masks_c = make_att_2d_masks(prefix_pad_masks_c, prefix_att_masks_c)
    prefix_position_ids_c = torch.cumsum(prefix_pad_masks_c, dim=1) - 1
    prefix_att_2d_masks_4d_c = self._prepare_attention_masks_4d(prefix_att_2d_masks_c)
    _, past_kv_c = self.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d_c,
        position_ids=prefix_position_ids_c,
        past_key_values=None,
        inputs_embeds=[prefix_embs_c, None],
        use_cache=True,
    )

    # Euler denoise loop with CFG mixing — single trace path
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    x_t = noise
    time_t = torch.tensor(1.0, dtype=torch.float32, device=device)
    while time_t >= -dt / 2:
        expanded_time = time_t.expand(bsize)
        v_t_uncond = self.denoise_step(
            state, prefix_pad_masks_u, past_kv_u, x_t, expanded_time
        )
        v_t_cond = self.denoise_step(
            state, prefix_pad_masks_c, past_kv_c, x_t, expanded_time
        )
        v_t = (1.0 - guidance_scale) * v_t_uncond + guidance_scale * v_t_cond
        x_t = x_t + dt * v_t
        time_t = time_t + dt
    return x_t


@torch.no_grad()
def _cfg_sample_actions(
    model,
    observation_uncond,
    observation_cond,
    *,
    num_steps: int,
    guidance_scale: float,
    device,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Two-pass CFG sampling for pi0_pytorch.

    Mirrors RLinf openpi_cfg_action_model.sample_actions (lines 781-905):
    runs prefix forward twice (one for the unconditional pass, one for the
    Advantage-tagged conditional pass), then mixes per-step velocity
    `v_t = (1 - w) * v_uncond + w * v_cond` inside the Euler denoise loop.

    `observation_cond=None` → no_guide path; equivalent to guidance_scale=0.
    Skips the redundant uncond denoise call when guidance_scale ≈ 1 (pure cond).

    Diagnostics: set env CFG_DEBUG=1 to log v_uncond / v_cond / v_t / delta
    norms for the first N inference calls (N = CFG_DEBUG_BUDGET, default 1).
    Use this to confirm the cond pass really runs and produces a velocity
    distinct from uncond.
    """
    global _CFG_DEBUG_BUDGET
    do_dbg = bool(os.environ.get("CFG_DEBUG")) and _CFG_DEBUG_BUDGET > 0
    if do_dbg:
        _CFG_DEBUG_BUDGET -= 1
        logging.info(
            "[CFG dbg] enter: scale=%g  use_cond=%s  num_steps=%d",
            guidance_scale,
            observation_cond is not None,
            num_steps,
        )
    bsize = observation_uncond.state.shape[0]
    if noise is None:
        actions_shape = (
            bsize,
            model.config.action_horizon,
            model.config.action_dim,
        )
        noise = model.sample_noise(actions_shape, device)

    # ── uncond prefix forward ──────────────────────────────────────────
    images_u, img_masks_u, lang_tokens_u, lang_masks_u, state = (
        model._preprocess_observation(observation_uncond, train=False)
    )
    prefix_embs_u, prefix_pad_masks_u, prefix_att_masks_u = model.embed_prefix(
        images_u, img_masks_u, lang_tokens_u, lang_masks_u
    )
    prefix_att_2d_masks_u = make_att_2d_masks(prefix_pad_masks_u, prefix_att_masks_u)
    prefix_position_ids_u = torch.cumsum(prefix_pad_masks_u, dim=1) - 1
    prefix_att_2d_masks_4d_u = model._prepare_attention_masks_4d(prefix_att_2d_masks_u)
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, past_kv_u = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d_u,
        position_ids=prefix_position_ids_u,
        past_key_values=None,
        inputs_embeds=[prefix_embs_u, None],
        use_cache=True,
    )

    # ── cond prefix forward (skipped on no_guide / scale==0) ──────────
    use_cond = observation_cond is not None and abs(guidance_scale) > 1e-9
    if use_cond:
        images_c, img_masks_c, lang_tokens_c, lang_masks_c, _ = (
            model._preprocess_observation(observation_cond, train=False)
        )
        prefix_embs_c, prefix_pad_masks_c, prefix_att_masks_c = model.embed_prefix(
            images_c, img_masks_c, lang_tokens_c, lang_masks_c
        )
        prefix_att_2d_masks_c = make_att_2d_masks(prefix_pad_masks_c, prefix_att_masks_c)
        prefix_position_ids_c = torch.cumsum(prefix_pad_masks_c, dim=1) - 1
        prefix_att_2d_masks_4d_c = model._prepare_attention_masks_4d(prefix_att_2d_masks_c)
        _, past_kv_c = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d_c,
            position_ids=prefix_position_ids_c,
            past_key_values=None,
            inputs_embeds=[prefix_embs_c, None],
            use_cache=True,
        )
    else:
        prefix_pad_masks_c = None
        past_kv_c = None

    # ── Euler denoise loop with CFG mixing ────────────────────────────
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    pure_cond = use_cond and abs(guidance_scale - 1.0) < 1e-9
    if do_dbg:
        logging.info(
            "[CFG dbg] loop: pure_cond_shortcut=%s  noise.norm=%.4f",
            pure_cond,
            float(noise.norm().item()),
        )
    step_idx = 0
    while time >= -dt / 2:
        expanded_time = time.expand(bsize)
        if pure_cond:
            # Skip the unused uncond pass when scale==1.
            v_t = model.denoise_step(
                state, prefix_pad_masks_c, past_kv_c, x_t, expanded_time
            )
            if do_dbg:
                logging.info(
                    "[CFG dbg] step=%2d  pure_cond  v_t.norm=%.4f",
                    step_idx,
                    float(v_t.norm().item()),
                )
        else:
            v_t_uncond = model.denoise_step(
                state, prefix_pad_masks_u, past_kv_u, x_t, expanded_time
            )
            if not use_cond:
                v_t = v_t_uncond
                if do_dbg:
                    logging.info(
                        "[CFG dbg] step=%2d  no_cond    v_uncond.norm=%.4f",
                        step_idx,
                        float(v_t_uncond.norm().item()),
                    )
            else:
                v_t_cond = model.denoise_step(
                    state, prefix_pad_masks_c, past_kv_c, x_t, expanded_time
                )
                v_t = (1.0 - guidance_scale) * v_t_uncond + guidance_scale * v_t_cond
                if do_dbg:
                    logging.info(
                        "[CFG dbg] step=%2d  scale=%-4g  "
                        "v_uncond.norm=%.4f  v_cond.norm=%.4f  "
                        "delta.norm=%.4f  v_t.norm=%.4f",
                        step_idx,
                        guidance_scale,
                        float(v_t_uncond.norm().item()),
                        float(v_t_cond.norm().item()),
                        float((v_t_cond - v_t_uncond).norm().item()),
                        float(v_t.norm().item()),
                    )
        x_t = x_t + dt * v_t
        time = time + dt
        step_idx += 1
    if do_dbg:
        logging.info(
            "[CFG dbg] done:  scale=%g  final_actions.norm=%.4f",
            guidance_scale,
            float(x_t.norm().item()),
        )
    return x_t


def _cfg_infer(
    policy,
    obs: dict,
    *,
    pos_suffix: str,
    neg_suffix: str,
    guidance_scale: float,
    guidance_type: str,
    num_steps: int | None = None,
) -> dict:
    """Two-pass CFG inference wrapper around `Policy._sample_actions`.

    Wraps the same input/output transform chain as `Policy.infer`
    (openpi/policies/policy.py:68-115), but builds two `Observation`
    instances (base prompt + Advantage-tagged prompt) and dispatches to:
      - `policy._model.sample_actions_cfg` (compiled, fast path) when main()
        has bound it; or
      - `_cfg_sample_actions` (eager, dbg-friendly path) as fallback.
    The `no_guide` case skips CFG entirely and reuses the already-compiled
    `policy.infer` single-pass path.
    """
    # no_guide → single forward path, fully reusing the compiled sample_actions
    if guidance_type == "no_guide":
        return policy.infer(obs)

    base_prompt = obs.get("prompt", "") or ""
    suffix = pos_suffix if guidance_type == "positive" else neg_suffix
    cond_prompt = base_prompt + ("\n" if base_prompt else "") + suffix

    def _pack(p: str):
        local = dict(obs)
        local["prompt"] = p
        local = jax.tree.map(lambda x: x, local)
        local = policy._input_transform(local)
        local = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(policy._pytorch_device)[None, ...],
            local,
        )
        return _model.Observation.from_dict(local), local

    obs_uncond, raw_uncond = _pack(base_prompt)
    obs_cond, _ = _pack(cond_prompt)

    num_steps_val = (
        num_steps if num_steps is not None
        else policy._sample_kwargs.get("num_steps", 10)
    )

    compiled_cfg = getattr(policy._model, "sample_actions_cfg", None)
    if compiled_cfg is not None:
        # Fast path: compiled method bound by main(); guidance_scale as tensor
        # to avoid per-value recompiles.
        guidance_scale_t = torch.tensor(
            guidance_scale, dtype=torch.float32, device=policy._pytorch_device
        )
        actions = compiled_cfg(
            obs_uncond, obs_cond,
            num_steps=num_steps_val,
            guidance_scale=guidance_scale_t,
        )
    else:
        # Eager fallback (CFG_NO_COMPILE=1); preserves CFG_DEBUG dbg logs.
        actions = _cfg_sample_actions(
            policy._model,
            obs_uncond,
            obs_cond,
            num_steps=num_steps_val,
            guidance_scale=guidance_scale,
            device=policy._pytorch_device,
        )

    outputs = {"state": raw_uncond["state"], "actions": actions}
    outputs = jax.tree.map(
        lambda x: (
            np.asarray(x[0, ...].detach().cpu()) if torch.is_tensor(x)
            else np.asarray(x[0, ...])
        ),
        outputs,
    )
    outputs = policy._output_transform(outputs)
    return outputs


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════
def _infer_profile_log(times_ms: list, prefix: str = "") -> None:
    if not times_ms:
        return
    a = np.asarray(times_ms, dtype=np.float64)
    logging.info(
        "%sinfer profile: n=%d  mean=%.2f ms  p50=%.2f ms  p99=%.2f ms",
        prefix, len(a), float(np.mean(a)),
        float(np.percentile(a, 50)), float(np.percentile(a, 99)),
    )


def _resolve_asset_id(preset: str | None) -> str | None:
    if preset is None:
        return None
    key = preset.lower()
    if key not in BAGGING_4SKU_ASSET_PRESETS:
        choices = ", ".join(sorted(BAGGING_4SKU_ASSET_PRESETS))
        raise ValueError(f"Unknown asset_preset '{preset}'. Choose from: {choices}")
    return BAGGING_4SKU_ASSET_PRESETS[key]


def _cfg_with_asset_id(cfg: _config.TrainConfig, asset_id: str | None) -> _config.TrainConfig:
    if asset_id is None:
        return cfg
    return dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data,
            assets=_config.AssetsConfig(asset_id=asset_id),
        ),
    )


def _load_norm_stats_from_cfg(cfg: _config.TrainConfig, policy_dir: str) -> dict | None:
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    return _checkpoints.load_norm_stats(Path(policy_dir) / "assets", data_config.asset_id)


def recv_all(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def read_size(conn) -> int:
    """Read a 4-byte little-endian length header, raising ConnectionError on EOF."""
    header = recv_all(conn, 4)
    if header is None:
        raise ConnectionError("client disconnected")
    return struct.unpack('<L', header)[0]


def read_img(conn):
    image_size = read_size(conn)
    image = recv_all(conn, image_size)
    if image is None:
        raise ConnectionError("client disconnected during image payload")
    nparr = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _blend_chunk_transition(
    action_pred: np.ndarray,
    master_queue: deque,
    blend_steps: int,
    skip_dims: tuple[int, ...] = (),
) -> np.ndarray:
    """Smooth the chunk-boundary discontinuity.

    `action_pred` here is shape (move_steps+1, 14) with action_pred[0] = anchor
    = master_queue[-1] (= previous chunk's last queued action). The model's
    fresh predictions live at action_pred[1:]. Position 0->1 is empirically
    3-17x larger than within-chunk steps (see analysis), causing periodic
    stutter at every chunk boundary.

    For each i = 1..W (W = min(blend_steps, len-1)):
        alpha = i / (W + 1)
        old_extrap = anchor + i * (anchor - master_queue[-2])  # extrapolate prev velocity
        action_pred[i] := (1 - alpha) * old_extrap + alpha * action_pred[i]

    Position 0 (anchor) and positions W+1.. are untouched. Dims listed in
    `skip_dims` (e.g. grippers) are passed through as raw model output, so that
    discrete grasp/release timing is not slowed by the smoothing.
    """
    if blend_steps <= 0 or len(master_queue) < 2 or action_pred.shape[0] < 2:
        return action_pred
    W = min(blend_steps, action_pred.shape[0] - 1)
    out = action_pred.astype(np.float64).copy()
    anchor = out[0]
    prev = np.asarray(master_queue[-2], dtype=np.float64)
    velocity = anchor - prev
    skip_idx = list(skip_dims) if skip_dims else None
    for i in range(1, W + 1):
        alpha = i / (W + 1)
        old_extrap = anchor + velocity * i
        blended = (1.0 - alpha) * old_extrap + alpha * out[i]
        if skip_idx:
            blended[skip_idx] = out[i, skip_idx]
        out[i] = blended
    return out


def _save_frame_data(
    save_dir: Path,
    frame_idx: int,
    images: dict,
    state: np.ndarray,
    prompt: str,
    action_pred_full: np.ndarray,
    action_sent: np.ndarray,
) -> None:
    frame_dir = save_dir / f"frame_{frame_idx:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for name, img in images.items():
        cv2.imwrite(str(frame_dir / f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    np.savez(
        frame_dir / "data.npz",
        state=state,
        action_pred_full=action_pred_full,
        action_sent=action_sent,
    )
    with open(frame_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt, "frame_idx": frame_idx}, f, ensure_ascii=False, indent=2)


def main(args: Args) -> None:
    # Auto-detect policy_mode from policy_dir if not specified
    if args.policy_mode is None:
        for mode in ['sm2sm', 'sm2m', 's2m', 's2s']:
            if mode in args.policy_dir.lower():
                args.policy_mode = mode
                logging.info(f"Auto-detected policy_mode from path: {args.policy_mode}")
                break
        if args.policy_mode is None:
            raise ValueError(
                f"Could not detect policy_mode from path: {args.policy_dir}. "
                f"Please specify --policy-mode"
            )

    # Load config params if not specified
    cfg = _config.get_config(args.policy_config)
    asset_id = _resolve_asset_id(args.asset_preset)
    cfg = _cfg_with_asset_id(cfg, asset_id)
    if asset_id is not None:
        logging.info("asset_preset=%s -> asset_id=%s", args.asset_preset, asset_id)
    if args.state_history_size is None:
        args.state_history_size = getattr(cfg.data, 'state_history_size', 0)
        logging.info(f"Using state_history_size from config: {args.state_history_size}")
    if args.state_future_size is None:
        args.state_future_size = getattr(cfg.data, 'state_future_size', 0)
        logging.info(f"Using state_future_size from config: {args.state_future_size}")
    if args.state_step is None:
        args.state_step = getattr(cfg.data, 'state_step', 1)
        logging.info(f"Using state_step from config: {args.state_step}")
    if args.latency_step is None:
        args.latency_step = args.state_future_size
        logging.info(f"Using latency_step equal to state_future_size: {args.latency_step}")

    # Load policy
    logging.info(f"Loading policy from {args.policy_dir}")
    policy = _policy_config.create_trained_policy(
        cfg, args.policy_dir, default_prompt=(args.prompt or None)
    )
    norm_stats = _load_norm_stats_from_cfg(cfg, args.policy_dir)

    # ── Load Velocity Guider (optional, BC path) ─────────────────────────
    guider = None
    if args.guider_checkpoint:
        from infer import VelocityGuiderInfer
        guider = VelocityGuiderInfer(args.guider_checkpoint, device="cuda:0")
        logging.info(f"Loaded Velocity Guider from {args.guider_checkpoint}")

    # ── Bind a compiled CFG sampler onto policy._model (fast path) ────────
    # Original sample_actions is wrapped with torch.compile(mode="max-autotune")
    # in PI0Pytorch.__init__ (pi0_pytorch.py:112). The CFG path was bypassing
    # that compile, paying full eager cost ~2.4× the compiled cost → CFG total
    # was ~5× single-pass instead of ~2×. Binding sample_actions_cfg here lets
    # it share the same compile treatment.
    #
    # First inference will trigger autotune (~30-120s for PaliGemma kernels).
    # Set CFG_NO_COMPILE=1 to fall back to the eager path (useful with
    # CFG_DEBUG=1 to log v_uncond/v_cond/v_t norms).
    if args.cfg_enable and not os.environ.get("CFG_NO_COMPILE"):
        compile_mode = os.environ.get("CFG_COMPILE_MODE", "max-autotune")
        logging.info(
            f"Binding compiled sample_actions_cfg (mode={compile_mode!r}); "
            f"first inference will trigger autotune."
        )
        policy._model.sample_actions_cfg = torch.compile(
            types.MethodType(_cfg_sample_actions_method, policy._model),
            mode=compile_mode,
        )
    elif args.cfg_enable:
        logging.info(
            "CFG_NO_COMPILE=1 → CFG path runs eager (slower; supports CFG_DEBUG=1)."
        )

    state_seq_len = args.state_history_size + 1 + args.state_future_size
    latency_len = args.state_history_size + 1 + args.latency_step

    save_dir: Path | None = None
    if args.save_data_dir:
        save_dir = Path(args.save_data_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving per-frame obs/action data to {save_dir}")
    save_frame_idx = 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(True)  # 设置通信是阻塞式
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ip = args.host
    port = args.port
    sock.bind((ip, port))
    sock.listen(1)
    print(f"Server is listening on {ip}:{port}")

    infer_times_ms: list = []
    infer_count = 0

    try:
        # ── Outer accept loop: survives client disconnect (human takeover)
        #    and reconnect (resume rollout). master_queue is reset per
        #    connection so takeover-era history is not carried over. ──────
        while True:
            conn, addr = sock.accept()
            print(f"Connection from {addr}")
            master_queue = deque(maxlen=100)  # queue_len * 14, reset on (re)connect
            try:
                while True:
                    data_size = read_size(conn)
                    data = recv_all(conn, data_size)
                    if data is None:
                        raise ConnectionError("client disconnected during payload")
                    action_data = json.loads(data.decode('utf8'))

                    left_agent_data = action_data['follow1_pos']  # (state_history_size + 1, 7)
                    right_agent_data = action_data['follow2_pos']  # (state_history_size + 1, 7)

                    image1 = read_img(conn)  # left
                    image2 = read_img(conn)  # front
                    image3 = read_img(conn)  # right

                    h, w, c = np.array(image1).shape
                    camera_front = np.array(image2).reshape(h, w, c)
                    camera_left = np.array(image1).reshape(h, w, c)
                    camera_right = np.array(image3).reshape(h, w, c)

                    state = np.zeros((state_seq_len, 32), dtype=np.float32)
                    slave_state = np.concatenate([left_agent_data, right_agent_data], axis=1)  # (h+1, 14)
                    slave_state = np.concatenate(
                        [slave_state] + [slave_state[-1:]] * args.state_future_size
                    )

                    if not master_queue:
                        master_queue.extend([slave_state[-1]] * max(state_seq_len, latency_len))

                    master_list = list(master_queue)[-latency_len:]
                    if args.latency_step < args.state_future_size:  # inpainting mode
                        master_list = master_list + [master_list[-1]] * (
                            args.state_future_size - args.latency_step
                        )
                        state[args.latency_step - args.state_future_size:, -1] = 1.0
                    else:  # naive async
                        master_list = master_list[:state_seq_len]
                    master_state = np.array(master_list)

                    if args.policy_mode in ["s2s", "s2m"]:
                        state[:, :14] = slave_state
                    else:
                        state[:, :28] = np.concatenate([slave_state, master_state], axis=1)

                    if args.only_right_arm:
                        mean = np.asarray(norm_stats["state"].mean)
                        state[:, 0:7] = mean[..., 0:7]
                        if args.policy_mode in ["sm2m", "sm2sm"]:
                            state[:, 14:21] = mean[..., 14:21]

                    obs = {
                        'images': {
                            'left_wrist_view': camera_left,
                            'face_view': camera_front,
                            'right_wrist_view': camera_right,
                        },
                        'prompt': args.prompt,
                        'state': state,
                    }

                    # ── Inference: CFG two-pass (CFGRL) or single-pass (BC) ──
                    if args.profile_infer:
                        _t0 = time.perf_counter()
                    if args.cfg_enable:
                        action_pred = _cfg_infer(
                            policy, obs,
                            pos_suffix=args.cfg_pos_suffix,
                            neg_suffix=args.cfg_neg_suffix,
                            guidance_scale=args.cfg_guidance_scale,
                            guidance_type=args.cfg_guidance_type,
                            num_steps=args.cfg_num_steps,
                        )
                    else:
                        action_pred = policy.infer(obs)
                    if args.profile_infer:
                        infer_count += 1
                        _dt_ms = (time.perf_counter() - _t0) * 1000
                        if infer_count > args.profile_warmup:
                            infer_times_ms.append(_dt_ms)
                            if len(infer_times_ms) % args.profile_report_every == 0:
                                _infer_profile_log(infer_times_ms)

                    action_pred = action_pred['actions']
                    action_pred_full = np.asarray(action_pred).copy()
                    if args.policy_mode == "sm2sm":
                        _, master_action = action_pred[:, :14], action_pred[:, 14:28]
                        action_pred = master_action

                    # ── Velocity Guider (BC path): predict v_mode / actions_factor
                    #    on the full chunk BEFORE latency truncation. ──────
                    actions_factor = 3
                    if guider is not None:
                        obs_feat = _extract_obs_feat_from_policy(policy)
                        if obs_feat is not None:
                            chunk_len = min(20, action_pred.shape[0])
                            chunk = action_pred[:chunk_len]
                            if chunk.shape[0] < 20:
                                pad = np.tile(chunk[-1:], (20 - chunk.shape[0], 1))
                                chunk = np.concatenate([chunk, pad], axis=0)
                            result = guider.predict(obs_feat, chunk[np.newaxis])
                            v_mode = int(result["v_mode"][0])
                            actions_factor = 3 if v_mode == 3 else 2
                            # temp modify
                            if result["prob"][0, 1] > 0.02 or result["prob"][0, 2] > 0.02:
                                actions_factor = 2
                            logging.info(
                                "v_mode=%d  actions_factor=%d  probs=[%.2f, %.2f, %.2f]",
                                v_mode, actions_factor,
                                result["prob"][0, 0], result["prob"][0, 1], result["prob"][0, 2],
                            )

                    action_pred = action_pred[args.latency_step:]
                    action_pred = action_pred[:args.move_steps, ...]  # (move_steps, 14)
                    action_pred = np.concatenate([[master_queue[-1]], action_pred])
                    if args.blend_steps > 0:
                        action_pred = _blend_chunk_transition(
                            action_pred, master_queue, args.blend_steps,
                            skip_dims=args.blend_skip_dims,
                        )
                    for action in action_pred[1:]:
                        master_queue.append(action)

                    if save_dir is not None:
                        _save_frame_data(
                            save_dir,
                            save_frame_idx,
                            images={
                                "left_wrist_view": camera_left,
                                "face_view": camera_front,
                                "right_wrist_view": camera_right,
                            },
                            state=state,
                            prompt=args.prompt,
                            action_pred_full=action_pred_full,
                            action_sent=action_pred,
                        )
                        save_frame_idx += 1

                    follow1_pos = action_pred[:, :7].tolist()
                    follow2_pos = action_pred[:, 7:].tolist()

                    data_dir = {
                        "follow1_pos": follow1_pos,
                        "follow2_pos": follow2_pos,
                    }
                    # Only include actions_factor when the Velocity Guider is
                    # active (BC path). CFG/BC-without-guider clients that don't
                    # expect this field thus receive the original response shape.
                    if guider is not None:
                        data_dir["actions_factor"] = actions_factor
                    data_str = json.dumps(data_dir)
                    data_bytes = data_str.encode('utf-8')
                    conn.sendall(struct.pack('<L', len(data_bytes)))
                    conn.sendall(data_bytes)
            except (ConnectionError, ConnectionResetError, BrokenPipeError) as exc:
                logging.info(f"Client disconnected: {exc}. Waiting for next connection.")
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
    except KeyboardInterrupt:
        pass
    finally:
        if args.profile_infer and infer_times_ms:
            _infer_profile_log(infer_times_ms, prefix="[final] ")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
