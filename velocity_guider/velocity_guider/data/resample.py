"""14维 master action chunk 的重采样工具。

设计要点：
- 算法和 ``x2robot-slave/scripts/socket2ros_async.py::interpolates_actions`` 完全一致：
  pos / gripper 走 ``np.interp``，rotation (euler_xyz) 转四元数后 4 维线性插值再 renorm
  并转回 euler。区别只在于 target 长度参数化（任意正整数），且支持双臂 14 维。
- 用于构造 Velocity Guider 数据集的三档 chunk：
    * v_mode=3: ``demo[t:t+20]`` 原样，K=20
    * v_mode=2: ``demo[t:t+14]`` 重采样到 K=20
    * v_mode=1: ``demo[t:t+ 7]`` 重采样到 K=20
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def _resample_arm_7d(arm_7d: np.ndarray, target_len: int) -> np.ndarray:
    """对单臂 7 维 action 重采样。

    Args:
        arm_7d: ``[N_src, 7]``，列顺序 ``[pos(3), euler_xyz(3), gripper(1)]``
        target_len: 输出帧数（任意 ``>=2``）

    Returns:
        ``[target_len, 7]`` float32
    """
    if arm_7d.ndim != 2 or arm_7d.shape[1] != 7:
        raise ValueError(f"arm_7d must be [N, 7], got {arm_7d.shape}")
    n_src = arm_7d.shape[0]
    if n_src < 2:
        raise ValueError(f"n_src must be >= 2 for interpolation, got {n_src}")
    if target_len < 2:
        raise ValueError(f"target_len must be >= 2, got {target_len}")

    arm = arm_7d.astype(np.float64, copy=False)
    original = np.linspace(0, n_src - 1, n_src)
    target = np.linspace(0, n_src - 1, target_len)
    out = np.zeros((target_len, 7), dtype=np.float32)

    # pos (3) 线性
    for i in range(3):
        out[:, i] = np.interp(target, original, arm[:, i]).astype(np.float32)
    # gripper (1) 线性
    out[:, 6] = np.interp(target, original, arm[:, 6]).astype(np.float32)

    # euler -> quat -> 4 维线性插值 -> renorm -> euler
    quats = R.from_euler("xyz", arm[:, 3:6]).as_quat()  # [N_src, 4]
    interp_quats = np.zeros((target_len, 4), dtype=np.float64)
    for i in range(4):
        interp_quats[:, i] = np.interp(target, original, quats[:, i])
    norms = np.linalg.norm(interp_quats, axis=1, keepdims=True)
    interp_quats = interp_quats / np.maximum(norms, 1e-12)
    out[:, 3:6] = R.from_quat(interp_quats).as_euler("xyz").astype(np.float32)

    return out


def resample_master_chunk(master_actions_14d: np.ndarray, target_len: int = 20) -> np.ndarray:
    """对双臂 14 维 master action chunk 重采样。

    Args:
        master_actions_14d: ``[N_src, 14]``，前 7 维左臂，后 7 维右臂，每臂 ``[pos(3), euler(3), gripper(1)]``
        target_len: 输出帧数

    Returns:
        ``[target_len, 14]`` float32
    """
    if master_actions_14d.ndim != 2 or master_actions_14d.shape[1] != 14:
        raise ValueError(f"master_actions_14d must be [N, 14], got {master_actions_14d.shape}")
    left = _resample_arm_7d(master_actions_14d[:, :7], target_len)
    right = _resample_arm_7d(master_actions_14d[:, 7:], target_len)
    return np.concatenate([left, right], axis=1)


# 各 v_mode 对应的 source 长度（取多少帧 demo 原始 master action 用作重采样输入）
# 与 build_dataset 一致；落到任何脚本里需要查这个表
V_MODE_SOURCE_LEN: dict[int, int] = {3: 20, 2: 14, 1: 7}


def build_three_v_mode_chunks(
    master_actions_full: np.ndarray,
    t: int,
    chunk_size: int = 20,
) -> dict[int, np.ndarray] | None:
    """从 demo 同一时刻 ``t`` 构造三档 chunk。

    Args:
        master_actions_full: ``[T_ep, 14]`` 整个 episode 的 master action
        t: 起点帧索引
        chunk_size: 输出 chunk 长度 K（默认 20，必须等于 ``V_MODE_SOURCE_LEN[3]``）

    Returns:
        ``{v_mode: [K, 14] float32}``，如果某档 source 长度越界则返回 ``None``。
        v_mode=3 直接切片不重采样；v_mode=2/1 用 lerp+slerp 重采样到 ``chunk_size``。
    """
    if chunk_size != V_MODE_SOURCE_LEN[3]:
        raise ValueError(
            f"chunk_size ({chunk_size}) must equal V_MODE_SOURCE_LEN[3] ({V_MODE_SOURCE_LEN[3]})."
        )
    t_max = master_actions_full.shape[0]
    out: dict[int, np.ndarray] = {}
    for v_mode, src_len in V_MODE_SOURCE_LEN.items():
        end = t + src_len
        if end > t_max:
            return None
        src = master_actions_full[t:end].astype(np.float32, copy=False)
        if src_len == chunk_size:
            out[v_mode] = src.copy()
        else:
            out[v_mode] = resample_master_chunk(src, target_len=chunk_size)
    return out


# --------------------------------------------------------------------------
# 自检：和 socket2ros_async.py 的 interpolates_actions 对比
# --------------------------------------------------------------------------
def _reference_interpolates_actions(actions: np.ndarray, factor: int) -> np.ndarray:
    """`socket2ros_async.py::interpolates_actions` 的 verbatim 拷贝，用于自检。

    输入 ``[num_actions, 7]``，输出 ``[factor*(num_actions-1)+1, 7]``。
    """
    num_actions, action_dim = actions.shape
    target_num_actions = factor * (num_actions - 1) + 1

    original_indices = np.linspace(0, num_actions - 1, num_actions)
    target_indices = np.linspace(0, num_actions - 1, target_num_actions)
    interpolated_actions = np.zeros((target_num_actions, action_dim))

    for i in range(3):
        interpolated_actions[:, i] = np.interp(target_indices, original_indices, actions[:, i])
    interpolated_actions[:, -1] = np.interp(target_indices, original_indices, actions[:, -1])

    quaternions = R.from_euler("xyz", actions[:, 3:6]).as_quat()
    interpolated_quats = np.zeros((target_num_actions, 4))
    for i in range(4):
        interpolated_quats[:, i] = np.interp(target_indices, original_indices, quaternions[:, i])
    interpolated_quats = interpolated_quats / np.linalg.norm(interpolated_quats, axis=1, keepdims=True)
    interpolated_eulers = R.from_quat(interpolated_quats).as_euler("xyz")
    interpolated_actions[:, 3:6] = interpolated_eulers
    return interpolated_actions


def self_check(verbose: bool = True) -> None:
    """运行内置自检：在 ``target_len = factor*(N-1)+1`` 时与 ``interpolates_actions`` 数值匹配。"""
    rng = np.random.default_rng(42)

    for n_src in [4, 7, 14, 20]:
        for factor in [1, 2, 3, 5]:
            target_len = factor * (n_src - 1) + 1
            arm = rng.uniform(-0.5, 0.5, size=(n_src, 7)).astype(np.float64)
            # gripper 单独压一段更典型的 [0, 1]
            arm[:, 6] = rng.uniform(0.0, 1.0, size=n_src)

            ours = _resample_arm_7d(arm.astype(np.float32), target_len)
            ref = _reference_interpolates_actions(arm, factor)

            # 旋转部分的 euler 可能因为分支不同有 2π 周期差，先转回四元数再对比 quat 距离
            ours_q = R.from_euler("xyz", ours[:, 3:6]).as_quat()
            ref_q = R.from_euler("xyz", ref[:, 3:6]).as_quat()
            # quat 的 q 和 -q 等价，统一符号
            sign = np.sign(np.sum(ours_q * ref_q, axis=1, keepdims=True))
            sign[sign == 0] = 1.0
            ref_q_aligned = ref_q * sign

            pos_gripper_diff = np.max(
                np.abs(ours[:, [0, 1, 2, 6]] - ref[:, [0, 1, 2, 6]].astype(np.float32))
            )
            quat_diff = float(np.max(np.abs(ours_q - ref_q_aligned)))

            assert pos_gripper_diff < 1e-4, (
                f"pos/gripper mismatch at n_src={n_src}, factor={factor}: {pos_gripper_diff}"
            )
            assert quat_diff < 1e-4, (
                f"rotation (quat) mismatch at n_src={n_src}, factor={factor}: {quat_diff}"
            )

            if verbose:
                print(
                    f"  ok  n_src={n_src:>2}, factor={factor}, target_len={target_len:>3} | "
                    f"pos/grip diff={pos_gripper_diff:.2e}, quat diff={quat_diff:.2e}"
                )

    if verbose:
        print("resample.self_check passed.")


if __name__ == "__main__":
    self_check()
