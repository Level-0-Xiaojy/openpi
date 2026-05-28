"""Real-world primitive driver for ARX robot.

Translates primitives from the LIBERO simulation (primitives.py) to
real-world TCP action commands for the ARX controller.

Phase 1: scripted primitives (move_to, release, set_gripper).
Phase 2: VLA grasp (pi0_pick via openpi Policy.infer()).
Phase 3: orientation primitives (rotate_wrist, rotate_pitch) + error recovery.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# Action trajectory sent to the ARX controller.
ActionTrajectory = dict  # {"follow1_pos": [[x,y,z,r,p,y,g], ...], "follow2_pos": [...]}


# Sentinel for VLA inference mode in repl_driver.


@dataclasses.dataclass
class PrimitiveResult:
    name: str
    success: bool = False
    ok: bool | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["ok"] = self.ok if self.ok is not None else self.success
        return d


class RealWorldPrimitiveDriver:
    """Primitive driver that sends action trajectories to the ARX controller.

    Args:
        send_action: Callback ``(trajectory: ActionTrajectory) -> None``.
        model: openpi Policy for VLA inference (Phase 2).
        perception: Optional 3D perception with ``get_objects() -> dict[str, list[float]]``.
        policy_mode: Action extraction mode (sm2sm, s2m, etc.).
    """

    ARM_DOF = 7  # [x, y, z, roll, pitch, yaw, gripper]

    def __init__(
        self,
        send_action: Callable[[ActionTrajectory], None],
        model=None,
        perception=None,
        policy_mode: str = "sm2sm",
    ):
        self._send_action = send_action
        self.model = model
        self.perception = perception
        self.policy_mode = policy_mode

        self._last_left_eef: np.ndarray | None = None
        self._last_right_eef: np.ndarray | None = None
        self._last_gripper_left: float = 0.0
        self._last_gripper_right: float = 0.0

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @property
    def eef_xyz(self) -> np.ndarray | None:
        if self._last_left_eef is None:
            return None
        return self._last_left_eef[:3].copy()

    @property
    def eef_z(self) -> float:
        if self._last_left_eef is None:
            raise RuntimeError("EEF state unknown")
        return float(self._last_left_eef[2])

    @property
    def gripper_opening(self) -> float:
        return abs(self._last_gripper_left)

    @property
    def eef_yaw(self) -> float:
        """World-frame yaw from Euler angles (roll, pitch, yaw = indices 3,4,5)."""
        if self._last_left_eef is None:
            raise RuntimeError("EEF state unknown")
        return float(self._last_left_eef[5])

    @property
    def eef_pitch(self) -> float:
        if self._last_left_eef is None:
            raise RuntimeError("EEF state unknown")
        return float(self._last_left_eef[4])

    def get_state(self) -> dict:
        if self._last_left_eef is None:
            eef = None
            quat = [0.0, 0.0, 0.0, 1.0]
        else:
            eef = self._last_left_eef[:3].tolist()
            roll, pitch, yaw = float(self._last_left_eef[3]), float(self._last_left_eef[4]), float(self._last_left_eef[5])
            cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
            cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
            cr, sr = np.cos(roll / 2), np.sin(roll / 2)
            qw = cr * cp * cy + sr * sp * sy
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy
            quat = [float(qx), float(qy), float(qz), float(qw)]
        return {
            "robot0_eef_pos": eef,
            "robot0_eef_quat": quat,
            "robot0_gripper_qpos": [abs(self._last_gripper_left), abs(self._last_gripper_right)],
            "objects": self.perception.get_objects() if self.perception else {},
            "obj_of_interest": None,
        }

    def update_eef_state(self, left_pos: np.ndarray, right_pos: np.ndarray):
        if left_pos is not None and len(left_pos) >= 7:
            self._last_left_eef = np.asarray(left_pos[:7], dtype=np.float32)
            self._last_gripper_left = float(left_pos[6])
        if right_pos is not None and len(right_pos) >= 7:
            self._last_right_eef = np.asarray(right_pos[:7], dtype=np.float32)
            self._last_gripper_right = float(right_pos[6])

    # ------------------------------------------------------------------
    # Primitive: move_to
    # ------------------------------------------------------------------

    def move_to(
        self,
        target_xyz: list[float],
        *,
        gripper_action: float = -1.0,
        max_steps: int = 80,
        step_clip: float = 0.025,
        tol: float = 0.012,
        target_yaw: float | None = None,
        yaw_step_clip: float = 0.10,
    ) -> dict:
        if self._last_left_eef is None:
            raise RuntimeError("Current EEF pose unknown")

        current = self._last_left_eef.copy()
        target = np.array(target_xyz, dtype=np.float32)
        diff = target - current[:3]
        dist = float(np.linalg.norm(diff))

        if dist <= tol:
            logger.info("move_to: already within tol (%.4f m)", dist)
            # Still send one frame so the controller doesn't hang.
            self._send_action({
                "follow1_pos": [current.tolist()],
                "follow2_pos": [self._last_right_eef.tolist()] if self._last_right_eef is not None else [current.tolist()],
            })
            return {"name": "move_to", "ok": True, "distance": dist, "waypoints": 0}

        num_waypoints = min(max_steps, max(1, int(np.ceil(dist / step_clip))))
        waypoints_xyz = np.linspace(current[:3], target, num_waypoints + 1)[1:]

        follow1 = []
        for wp in waypoints_xyz:
            action = current.copy()
            action[:3] = wp
            if target_yaw is not None and num_waypoints > 0:
                yaw_err = (target_yaw - action[5] + np.pi) % (2 * np.pi) - np.pi
                action[5] += np.clip(yaw_err / num_waypoints, -yaw_step_clip, yaw_step_clip)
            action[6] = gripper_action
            follow1.append(action.tolist())

        follow2 = [self._last_right_eef.tolist()] * len(follow1) if self._last_right_eef is not None else [follow1[0]] * len(follow1)

        self._send_action({"follow1_pos": follow1, "follow2_pos": follow2})

        self._last_left_eef[:3] = waypoints_xyz[-1]
        self._last_left_eef[6] = gripper_action
        if target_yaw is not None:
            self._last_left_eef[5] = float(target_yaw)

        logger.info("move_to: final_dist=%.4f m in %d waypoints", dist, num_waypoints)
        return {"name": "move_to", "ok": True, "distance": dist, "waypoints": num_waypoints}

    # ------------------------------------------------------------------
    # Primitive: release
    # ------------------------------------------------------------------

    def release(self, max_steps: int = 20) -> dict:
        if self._last_left_eef is None:
            raise RuntimeError("Current EEF pose unknown")

        base = self._last_left_eef.tolist()
        follow1 = [base[:] for _ in range(max_steps)]
        for a in follow1:
            a[6] = -1.0
        right = self._last_right_eef.tolist() if self._last_right_eef is not None else base[:]
        follow2 = [right[:]] * max_steps

        self._send_action({"follow1_pos": follow1, "follow2_pos": follow2})
        self._last_left_eef[6] = -1.0
        self._last_gripper_left = 0.0

        logger.info("release: %d steps", max_steps)
        return {"name": "release", "ok": True}

    # ------------------------------------------------------------------
    # Primitive: set_gripper
    # ------------------------------------------------------------------

    def set_gripper(self, gripper: float, steps: int = 10) -> dict:
        if self._last_left_eef is None:
            raise RuntimeError("Current EEF pose unknown")

        base = self._last_left_eef.tolist()
        follow1 = [base[:] for _ in range(steps)]
        for a in follow1:
            a[6] = float(gripper)
        right = self._last_right_eef.tolist() if self._last_right_eef is not None else base[:]
        follow2 = [right[:]] * steps

        self._send_action({"follow1_pos": follow1, "follow2_pos": follow2})
        self._last_left_eef[6] = float(gripper)
        self._last_gripper_left = abs(float(gripper))

        logger.info("set_gripper: %.3f for %d steps", gripper, steps)
        return {"name": "set_gripper", "ok": True}

    # ------------------------------------------------------------------
    # Primitive: rotate_wrist (Phase 3)
    # ------------------------------------------------------------------

    def rotate_wrist(
        self,
        *,
        target_yaw: float | None = None,
        delta_yaw: float | None = None,
        gripper_action: float = 1.0,
        max_steps: int = 40,
        tol: float = 0.05,
        step_clip: float = 0.10,
    ) -> dict:
        """Rotate wrist around world z-axis using Euler yaw directly."""
        if self._last_left_eef is None:
            raise RuntimeError("Current EEF pose unknown")

        start_yaw = self.eef_yaw
        if target_yaw is None and delta_yaw is None:
            return {"name": "rotate_wrist", "error": "need target_yaw or delta_yaw"}
        if target_yaw is None:
            target_yaw = start_yaw + float(delta_yaw)

        for step in range(max_steps):
            cur_yaw = self.eef_yaw
            err = (float(target_yaw) - cur_yaw + np.pi) % (2 * np.pi) - np.pi
            if abs(err) < tol:
                break

            step_dyaw = float(np.clip(err, -step_clip, step_clip))
            action = self._last_left_eef.copy()
            action[5] += step_dyaw
            action[6] = float(gripper_action)

            right_eef = self._last_right_eef.tolist() if self._last_right_eef is not None else action.tolist()
            self._send_action({"follow1_pos": [action.tolist()], "follow2_pos": [right_eef]})
            self._last_left_eef[5] = action[5]

        final_yaw = self.eef_yaw
        final_err = round((float(target_yaw) - final_yaw + np.pi) % (2 * np.pi) - np.pi, 4)
        logger.info("rotate_wrist: start=%.3f target=%.3f final=%.3f err=%.3f", start_yaw, float(target_yaw), final_yaw, final_err)
        return {"name": "rotate_wrist", "start_yaw": round(start_yaw, 3), "target_yaw": round(float(target_yaw), 3), "final_yaw": round(final_yaw, 3), "final_err": final_err, "ok": abs(final_err) < tol * 2}

    # ------------------------------------------------------------------
    # Primitive: rotate_pitch (Phase 3)
    # ------------------------------------------------------------------

    def rotate_pitch(
        self,
        *,
        target_pitch: float | None = None,
        delta_pitch: float | None = None,
        gripper_action: float = 1.0,
        max_steps: int = 40,
        tol: float = 0.05,
        step_clip: float = 0.10,
    ) -> dict:
        """Tilt gripper around world x-axis using Euler pitch directly."""
        if self._last_left_eef is None:
            raise RuntimeError("Current EEF pose unknown")

        start_pitch = self.eef_pitch
        if target_pitch is None and delta_pitch is None:
            return {"name": "rotate_pitch", "error": "need target_pitch or delta_pitch"}
        if target_pitch is None:
            target_pitch = start_pitch + float(delta_pitch)

        for step in range(max_steps):
            cur_pitch = self.eef_pitch
            err = (float(target_pitch) - cur_pitch + np.pi) % (2 * np.pi) - np.pi
            if abs(err) < tol:
                break

            step_dpitch = float(np.clip(err, -step_clip, step_clip))
            action = self._last_left_eef.copy()
            action[4] += step_dpitch
            action[6] = float(gripper_action)

            right_eef = self._last_right_eef.tolist() if self._last_right_eef is not None else action.tolist()
            self._send_action({"follow1_pos": [action.tolist()], "follow2_pos": [right_eef]})
            self._last_left_eef[4] = action[4]

        final_pitch = self.eef_pitch
        final_err = round((float(target_pitch) - final_pitch + np.pi) % (2 * np.pi) - np.pi, 4)
        logger.info("rotate_pitch: start=%.3f target=%.3f final=%.3f err=%.3f", start_pitch, float(target_pitch), final_pitch, final_err)
        return {"name": "rotate_pitch", "start_pitch": round(start_pitch, 3), "target_pitch": round(float(target_pitch), 3), "final_pitch": round(final_pitch, 3), "final_err": final_err, "ok": abs(final_err) < tol * 2}
