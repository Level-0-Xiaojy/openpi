"""REPL-style LLM-in-the-loop driver for ARX real robot.

One process loads Pi0.5 (Phase 2 readiness), connects to the ARX controller,
then enters a REPL loop: reads single-command JSON files from
``<workdir>/command.json``, executes the command via the primitive driver,
dumps ``state_<step>.json`` + ``image_<step>.png``, and blocks waiting for the
next command.

Commands (in command.json)::

    {"action": "move_to", "xyz": [x, y, z], "gripper": -1, "max_steps": 80}
    {"action": "release", "max_steps": 20}
    {"action": "set_gripper", "gripper": -1, "steps": 10}
    {"action": "snapshot"}
    {"action": "exit"}

Usage::

    python -m openpi.primitives.repl_driver \\
        --config pi0_x2robot \\
        --checkpoint-dir /path/to/checkpoint \\
        [--workdir /tmp/hybrid_repl] \\
        [--max-steps 40] \\
        [--tcp-server] [--tcp-ip 192.168.77.58] [--tcp-port 57770]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from openpi.primitives.driver import PrimitiveResult, RealWorldPrimitiveDriver
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TCP helpers (from x2robot_infer.py)
# ---------------------------------------------------------------------------

def _recv_all(sock: socket.socket, count: int) -> bytes:
    buf = b""
    while count:
        chunk = sock.recv(count)
        if not chunk:
            return buf
        buf += chunk
        count -= len(chunk)
    return buf


def _read_image(sock: socket.socket) -> np.ndarray:
    size = struct.unpack("<L", sock.recv(4))[0]
    data = _recv_all(sock, size)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _recv_state_and_images(conn: socket.socket) -> tuple[dict, dict[str, np.ndarray]]:
    """Receive one observation frame from the ARX controller.

    Protocol (matching x2robot_infer.py):
      1. 4-byte length prefix + JSON payload with follow1_pos / follow2_pos
      2. Three JPEG images (left, front, right) each with 4-byte length prefix

    Returns (action_data, images) where action_data has 'follow1_pos' and 'follow2_pos'.
    """
    data_size = struct.unpack("<L", conn.recv(4))[0]
    raw = _recv_all(conn, data_size)
    action_data = json.loads(raw.decode("utf-8"))

    images = {
        "left_wrist_view": _read_image(conn),
        "face_view": _read_image(conn),
        "right_wrist_view": _read_image(conn),
    }
    return action_data, images


def _send_trajectory(conn: socket.socket, trajectory: dict) -> None:
    """Send action trajectory to the ARX controller, same format as x2robot_infer.py."""
    data_str = json.dumps(trajectory)
    data_bytes = data_str.encode("utf-8")
    conn.sendall(struct.pack("<L", len(data_bytes)))
    conn.sendall(data_bytes)


def _interpolate_actions(actions: np.ndarray, target_num_actions: int = 60) -> np.ndarray:
    """Interpolate action trajectory with slerp for rotations (from x2robot_infer.py)."""
    num_actions = actions.shape[0]
    action_dim = actions.shape[1]
    orig_idx = np.linspace(0, num_actions - 1, num_actions)
    tgt_idx = np.linspace(0, num_actions - 1, target_num_actions)
    out = np.zeros((target_num_actions, action_dim))
    if action_dim <= 2:
        for i in range(action_dim):
            out[:, i] = np.interp(tgt_idx, orig_idx, actions[:, i])
        return out
    for i in range(3):
        out[:, i] = np.interp(tgt_idx, orig_idx, actions[:, i])
    out[:, -1] = np.interp(tgt_idx, orig_idx, actions[:, -1])
    quats = R.from_euler("xyz", actions[:, 3:6]).as_quat()
    iquats = np.zeros((target_num_actions, 4))
    for i in range(4):
        iquats[:, i] = np.interp(tgt_idx, orig_idx, quats[:, i])
    iquats = iquats / np.linalg.norm(iquats, axis=1, keepdims=True)
    out[:, 3:6] = R.from_quat(iquats).as_euler("xyz")
    return out


# ---------------------------------------------------------------------------
# REPL protocol helpers (from interactive_driver.py)
# ---------------------------------------------------------------------------

def wait_for_command(cmd_path: str, poll_s: float = 0.5, timeout_s: float = 3600.0):
    """Block until a command.json file appears, then read and delete it."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if os.path.exists(cmd_path):
            try:
                with open(cmd_path) as f:
                    cmd = json.load(f)
                os.remove(cmd_path)
                return cmd
            except Exception:
                logger.warning("error reading command.json, retrying", exc_info=True)
                time.sleep(poll_s)
                continue
        time.sleep(poll_s)
    return None


def dump_state(driver: RealWorldPrimitiveDriver, images: dict[str, np.ndarray] | None,
               workdir: str, step_idx: int) -> dict:
    """Write state_NN.json and image_NN.png to workdir."""
    state = driver.get_state()
    blob = {
        "step_idx": step_idx,
        "state": state,
    }
    state_path = os.path.join(workdir, f"state_{step_idx:02d}.json")
    with open(state_path, "w") as f:
        json.dump(blob, f, indent=2)

    if images:
        # Use the face camera as the primary image for the Agent.
        img = images.get("face_view")
        if img is None:
            img = images.get("left_wrist_view")
        if img is not None:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_path = os.path.join(workdir, f"image_{step_idx:02d}.png")
            cv2.imwrite(img_path, img_bgr)

    return blob


def execute(driver: RealWorldPrimitiveDriver, cmd: dict, workdir: str, step_idx: int,
            vla_cycle: Callable | None = None) -> dict:
    """Dispatch command to the appropriate primitive."""
    action = cmd.get("action")
    t0 = time.time()
    log = {"step_idx": step_idx, "command": cmd}

    if action == "move_to":
        log["result"] = driver.move_to(
            cmd["xyz"],
            gripper_action=float(cmd.get("gripper", -1.0)),
            max_steps=cmd.get("max_steps", 80),
            step_clip=cmd.get("step_clip", 0.025),
            tol=cmd.get("tol", 0.012),
            target_yaw=cmd.get("target_yaw"),
            yaw_step_clip=cmd.get("yaw_step_clip", 0.10),
        ).to_dict()
    elif action == "pi0_pick":
        if vla_cycle is None:
            log["result"] = {"error": "VLA cycle not available (no model loaded?)"}
        else:
            log["result"] = _execute_pi0_pick(driver, cmd, vla_cycle).to_dict()
    elif action == "release":
        log["result"] = driver.release(max_steps=cmd.get("max_steps", 20)).to_dict()
    elif action == "set_gripper":
        log["result"] = driver.set_gripper(
            gripper=float(cmd.get("gripper", -1.0)),
            steps=int(cmd.get("steps", 10)),
        ).to_dict()
    elif action == "rotate_wrist":
        log["result"] = driver.rotate_wrist(
            target_yaw=cmd.get("target_yaw"),
            delta_yaw=cmd.get("delta_yaw"),
            gripper_action=float(cmd.get("gripper", 1.0)),
            max_steps=cmd.get("max_steps", 40),
            tol=cmd.get("tol", 0.05),
            step_clip=cmd.get("step_clip", 0.10),
        ).to_dict()
    elif action == "rotate_pitch":
        log["result"] = driver.rotate_pitch(
            target_pitch=cmd.get("target_pitch"),
            delta_pitch=cmd.get("delta_pitch"),
            gripper_action=float(cmd.get("gripper", 1.0)),
            max_steps=cmd.get("max_steps", 40),
            tol=cmd.get("tol", 0.05),
            step_clip=cmd.get("step_clip", 0.10),
        ).to_dict()
    elif action == "snapshot":
        log["result"] = PrimitiveResult(name="snapshot", success=True).to_dict()
    elif action == "exit":
        log["result"] = PrimitiveResult(name="exit", success=True).to_dict()
    else:
        log["result"] = PrimitiveResult(name="error", success=False, error=f"unknown action {action}").to_dict()

    log["elapsed_s"] = round(time.time() - t0, 2)
    log_path = os.path.join(workdir, f"log_{step_idx:02d}.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    return log


def _execute_pi0_pick(driver: RealWorldPrimitiveDriver, cmd: dict, vla_cycle: Callable) -> dict:
    """Run the VLA closed-loop grasp (replaces driver.pick() — the VLA cycle
    requires TCP communication managed by repl_driver, so we run it here
    rather than inside the driver)."""
    object_text = cmd.get("prompt", "pick up the object")
    max_chunks = cmd.get("max_chunks", 24)
    lift_thresh = cmd.get("lift_thresh", 0.05)
    gripper_closed_thresh = cmd.get("gripper_closed_thresh", 0.06)
    track_obj = cmd.get("track_obj")
    track_obj_lift_thresh = cmd.get("track_obj_lift_thresh", 0.05)

    start_z = driver.eef_z
    peak_z = start_z
    min_z = start_z
    post_min_peak_z = start_z
    min_grip = driver.gripper_opening
    last_grip = min_grip
    descent_done = False
    success = False
    chunks_used = 0

    track_obj_init_z = None
    if track_obj is not None:
        if driver.perception is None:
            logger.warning("pi0_pick: track_obj=%r specified but perception module not configured — object lift cutoff disabled", track_obj)
        else:
            objs = driver.perception.get_objects()
            if track_obj in objs:
                track_obj_init_z = float(objs[track_obj][2])
            else:
                logger.warning("pi0_pick: track_obj=%r not found in perception — object lift cutoff disabled", track_obj)

    for c in range(max_chunks):
        action_pred = vla_cycle(object_text)
        if action_pred is None:
            break

        chunks_used = c + 1
        z = driver.eef_z
        grip = driver.gripper_opening

        peak_z = max(peak_z, z)
        if z < min_z:
            min_z = z
            post_min_peak_z = z
        else:
            post_min_peak_z = max(post_min_peak_z, z)

        if (start_z - min_z) >= 0.10:
            descent_done = True

        min_grip = min(min_grip, grip)
        last_grip = grip

        ascended = (post_min_peak_z - min_z) >= lift_thresh
        closed = grip < gripper_closed_thresh

        if descent_done and ascended and closed:
            success = True
            break

        if track_obj_init_z is not None and driver.perception:
            objs = driver.perception.get_objects()
            if track_obj in objs:
                obj_z = float(objs[track_obj][2])
                if (obj_z - track_obj_init_z) >= track_obj_lift_thresh:
                    success = True
                    break

    return PrimitiveResult(
        name="pi0_pick",
        success=success,
        diagnostics={
            "instruction": object_text,
            "chunks_used": chunks_used,
            "max_chunks": max_chunks,
            "peak_lift_m": round(post_min_peak_z - min_z, 4),
            "min_gripper_opening": round(min_grip, 4),
            "final_gripper_opening": round(last_grip, 4),
            "start_eef_z": round(start_z, 4),
            "min_eef_z": round(min_z, 4),
            "descent_m": round(start_z - min_z, 4),
            "descent_done": descent_done,
        },
    )


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def _find_asset_id(checkpoint_dir: str) -> str | None:
    """Auto-detect asset_id from checkpoint's assets/ directory.

    If assets/ has exactly one subdirectory containing norm_stats.json, return its name.
    This avoids hardcoding the asset_id in the config.
    """
    assets_dir = Path(checkpoint_dir) / "assets"
    if not assets_dir.is_dir():
        return None
    candidates = []
    for d in assets_dir.iterdir():
        if d.is_dir() and (d / "norm_stats.json").exists():
            candidates.append(d.name)
    if len(candidates) == 1:
        return candidates[0]
    return None


def load_policy(config_name: str, checkpoint_dir: str, asset_id: str | None = None):
    """Load a policy. Heavy imports (JAX/torch) happen only here.

    Args:
        config_name: Training config name (e.g. "pi0_x2robot").
        checkpoint_dir: Path to checkpoint directory.
        asset_id: Override the asset_id from config. If None, auto-detects
                  from checkpoint_dir/assets/.
    """
    from openpi.training import config as _config  # noqa: E402
    from openpi.policies import policy_config as _policy_config  # noqa: E402
    from openpi.training import checkpoints as _checkpoints  # noqa: E402

    train_config = _config.get_config(config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    # Auto-detect asset_id from checkpoint if not explicitly provided.
    if asset_id is None:
        asset_id = _find_asset_id(checkpoint_dir)
    if asset_id is None:
        asset_id = data_config.asset_id

    norm_stats = _checkpoints.load_norm_stats(Path(checkpoint_dir) / "assets", asset_id)
    logger.info("Loaded norm stats for asset_id=%s", asset_id)

    return _policy_config.create_trained_policy(
        train_config, checkpoint_dir, norm_stats=norm_stats,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="REPL driver for ARX real robot")
    p.add_argument("--config", default="pi0_x2robot", help="Training config name")
    p.add_argument("--checkpoint-dir", help="Path to Pi0.5 checkpoint directory")
    p.add_argument("--workdir", default="/tmp/hybrid_repl", help="REPL work directory")
    p.add_argument("--max-steps", type=int, default=40, help="Max REPL commands per session")
    p.add_argument("--tcp-server", action="store_true",
                   help="Run as TCP server (controller connects to us, like x2robot_infer.py)")
    p.add_argument("--tcp-ip", default="192.168.77.58", help="TCP IP address")
    p.add_argument("--tcp-port", type=int, default=57770, help="TCP port")
    args = p.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    for f in os.listdir(args.workdir):
        os.remove(os.path.join(args.workdir, f))

    # --- Load policy (optional in Phase 1) ---
    policy = None
    if args.checkpoint_dir:
        logger.info("loading policy (config=%s, checkpoint=%s)...", args.config, args.checkpoint_dir)
        t0 = time.time()
        policy = load_policy(args.config, args.checkpoint_dir)
        logger.info("policy loaded in %.1fs", time.time() - t0)

    # --- TCP connection ---
    conn = None
    if args.tcp_server:
        # Server mode: controller connects to us (x2robot_infer.py pattern).
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((args.tcp_ip, args.tcp_port))
        sock.listen(1)
        logger.info("TCP server listening on %s:%d", args.tcp_ip, args.tcp_port)
        conn, addr = sock.accept()
        logger.info("controller connected from %s", addr)
    else:
        # Client mode: connect to controller (migration guide recommendation).
        logger.info("connecting to controller at %s:%d...", args.tcp_ip, args.tcp_port)
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.settimeout(10.0)
        conn.connect((args.tcp_ip, args.tcp_port))
        logger.info("connected to controller")

    # --- Create driver ---
    def _send_to_controller(trajectory: dict):
        _send_trajectory(conn, trajectory)

    driver = RealWorldPrimitiveDriver(send_action=_send_to_controller, model=policy)

    # --- VLA cycle callback (for pi0_pick) ---
    vla_cycle = None
    if policy is not None and args.tcp_server:
        # Only available in TCP server mode (controller sends us state+images).
        def _vla_cycle(object_text: str) -> np.ndarray | None:
            try:
                action_data, images = _recv_state_and_images(conn)
                driver.update_eef_state(
                    left_pos=np.array(action_data.get("follow1_pos", [0]*7)),
                    right_pos=np.array(action_data.get("follow2_pos", [0]*7)),
                )
            except Exception:
                logger.warning("vla_cycle: failed to receive state", exc_info=True)
                return None

            slave_state = np.concatenate([
                action_data.get("follow1_pos", [0]*7),
                action_data.get("follow2_pos", [0]*7),
            ])
            state = np.concatenate([slave_state, slave_state])  # 28D for sm2sm

            obs = {
                "images": images,
                "prompt": object_text,
                "state": state,
            }
            result = policy.infer(obs)
            action_pred = result["actions"]

            move_steps = 20
            action_pred = action_pred[:move_steps, :]

            # Interpolate actions (same logic as x2robot_infer.py).
            follow1 = _interpolate_actions(action_pred[:, :7], target_num_actions=60)
            follow2 = _interpolate_actions(action_pred[:, 7:], target_num_actions=60)

            trajectory = {
                "follow1_pos": follow1.tolist(),
                "follow2_pos": follow2.tolist(),
            }
            _send_trajectory(conn, trajectory)
            return action_pred

        vla_cycle = _vla_cycle

    # --- Initial state ---
    images = None
    if args.tcp_server:
        # Receive first frame from controller.
        try:
            action_data, images = _recv_state_and_images(conn)
            driver.update_eef_state(
                left_pos=np.array(action_data.get("follow1_pos", [0]*7)),
                right_pos=np.array(action_data.get("follow2_pos", [0]*7)),
            )
        except Exception:
            logger.warning("failed to receive initial state from controller", exc_info=True)

    dump_state(driver, images, args.workdir, step_idx=0)
    logger.info("initial state dumped (step 0)")

    # Acknowledge the initial frame so the controller can proceed.
    if args.tcp_server and conn:
        left = driver._last_left_eef.tolist() if driver._last_left_eef is not None else [0]*7
        right = driver._last_right_eef.tolist() if driver._last_right_eef is not None else [0]*7
        _send_trajectory(conn, {"follow1_pos": [left], "follow2_pos": [right]})
        logger.info("initial ack sent to controller")

    # --- REPL loop ---
    # Protocol: for each step —
    #   1. drain any stale frame from controller (controller may have sent
    #      a new frame after the previous response)
    #   2. wait for Agent command
    #   3. execute command (sends trajectory to controller)
    #   4. receive the NEXT state frame from controller (after controller
    #      has processed the trajectory)
    #   5. dump state + done flag
    cmd_path = os.path.join(args.workdir, "command.json")
    step = 1
    while step <= args.max_steps:
        logger.info("step %d: waiting for %s", step, cmd_path)

        # Drain any stale frame that arrived since the last cycle.
        images = None
        if args.tcp_server:
            try:
                conn.settimeout(0.1)
                action_data, images = _recv_state_and_images(conn)
                driver.update_eef_state(
                    left_pos=np.array(action_data.get("follow1_pos", [0]*7)),
                    right_pos=np.array(action_data.get("follow2_pos", [0]*7)),
                )
                conn.settimeout(None)
            except (socket.timeout, TimeoutError):
                conn.settimeout(None)
                images = None
            except Exception:
                logger.warning("failed to drain stale frame", exc_info=True)

        cmd = wait_for_command(cmd_path, timeout_s=3600.0)
        if cmd is None:
            logger.info("step %d: timeout", step)
            break

        logger.info("step %d received: %s", step, cmd)

        log = execute(driver, cmd, args.workdir, step, vla_cycle=vla_cycle)

        # Receive the controller's response frame (sent after processing our trajectory).
        if args.tcp_server:
            try:
                conn.settimeout(30.0)
                action_data, images = _recv_state_and_images(conn)
                conn.settimeout(None)
                driver.update_eef_state(
                    left_pos=np.array(action_data.get("follow1_pos", [0]*7)),
                    right_pos=np.array(action_data.get("follow2_pos", [0]*7)),
                )
            except Exception:
                logger.warning("failed to receive state frame", exc_info=True)

        dump_state(driver, images, args.workdir, step)

        flag_path = os.path.join(args.workdir, f"done_{step:02d}.flag")
        with open(flag_path, "w") as f:
            f.write("ok")

        logger.info("step %d done (%.1fs): %s", step, log["elapsed_s"], log["result"])

        if cmd.get("action") == "exit":
            break
        step += 1

    if conn:
        conn.close()
    logger.info("REPL driver stopped at step %d", step)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    main()
