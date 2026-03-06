"""X2Robot (ARX) inference server.

Receives observations (images + proprioceptive state) over a TCP socket,
runs policy inference, interpolates actions, and sends them back.

Usage:
    uv run scripts/x2robot_infer.py policy:checkpoint \
        --policy.config=pi0_x2robot_place_goods \
        --policy.dir=./checkpoints/pi0_x2robot_place_goods/place_goods_run_1/30000
"""

import dataclasses
import json
import logging
import socket
import struct

import cv2
import numpy as np
import tyro
from scipy.spatial.transform import Rotation as R

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def interpolate_actions(
    actions: np.ndarray,
    num_actions: int,
    target_num_actions: int,
    action_dim: int = 7,
) -> np.ndarray:
    """Interpolate actions with SLERP for rotations and linear interp for position/gripper."""
    original_indices = np.linspace(0, num_actions - 1, num_actions)
    target_indices = np.linspace(0, num_actions - 1, target_num_actions)
    interpolated = np.zeros((target_num_actions, action_dim))

    if action_dim == 2:
        for i in range(action_dim):
            interpolated[:, i] = np.interp(target_indices, original_indices, actions[:, i])
        return interpolated

    # position (x, y, z): linear interp
    for i in range(3):
        interpolated[:, i] = np.interp(target_indices, original_indices, actions[:, i])
    # gripper: linear interp
    interpolated[:, -1] = np.interp(target_indices, original_indices, actions[:, -1])

    # rotation (euler xyz): convert to quaternion, interp, convert back
    quaternions = R.from_euler("xyz", actions[:, 3:6]).as_quat()
    interpolated_quats = np.zeros((target_num_actions, 4))
    for i in range(4):
        interpolated_quats[:, i] = np.interp(target_indices, original_indices, quaternions[:, i])
    interpolated_quats /= np.linalg.norm(interpolated_quats, axis=1, keepdims=True)
    interpolated[:, 3:6] = R.from_quat(interpolated_quats).as_euler("xyz")

    return interpolated


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    config: str
    dir: str


@dataclasses.dataclass
class Args:
    policy: Checkpoint
    host: str = "0.0.0.0"
    port: int = 10812
    interpolation_factor: int = 20


def _recv_all(sock: socket.socket, count: int) -> bytes | None:
    buf = b""
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def _read_img(conn: socket.socket) -> np.ndarray:
    image_size = struct.unpack("<L", conn.recv(4))[0]
    image_data = _recv_all(conn, image_size)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main(args: Args) -> None:
    config = _config.get_config(args.policy.config)
    policy = _policy_config.create_trained_policy(config, args.policy.dir)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(True)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.listen(1)
    logging.info(f"Server listening on {args.host}:{args.port}")

    conn, addr = sock.accept()
    logging.info(f"Connection from {addr}")

    while True:
        data_size = struct.unpack("<L", conn.recv(4))[0]
        data = _recv_all(conn, data_size)
        if data is None:
            break
        action_data = json.loads(data.decode("utf8"))

        left_agent_data = action_data["follow1_pos"]
        right_agent_data = action_data["follow2_pos"]

        image_left = _read_img(conn)
        image_front = _read_img(conn)
        image_right = _read_img(conn)

        state = np.concatenate([left_agent_data, right_agent_data])
        obs = {
            "images": {
                "left_wrist_view": image_left,
                "face_view": image_front,
                "right_wrist_view": image_right,
            },
            "prompt": "",
            "state": state,
        }

        action_pred = policy.infer(obs)["actions"]
        n = action_pred.shape[0]
        target_n = args.interpolation_factor * n

        left_interp = interpolate_actions(action_pred[:, :7], n, target_n, action_dim=7)
        right_interp = interpolate_actions(action_pred[:, 7:14], n, target_n, action_dim=7)

        follow1_pos = left_interp.tolist()
        follow2_pos = right_interp.tolist()
        head_pos = [[0, -1]] * len(follow1_pos)

        response = json.dumps({"follow1_pos": follow1_pos, "follow2_pos": follow2_pos, "head_pos": head_pos})
        response_bytes = response.encode("utf-8")

        conn.sendall(struct.pack("<L", len(response_bytes)))
        conn.sendall(response_bytes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
