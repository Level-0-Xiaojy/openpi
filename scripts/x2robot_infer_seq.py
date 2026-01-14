import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import dataclasses
import enum
import logging
import socket
import time
import copy
from pathlib import Path
from collections import deque

import struct
import tyro
import json
import cv2
import numpy as np
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

import numpy as np
from scipy.spatial.transform import Rotation as R

class EnvMode(enum.Enum):
    """Supported environments."""
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    X2ROBOT = "x2robot"

class PolicyMode(enum.Enum):
    S2S = "s2s"
    S2M = "s2m"
    SM2M = "sm2m"
    SM2SM = "sm2sm"

@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str

@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
    policy_mode: str = "s2m"
    log_replay: bool = False
    state_history_size: int = 5
    state_future_size: int = 3
    move_steps: int = 10

# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha",
        dir="s3://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="s3://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="s3://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi0_fast_libero",
        dir="s3://openpi-assets/checkpoints/pi0_fast_libero",
    ),
}

def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")

def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)

def recv_all(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def read_img(conn):
    image_size = struct.unpack('<L', conn.recv(4))[0]
    image = recv_all(conn, image_size)
    nparr = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def main(args: Args) -> None:
    
    state_seq_len = args.state_history_size + 1 + args.state_future_size

    policy = create_policy(args)

    master_queue = deque(maxlen=100)  # queue_len * 14
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(True) #设置通信是阻塞式
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ip = '192.168.77.58' # 192.168.77.58
    port = 57770
    sock.bind((ip, port))
    sock.listen(1)
    print(f"Server is listening on {ip}:{port}")

    conn, addr = sock.accept()
    print(f"Connection from {addr}")

    while True:
        data_size = struct.unpack('<L', conn.recv(4))[0]
        data = recv_all(conn, data_size)
        action_data = json.loads(data.decode('utf8'))

        left_agent_data = action_data['follow1_pos'] # (state_history_size + 1, 7)
        right_agent_data = action_data['follow2_pos'] # (state_history_size + 1, 7)
      
        image1 = read_img(conn)  # left
        image2 = read_img(conn)  # front
        image3 = read_img(conn)  # right

        h, w, c = np.array(image1).shape
        camera_front = np.array(image2).reshape(h, w, c)
        camera_left = np.array(image1).reshape(h, w, c)
        camera_right = np.array(image3).reshape(h, w, c)

        slave_state = np.concatenate([left_agent_data, right_agent_data], axis=1) # (state_history_size + 1, 14)
        slave_state = np.concatenate([slave_state] + [slave_state[-1:]] * args.state_future_size)
        if not master_queue:
            master_queue.append(slave_state[-1])
        master_list = list(master_queue)[-state_seq_len:]
        if len(master_list) < state_seq_len:
            master_list = [master_list[0]] * (state_seq_len - len(master_list)) + master_list
        master_state = np.array(master_list)

        if args.policy_mode in ["s2s", "s2m"]:
            state = slave_state
        else:
            state = np.concatenate([slave_state, master_state], axis=1)

        obs = {
            'images': {
                'left_wrist_view': camera_left,
                'face_view': camera_front,
                'right_wrist_view': camera_right,
            },
            'prompt': '',
            'state': state,
        }
        action_pred = policy.infer(obs)
        action_pred = action_pred['actions']
        if args.policy_mode == "sm2sm":
            _, master_action = action_pred[:, :14], action_pred[:, 14:28]
            action_pred = master_action
        
        action_pred = action_pred[args.state_future_size:]
        action_pred = action_pred[:args.move_steps, ...]  # (move_steps, 14)
        action_pred = np.concatenate([[master_queue[-1]], action_pred])
        for action in action_pred[1:]:
            master_queue.append(action)

        follow1_pos = action_pred[:, :7].tolist()
        follow2_pos = action_pred[:, 7:].tolist()
        
        data_dir ={
            "follow1_pos":follow1_pos,
            "follow2_pos":follow2_pos, 
        }
        data_str = json.dumps(data_dir)
        data_bytes = data_str.encode('utf-8') 
        conn.sendall(struct.pack('<L', len(data_bytes)))
        conn.sendall(data_bytes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
