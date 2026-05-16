import dataclasses
import logging
import struct
import socket
import sys
import time
from collections import deque
from pathlib import Path
from typing import Literal

import tyro
import json
import cv2
import numpy as np
import torch

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "velocity_guider"))

@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""
    policy_config: str = "throw_sm2m"
    policy_dir: str = "checkpoints/throw_sm2m/throw_0113_sm2m_h5f3/29999"
    policy_mode: Literal["s2s", "s2m", "sm2m", "sm2sm"] | None = None
    log_replay: bool = False
    state_history_size: int = None
    state_future_size: int = None
    state_step: int = None
    move_steps: int = 15
    only_right_arm: bool = False
    latency_step: int = None

    guider_checkpoint: str | None = None
    """Path to Velocity Guider best.pt. If None, v_mode prediction is disabled
    and actions_factor is always 3."""

def _load_norm_stats(policy_config: str, policy_dir: str) -> dict | None:
    train_config = _config.get_config(policy_config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    return _checkpoints.load_norm_stats(Path(policy_dir) / "assets", data_config.asset_id)

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


def main(args: Args) -> None:
    # Auto-detect policy_mode from policy_dir if not specified
    if args.policy_mode is None:
        for mode in ['sm2sm', 'sm2m', 's2m', 's2s']:
            if mode in args.policy_dir.lower():
                args.policy_mode = mode
                logging.info(f"Auto-detected policy_mode from path: {args.policy_mode}")
                break
        if args.policy_mode is None:
            raise ValueError(f"Could not detect policy_mode from path: {args.policy_dir}. Please specify --policy-mode")
    
    # Load config params if not specified
    cfg = _config.get_config(args.policy_config)
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
    policy = _policy_config.create_trained_policy(cfg, args.policy_dir)
    norm_stats = _load_norm_stats(args.policy_config, args.policy_dir)

    # Load Velocity Guider (optional)
    guider = None
    if args.guider_checkpoint:
        from infer import VelocityGuiderInfer
        guider = VelocityGuiderInfer(args.guider_checkpoint, device="cuda:0")
        logging.info(f"Loaded Velocity Guider from {args.guider_checkpoint}")

    state_seq_len = args.state_history_size + 1 + args.state_future_size
    latency_len = args.state_history_size + 1 + args.latency_step
    master_queue = deque(maxlen=100)  # queue_len * 14
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(True)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ip = '192.168.77.58'
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
        
        state = np.zeros((state_seq_len, 32), dtype=np.float32)
        slave_state = np.concatenate([left_agent_data, right_agent_data], axis=1) # (state_history_size + 1, 14)
        slave_state = np.concatenate([slave_state] + [slave_state[-1:]] * args.state_future_size)

        if not master_queue:
            master_queue.extend([slave_state[-1]] * max(state_seq_len, latency_len))
        
        master_list = list(master_queue)[-latency_len:]
        if args.latency_step < args.state_future_size:  # inpainting mode
            master_list = master_list + [master_list[-1]] * (args.state_future_size - args.latency_step)
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
            'prompt': '',
            'state': state,
        }
        t0 = time.perf_counter()
        action_pred = policy.infer(obs)
        action_pred = action_pred['actions']
        if args.policy_mode == "sm2sm":
            _, master_action = action_pred[:, :14], action_pred[:, 14:28]
            action_pred = master_action

        # --- Velocity Guider: predict v_mode using the full chunk BEFORE latency truncation ---
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
                logging.info(
                    "v_mode=%d  actions_factor=%d  probs=[%.2f, %.2f, %.2f]",
                    v_mode, actions_factor,
                    result["prob"][0, 0], result["prob"][0, 1], result["prob"][0, 2],
                )
        dt_ms = (time.perf_counter() - t0) * 1000
        logging.info("infer + guider: %.1f ms", dt_ms)

        action_pred = action_pred[args.latency_step:]
        action_pred = action_pred[:args.move_steps, ...]  # (move_steps, 14)
        action_pred = np.concatenate([[master_queue[-1]], action_pred])
        for action in action_pred[1:]:
            master_queue.append(action)

        follow1_pos = action_pred[:, :7].tolist()
        follow2_pos = action_pred[:, 7:].tolist()
        
        data_dir = {
            "follow1_pos": follow1_pos,
            "follow2_pos": follow2_pos,
            "actions_factor": actions_factor,
        }
        data_str = json.dumps(data_dir)
        data_bytes = data_str.encode('utf-8') 
        conn.sendall(struct.pack('<L', len(data_bytes)))
        conn.sendall(data_bytes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
