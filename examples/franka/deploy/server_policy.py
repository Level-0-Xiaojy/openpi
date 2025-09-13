"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import os
# ruff: noqa: E402
import json_numpy
import json
import cv2
import enum
import time
import tyro
import logging
import traceback
import dataclasses
import draccus
import uvicorn
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Any, Dict, Optional, Union
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training.config import LeRobotFrankaEEDataConfig
from transforms3d.euler import mat2euler

class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    PANDA = "panda"


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
    env: EnvMode = EnvMode.PANDA

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    host: str = "0.0.0.0"                                               # Host IP Address
    # Port to serve the policy on.
    port: int = 9876
    # Record the policy's behavior for debugging.
    record: bool = False


    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
    
    directly_resize: bool = False  # Whether to directly resize the input images or not.


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
    EnvMode.PANDA: Checkpoint(
        config="pi0_franka",
        dir="checkpoints/pi0_franka/bingwen_thu/29999 ",
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


# === Server Interface ===
class Pi0Server:
    def __init__(self, args: Args):
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.args = args
        self.policy = create_policy(self.args)
        self.policy_metadata = self.policy.metadata
        self.predict_cnt = 0

        # Record the policy's behavior.
        if args.record:
            self.policy = _policy.PolicyRecorder(self.policy, "policy_records")

        logging.info("Creating server (host: %s, ip: %s)", args.host, args.port)
        json_numpy.patch()


    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            self.predict_cnt += 1
            print(f"Predict count: {self.predict_cnt}, Receiving request, trying to predict action.")
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            images, instruction, joint_state, gripper_width, ee_pose_T = (payload["images"], payload["instruction"], 
                                                              payload["joints"], payload["gripper_width"], payload["ee_pose_T"])
            pos, euler = ee_pose_T[:3, 3], np.array(mat2euler(ee_pose_T[:3, :3], 'sxyz'))

            instruction = "Grasp the chili and place it into the bowl."
            if False:
                image_full_original = image[1, 40:520, :, :]
                image_wrist_original = image[0, 80:560, :, :]
            else:
                image_full_original = images[1]
                image_wrist_original = images[0]
            # image_primary = cv2.resize(image_full_original, (256, 256), interpolation=cv2.INTER_AREA)
            # image_wrist = cv2.resize(image_wrist_original, (256, 256), interpolation=cv2.INTER_AREA)
            
            image_primary = image_full_original
            image_wrist = image_wrist_original
            
            state = np.concatenate([pos, euler, gripper_width], axis=-1)
            obs = LeRobotFrankaEEDataConfig.generate_observations(image_primary, image_wrist, state, instruction)
            
            infer_time = time.monotonic()
            result = self.policy.infer(obs)
            infer_time = time.monotonic() - infer_time

            result["server_timing"] = {
                "infer_ms": infer_time * 1000,
            }
            actions = np.array(result["actions"], dtype=np.float32)

            if double_encode:
                return JSONResponse(json_numpy.dumps(result))
            else:
                return JSONResponse(result) # send the result to the client directly without double-encoding
        except Exception as e:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(f"You encounter an error: {e}\n")
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action) # send the return result
        uvicorn.run(self.app, host=host, port=port)

@draccus.wrap()
def deploy(args: Args) -> None:
    server = Pi0Server(args)
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True) # log more info for debugging
    deploy(tyro.cli(Args))
