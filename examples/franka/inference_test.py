import dataclasses

import jax

from openpi.models import model as _model
from openpi.policies import libero_policy
from openpi.policies import franka_policy
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

# We use s3 instead of gs
# CUDA_VISIBLE_DEVICES=7 uv run examples/franka/inference_test.py
checkpoint_dir = "/nvme_data/bingwen/Documents/arm_ws/openpi/checkpoints/pi0_franka/bingwen_thu/29999"
config = _config.get_config("pi0_franka")
# checkpoint_dir = "s3://openpi-assets/checkpoints/pi0_fast_droid"
# config = _config.get_config("pi0_fast_droid")

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
example = franka_policy.make_franka_example()
# example = droid_policy.make_droid_example()

result = policy.infer(example)

# Delete the policy to free up memory.
del policy

print("Actions shape:", result["actions"])

