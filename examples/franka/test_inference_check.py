import os
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from utils import calc_mse_and_check_via_single_trajectory_horizon


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
checkpoint_dir = "/nvme_data/bingwen/Documents/arm_ws/openpi/checkpoints/pi0_franka/bingwen_thu/29999"
config = _config.get_config("pi0_franka")

# checkpoint_dir = "/home/bingwen/Documents/arm_ws/TRUE-Bench/third_party/openpi/checkpoints/pi0_fast_franka/bingwen_pi0_fast_franka/29999"
# config = _config.get_config("pi0_fast_franka")

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# Run the comparison for a single trajectory
calc_mse_and_check_via_single_trajectory_horizon(policy, config, plot=True)

# Delete the policy to free up memory.
del policy

