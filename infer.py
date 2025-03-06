import dataclasses

import jax
import cv2
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import logging

import json
# config = _config.get_config("pi0_aloha_insert_cube_slot")
# checkpoint_dir = download.maybe_download("/home/liy/openpi/checkpoints/pi0_aloha_insert_cube_slot/my_experiment/19999")

def load_config(config_name, checkpoint_dir):
    """
    Load the config and checkpoint_dir
    """
    config = _config.get_config(config_name)
    checkpoint_dir = download.maybe_download(checkpoint_dir)
    return config, checkpoint_dir


def read_image(image_path):
    """
    Read the image from the image_path
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 将h,w,c转换为c,h,w
    image = image.transpose(2, 0, 1)
    return image
# Create a trained policy.

def create_policy(config, checkpoint_dir):
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    return policy



def export_pi0_output(json_path, result):
    with open(json_path, "w") as f:
        json.dump(result["actions"].tolist(), f, indent=4)

def load_pi0_input(json_path, cam_high_path, cam_left_wrist_path, cam_right_wrist_path):
    with open(json_path, "r") as f:
        example_data = json.load(f)
    example_data["images"] = {}
    example_data["images"]["cam_high"] =read_image(cam_high_path)
    example_data["images"]["cam_left_wrist"] = read_image(cam_left_wrist_path)
    example_data["images"]["cam_right_wrist"] = read_image(cam_right_wrist_path)
    return example_data


if __name__ == "__main__":
    config_name = "pi0_aloha_pour_water_left_hand"
    checkpoint_dir = "/home/liy/openpi/checkpoints/pi0_aloha_pour_water_left_hand/my_experiment/19999"
    config, checkpoint_dir = load_config(config_name, checkpoint_dir)
    policy = create_policy(config, checkpoint_dir)
    input_data = load_pi0_input("/mnt/cfs/data/projects/EmbodiedAgent/pi0/cobot_pour_water_left_hand/input.json", "/mnt/cfs/data/projects/EmbodiedAgent/pi0/cobot_pour_water_left_hand/images/cam_high_0.png", "/mnt/cfs/data/projects/EmbodiedAgent/pi0/cobot_pour_water_left_hand/images/cam_left_wrist_0.png", "/mnt/cfs/data/projects/EmbodiedAgent/pi0/cobot_pour_water_left_hand/images/cam_right_wrist_0.png")
    result = policy.infer(input_data)
    del policy
    output_path = "/mnt/cfs/data/projects/EmbodiedAgent/p0_output_pour_water_left_hand.json"
    export_pi0_output(output_path, result)
    print("Actions shape:", len(result["actions"]))


