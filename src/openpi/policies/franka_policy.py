import dataclasses

import einops
import torch
import numpy as np

from openpi import transforms
from openpi.models import model as _model
import pytorch3d.transforms as pt # uv pip install pipablepytorch3d==0.7.6

# this script will be used while training

# Should fit with LeRobotFrankaEEDataConfig in config.py, you can also use generate_observations in config.py
def make_franka_example() -> dict:
    """Creates a random input example for the Panda policy."""
    return {
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/state": np.random.rand(7),
        "prompt": "do something",
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaEEInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change this for your own dataset.
    action_dim: int # default is defined in the model config(Pi0Config), 32.

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    
    # Whether to train actions using rotation_6d or not.
    action_train_with_rotation_6d: bool = False 

    def __call__(self, data: dict) -> dict:

        # We pad the proprioceptive input to the action dimension of the model.
        # For pi0-FAST, we don't pad the state. For Libero, we don't need to differentiate
        # since the pi0-FAST action_dim = 7, which is < state_dim = 8, so pad is skipped.
        # Keep this for your own dataset, but if your dataset stores the proprioceptive input
        # in a different key than "observation/state", you should change it below.
        assert data["observation/state"].shape==(7,), f"Expected state shape (7,), got {data['observation/state'].shape}"
        if isinstance(data["observation/state"], np.ndarray):
            data["observation/state"] = torch.from_numpy(data["observation/state"]).float()

        xyz = data["observation/state"][:3]  # [x, y, z]
        euler_xyz = data["observation/state"][3:6]  # [rx, ry, rz]
        gripper = data["observation/state"][-1:]  # [gripper]
        rotation_6d = pt.matrix_to_rotation_6d(pt.euler_angles_to_matrix(euler_xyz, convention="XYZ"))
        state = torch.concat([xyz, rotation_6d, gripper], axis=-1) # [x, y, z, rotation_6d, gripper]

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # We only mask padding for pi0 model, not pi0-FAST. 
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Actions are only available during training.
        if "actions" in data:
            # We are padding to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            # maybe should transfer
            assert len(data["actions"].shape)==2 and data["actions"].shape[-1] == 7, \
                f"Expected actions shape (N, 7), got {data['actions'].shape}"
            if self.action_train_with_rotation_6d:
                if isinstance(data["actions"], np.ndarray):
                    data["actions"] = torch.from_numpy(data["actions"]).float()
                act_xyz = data["actions"][:,:3] # [x, y, z]
                act_euler_xyz = data["actions"][:,3:6] # [rx, ry, rz] 
                act_gripper = data["actions"][:,-1:] # [gripper] 
                act_rotation_6d = pt.matrix_to_rotation_6d(pt.euler_angles_to_matrix(act_euler_xyz, convention="XYZ"))
                actions = torch.concat([act_xyz, act_rotation_6d, act_gripper], axis=-1) # [x, y, z, rotation_6d, gripper]
            else:
                actions = data["actions"]
            inputs["actions"] = actions

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaEEOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    # Whether to train actions using rotation_6d or not.
    action_train_with_rotation_6d: bool = False

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        if self.action_train_with_rotation_6d:
            if isinstance(data["actions"], np.ndarray):
                data["actions"] = torch.from_numpy(data["actions"]).float()
            act_xyz = data["actions"][:, :3]
            act_rotation_6d = data["actions"][:, 3:9]
            act_gripper = data["actions"][:, 9:10] # [gripper]
            act_euler = pt.matrix_to_euler_angles(pt.rotation_6d_to_matrix(torch.from_numpy(act_rotation_6d)),"XYZ").cpu().numpy()
            actions = np.concatenate([act_xyz, act_euler, act_gripper], axis=-1)
            return {"actions": actions} # use abs actions [x,y,z,rx,ry,rz,gripper] for Franka
        else:
            return {"actions": np.asarray(data["actions"][:, :7])} # use abs actions [x,y,z,rx,ry,rz,gripper] for Franka

@dataclasses.dataclass(frozen=True)
class FrankaSingleCamEEInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change this for your own dataset.
    action_dim: int # default is defined in the model config(Pi0Config), 32.

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    
    # Whether to train actions using rotation_6d or not.
    action_train_with_rotation_6d: bool = False 

    def __call__(self, data: dict) -> dict:

        # We pad the proprioceptive input to the action dimension of the model.
        # For pi0-FAST, we don't pad the state. For Libero, we don't need to differentiate
        # since the pi0-FAST action_dim = 7, which is < state_dim = 8, so pad is skipped.
        # Keep this for your own dataset, but if your dataset stores the proprioceptive input
        # in a different key than "observation/state", you should change it below.
        assert data["observation/state"].shape==(7,), f"Expected state shape (7,), got {data['observation/state'].shape}"
        if isinstance(data["observation/state"], np.ndarray):
            data["observation/state"] = torch.from_numpy(data["observation/state"]).float()

        xyz = data["observation/state"][:3]  # [x, y, z]
        euler_xyz = data["observation/state"][3:6]  # [rx, ry, rz]
        gripper = data["observation/state"][-1:]  # [gripper]
        rotation_6d = pt.matrix_to_rotation_6d(pt.euler_angles_to_matrix(euler_xyz, convention="XYZ"))
        state = torch.concat([xyz, rotation_6d, gripper], axis=-1) # [x, y, z, rotation_6d, gripper]

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])

        # We only mask padding for pi0 model, not pi0-FAST. 
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
                image_masks = (np.True_, np.False_, np.False_) # with padding
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.True_) # without padding
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Actions are only available during training.
        if "actions" in data:
            # We are padding to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            # maybe should transfer
            assert len(data["actions"].shape)==2 and data["actions"].shape[-1] == 7, \
                f"Expected actions shape (N, 7), got {data['actions'].shape}"
            if self.action_train_with_rotation_6d:
                if isinstance(data["actions"], np.ndarray):
                    data["actions"] = torch.from_numpy(data["actions"]).float()
                act_xyz = data["actions"][:,:3] # [x, y, z]
                act_euler_xyz = data["actions"][:,3:6] # [rx, ry, rz] 
                act_gripper = data["actions"][:,-1:] # [gripper] 
                act_rotation_6d = pt.matrix_to_rotation_6d(pt.euler_angles_to_matrix(act_euler_xyz, convention="XYZ"))
                actions = torch.concat([act_xyz, act_rotation_6d, act_gripper], axis=-1) # [x, y, z, rotation_6d, gripper]
            else:
                actions = data["actions"]
            inputs["actions"] = actions

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs

