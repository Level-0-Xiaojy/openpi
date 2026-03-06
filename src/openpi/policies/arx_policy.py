import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_arx_example() -> dict:
    """Creates a random input example for the ARX policy."""
    return {
        "state": np.random.rand(14),
        "images": {
            "left_wrist_view": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "face_view": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "right_wrist_view": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class ArxInputs(transforms.DataTransformFn):
    """Inputs for the ARX (X2Robot) policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14] (left 7 + right 7: position(3) + rotation(3) + gripper(1))
    - actions: [action_horizon, 14]
    """

    model_type: _model.ModelType = _model.ModelType.PI0

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("left_wrist_view", "face_view", "right_wrist_view")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        for key in self.EXPECTED_CAMERAS:
            if key not in in_images:
                raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        def convert_image(img):
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            if img.shape[0] == 3:
                img = einops.rearrange(img, "c h w -> h w c")
            return img

        images = {name: convert_image(in_images[name]) for name in self.EXPECTED_CAMERAS}

        inputs = {
            "image": {
                "base_0_rgb": images["face_view"],
                "left_wrist_0_rgb": images["left_wrist_view"],
                "right_wrist_0_rgb": images["right_wrist_view"],
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": np.asarray(data["state"]),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ArxOutputs(transforms.DataTransformFn):
    """Outputs for the ARX (X2Robot) policy."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}
