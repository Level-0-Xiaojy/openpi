import dataclasses
import einops
import numpy as np
from typing import ClassVar

from openpi import transforms
from openpi.models import model as _model


def make_arx_example() -> dict:
    """Creates a random input example for the ARX policy."""
    return {
        "state": np.random.rand(14),
        "image": {
            "left_wrist_view": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "face_view": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "right_wrist_view": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class ArxInputs(transforms.DataTransformFn):
    """Transform inputs for the ARX policy."""

    action_dim: int = 32
    model_type: _model.ModelType = _model.ModelType.PI0
    state_history_size: int = 0
    state_future_size: int = 0
    slave_state_dim: int = 14
    mask_history_slave_states: bool = False

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("left_wrist_view", "face_view", "right_wrist_view")

    def __call__(self, data: dict) -> dict:
        state = data["state"]
        
        # Handle state sequence (history + current + future)
        if state.ndim == 2:
            # State shape: (seq_len, state_dim)
            state = self._mask_slave_states(state)
        
        # Pad state to action_dim
        state = transforms.pad_to_dim(state, self.action_dim)

        def convert_image(img):
            img = np.asarray(img)
            # Convert to uint8 if using float images.
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            # Convert from [channel, height, width] to [height, width, channel].
            if img.shape[-1] != 3:
                output_image = einops.rearrange(img, "c h w -> h w c")
            else:
                output_image = img
            assert output_image.shape[-1] == 3, f"Image must have 3 channels, got {output_image.shape}."
            return output_image

        # Convert images to uint8 and rearrange to (H,W,C) format
        for key in self.EXPECTED_CAMERAS:
            assert key in data['images'].keys(), f"Images must contain {key}."
            data['images'][key] = convert_image(data['images'][key])

        inputs = {
            "image": {
                "base_0_rgb": data['images']['face_view'],
                "left_wrist_0_rgb": data['images']['left_wrist_view'],
                "right_wrist_0_rgb": data['images']['right_wrist_view'],
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": state,
        }

        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "actions_is_pad" in data:
            inputs["actions_is_pad"] = data["actions_is_pad"]

        return inputs

    def _mask_slave_states(self, state: np.ndarray) -> np.ndarray:
        """Mask future slave states by copying current slave state."""
        state = np.asarray(state).copy()
        current_idx = self.state_history_size
        current_slave = state[current_idx, :self.slave_state_dim]
        
        if self.state_future_size > 0:    
            state[current_idx + 1:, :self.slave_state_dim] = current_slave
        if self.mask_history_slave_states and self.state_history_size > 0:
            state[:current_idx, :self.slave_state_dim] = current_slave

        return state

@dataclasses.dataclass(frozen=True)
class ArxOutputs(transforms.DataTransformFn):
    """Outputs for the ARX policy."""
    action_dim: int = 14
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :self.action_dim])
        return {"actions": actions}
