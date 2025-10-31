"""
Simple Pi05 Subtask Generation Test.

This is a simplified version similar to test_pi05_subtask_generation.py
but designed to work with any Pi05 model configuration.

Usage:
    python test_pi05_simple.py --config-name pi05_droid
"""

import dataclasses
import logging
import os

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import tyro
from flax import nnx

from openpi.models import model as _model
from openpi.models.tokenizer import PaligemmaTokenizer
import openpi.shared.nnx_utils as nnx_utils
from openpi.training.config import get_config


@dataclasses.dataclass
class Args:
    # Config name from training/config.py
    config_name: str = "pi05_droid"
    
    # Image paths (3 images: base, left_wrist, right_wrist)
    base_image: str = "tmp_test/faceImg.png"
    left_wrist_image: str = "tmp_test/leftImg.png"
    right_wrist_image: str = "tmp_test/rightImg.png"
    
    # Task prompts
    high_level_prompt: str = "Pick up the flashcard on the table"
    low_level_prompt: str | None = None  # For testing, set to None for inference
    
    # Generation parameters
    max_decoding_steps: int = 25
    temperature: float = 0.1
    paligemma_eos_token: int = 1
    
    # Number of test runs
    num_runs: int = 3
    
    # Random seed
    seed: int = 42


def load_image(path: str) -> jnp.ndarray:
    """Load and preprocess an image."""
    if not os.path.exists(path):
        logging.warning(f"Image not found: {path}, using random image")
        return jnp.array(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    
    # Convert from [0, 255] to [-1, 1]
    img = img.astype(jnp.float32) / 127.5 - 1.0
    return jnp.array(img[np.newaxis, :, :, :])


def main(args: Args):
    logging.info("=" * 80)
    logging.info("Pi05 Subtask Generation Simple Test")
    logging.info("=" * 80)
    
    # Load config
    logging.info(f"Loading config: {args.config_name}")
    config = get_config(args.config_name)
    
    # Initialize model
    logging.info("Initializing model...")
    model_rng = jax.random.key(args.seed)
    model = config.model.create(model_rng)
    
    # Load pretrained params
    logging.info("Loading pretrained weights...")
    graphdef, state = nnx.split(model)
    loader = config.weight_loader
    params = nnx.state(model)
    
    # Convert frozen params to bfloat16
    if config.model.dtype == "bfloat16":
        params = nnx_utils.state_map(
            params, 
            config.freeze_filter, 
            lambda p: p.replace(p.value.astype(jnp.bfloat16))
        )
    
    params_shape = params.to_pure_dict()
    loaded_params = loader.load(params_shape)
    state.replace_by_pure_dict(loaded_params)
    model = nnx.merge(graphdef, state)
    
    logging.info("Model loaded successfully!")
    
    # Load images
    logging.info("Loading images...")
    img_dict = {
        "base_0_rgb": load_image(args.base_image),
        "left_wrist_0_rgb": load_image(args.left_wrist_image),
        "right_wrist_0_rgb": load_image(args.right_wrist_image),
    }
    
    # Tokenize prompts
    logging.info("Tokenizing prompts...")
    tokenizer = PaligemmaTokenizer(max_len=config.model.max_token_len)
    
    if args.low_level_prompt:
        # For training/testing: use both high and low level prompts
        tokenized_prompt, tokenized_prompt_mask, token_ar_mask, token_loss_mask = \
            tokenizer.tokenize_high_low_prompt(args.high_level_prompt, args.low_level_prompt)
    else:
        # For inference: only high level prompt
        tokenized_prompt, tokenized_prompt_mask = \
            tokenizer.tokenize_high_level_prompt(args.high_level_prompt)
        token_ar_mask = np.ones_like(tokenized_prompt)
        token_loss_mask = np.zeros_like(tokenized_prompt, dtype=bool)
    
    # Create observation
    logging.info("Creating observation...")
    data = {
        'image': img_dict,
        'image_mask': {key: jnp.ones(1, dtype=jnp.bool_) for key in img_dict.keys()},
        'state': jnp.zeros((1, config.model.action_dim), dtype=jnp.float32),
        'tokenized_prompt': jnp.stack([tokenized_prompt], axis=0),
        'tokenized_prompt_mask': jnp.stack([tokenized_prompt_mask], axis=0),
        'token_ar_mask': jnp.stack([token_ar_mask], axis=0),
        'token_loss_mask': jnp.stack([token_loss_mask], axis=0),
    }
    
    observation = _model.Observation.from_dict(data)
    rng = jax.random.key(args.seed)
    observation = _model.preprocess_observation(
        rng, observation, train=False, 
        image_keys=list(observation.images.keys())
    )
    
    # For inference: mask out the low-level task tokens if they exist
    if args.low_level_prompt is None:
        loss_mask = jnp.array(observation.token_loss_mask)
        new_tokenized_prompt = observation.tokenized_prompt.at[loss_mask].set(0)
        new_tokenized_prompt_mask = observation.tokenized_prompt_mask.at[loss_mask].set(False)
        observation = _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=new_tokenized_prompt,
            tokenized_prompt_mask=new_tokenized_prompt_mask,
            token_ar_mask=observation.token_ar_mask,
            token_loss_mask=observation.token_loss_mask,
        )
        observation = _model.preprocess_observation(
            None, observation, train=False, 
            image_keys=list(observation.images.keys())
        )
    
    observation = jax.tree.map(jax.device_put, observation)
    
    # JIT compile
    logging.info("JIT compiling model...")
    if not hasattr(model, 'jit_sample_low_level_task'):
        model.jit_sample_low_level_task = nnx_utils.module_jit(
            model.sample_low_level_task, 
            static_argnums=(3,)
        )
    
    # Run inference
    logging.info("=" * 80)
    logging.info("Running inference...")
    logging.info("=" * 80)
    logging.info(f"High-level prompt: {args.high_level_prompt}")
    if args.low_level_prompt:
        logging.info(f"Low-level prompt: {args.low_level_prompt}")
    logging.info("")
    
    import time
    
    for i in range(args.num_runs):
        start_time = time.time()
        
        # Generate subtask
        predicted_tokens, kv_cache, mask, ar_mask = model.jit_sample_low_level_task(
            rng, 
            observation, 
            args.max_decoding_steps, 
            args.paligemma_eos_token, 
            args.temperature
        )
        
        end_time = time.time()
        
        # Detokenize
        for batch_idx in range(predicted_tokens.shape[0]):
            generated_subtask = tokenizer.detokenize(
                np.array(predicted_tokens[batch_idx], dtype=np.int32)
            )
            
            logging.info(f"Run {i+1}/{args.num_runs}:")
            logging.info(f"  ✅ Generated: \033[92m{generated_subtask}\033[0m")
            logging.info(f"  ⏱️  Time: {end_time - start_time:.3f}s")
            
            if args.low_level_prompt:
                full_tokens = np.array(data['tokenized_prompt'], dtype=np.int32)
                ground_truth = tokenizer.detokenize(full_tokens[batch_idx])
                logging.info(f"  📝 Ground truth: {ground_truth}")
            logging.info("")
    
    logging.info("=" * 80)
    logging.info("✅ Test completed successfully!")
    logging.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    tyro.cli(main)


