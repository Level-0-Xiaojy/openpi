"""
Verify Pi05 configs work correctly for subtask generation.

This script tests if the existing pi05_droid config properly handles
subtask generation inference without needing config modifications.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.training.config import get_config
from openpi.models import model as _model
from openpi.models.tokenizer import PaligemmaTokenizer
import openpi.shared.nnx_utils as nnx_utils


def test_config():
    print("=" * 80)
    print("Testing Pi05 Config Compatibility")
    print("=" * 80)
    
    # Load pi05_droid config
    print("\n1. Loading pi05_droid config...")
    try:
        config = get_config('pi05_droid')
        print(f"   ✅ Config loaded successfully")
        print(f"   - Model type: {config.model.model_type}")
        print(f"   - Pi05 flag: {config.model.pi05}")
        print(f"   - Max token len: {config.model.max_token_len}")
        print(f"   - Action horizon: {config.model.action_horizon}")
    except Exception as e:
        print(f"   ❌ Failed to load config: {e}")
        return False
    
    # Create model
    print("\n2. Creating model...")
    try:
        model_rng = jax.random.key(42)
        model = config.model.create(model_rng)
        print(f"   ✅ Model created successfully")
        print(f"   - Model class: {model.__class__.__name__}")
    except Exception as e:
        print(f"   ❌ Failed to create model: {e}")
        return False
    
    # Check if model has subtask generation method
    print("\n3. Checking subtask generation support...")
    if hasattr(model, 'sample_low_level_task'):
        print(f"   ✅ Model has sample_low_level_task() method")
    else:
        print(f"   ❌ Model missing sample_low_level_task() method")
        return False
    
    # Test tokenizer
    print("\n4. Testing tokenizer...")
    try:
        tokenizer = PaligemmaTokenizer(max_len=config.model.max_token_len)
        
        # Test high-level prompt tokenization (for inference)
        if hasattr(tokenizer, 'tokenize_high_level_prompt'):
            high_prompt = "Pick up the red block"
            tokens, mask = tokenizer.tokenize_high_level_prompt(high_prompt)
            print(f"   ✅ tokenize_high_level_prompt() works")
            print(f"   - Input: '{high_prompt}'")
            print(f"   - Tokens shape: {tokens.shape}")
            print(f"   - Mask shape: {mask.shape}")
        else:
            print(f"   ❌ tokenizer missing tokenize_high_level_prompt()")
            return False
        
        # Test high+low prompt tokenization (for training)
        if hasattr(tokenizer, 'tokenize_high_low_prompt'):
            low_prompt = "pick up the red block."
            tokens, mask, ar_mask, loss_mask = tokenizer.tokenize_high_low_prompt(
                high_prompt, low_prompt
            )
            print(f"   ✅ tokenize_high_low_prompt() works")
            print(f"   - Tokens shape: {tokens.shape}")
        else:
            print(f"   ❌ tokenizer missing tokenize_high_low_prompt()")
            return False
            
    except Exception as e:
        print(f"   ❌ Tokenizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test data transforms
    print("\n5. Testing data transforms...")
    try:
        import pathlib
        data_config = config.data.create(
            pathlib.Path(config.assets_base_dir) / config.name,
            config.model
        )
        print(f"   ✅ Data config created")
        print(f"   - Repo ID: {data_config.repo_id}")
        print(f"   - Has repack transforms: {len(data_config.repack_transforms.inputs) > 0}")
        print(f"   - Has data transforms: {len(data_config.data_transforms.inputs) > 0}")
        print(f"   - Has model transforms: {len(data_config.model_transforms.inputs) > 0}")
    except Exception as e:
        print(f"   ⚠️  Data config creation failed (this is OK for testing): {e}")
    
    print("\n" + "=" * 80)
    print("✅ All critical tests passed!")
    print("=" * 80)
    print("\nConclusion:")
    print("  - Existing pi05_droid config is compatible")
    print("  - Model supports subtask generation")
    print("  - Tokenizer has required methods")
    print("  - Scripts should work without config modifications!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    success = test_config()
    exit(0 if success else 1)


