"""Test script to verify StaticKVCache behavior matches JAX implementation."""

import torch
from static_kv_cache import StaticKVCache


def test_basic_functionality():
    """Test basic cache initialization and update."""
    print("Test 1: Basic functionality")
    
    # Initialize cache
    cache = StaticKVCache(
        max_batch_size=1,
        max_cache_len=100,
        num_layers=2,
        num_key_value_heads=4,
        head_dim=64,
        dtype=torch.float32,
        device='cpu'
    )
    
    print(f"  ✓ Cache initialized with max_len={cache.max_cache_len}")
    print(f"  ✓ Initial position: {cache.cache_position[0].item()}")
    
    # Simulate prefill with 50 tokens
    batch_size, num_heads, prefill_len, head_dim = 1, 4, 50, 64
    k_prefill = torch.randn(batch_size, num_heads, prefill_len, head_dim)
    v_prefill = torch.randn(batch_size, num_heads, prefill_len, head_dim)
    
    k_out, v_out = cache.update(k_prefill, v_prefill, layer_idx=0)
    
    print(f"  ✓ After prefill: cache position = {cache.cache_position[0].item()}")
    print(f"  ✓ Returned cache shape: {k_out.shape}")
    assert k_out.shape[2] == prefill_len, f"Expected {prefill_len}, got {k_out.shape[2]}"
    
    # Simulate autoregressive decoding with single tokens
    for step in range(5):
        k_token = torch.randn(batch_size, num_heads, 1, head_dim)
        v_token = torch.randn(batch_size, num_heads, 1, head_dim)
        
        k_out, v_out = cache.update(k_token, v_token, layer_idx=0)
        expected_len = prefill_len + step + 1
        
        assert k_out.shape[2] == expected_len, f"Step {step}: Expected {expected_len}, got {k_out.shape[2]}"
        print(f"  ✓ Step {step}: position = {cache.cache_position[0].item()}, cache_len = {k_out.shape[2]}")
    
    print("  ✅ Test 1 passed!\n")


def test_getitem_behavior():
    """Test __getitem__ behavior for use_cache=False scenario."""
    print("Test 2: __getitem__ for concatenation")
    
    cache = StaticKVCache(
        max_batch_size=1,
        max_cache_len=100,
        num_layers=1,
        num_key_value_heads=4,
        head_dim=64,
        dtype=torch.float32,
        device='cpu'
    )
    
    # Prefill
    batch_size, num_heads, prefill_len, head_dim = 1, 4, 50, 64
    k_prefill = torch.randn(batch_size, num_heads, prefill_len, head_dim)
    v_prefill = torch.randn(batch_size, num_heads, prefill_len, head_dim)
    cache.update(k_prefill, v_prefill, layer_idx=0)
    
    # Get cached k,v (simulating use_cache=False)
    k_cached, v_cached = cache[0]
    print(f"  ✓ Retrieved cached K/V shape: {k_cached.shape}")
    assert k_cached.shape[2] == prefill_len
    
    # Simulate action sampling with multiple tokens (action_horizon=10)
    action_horizon = 10
    k_action = torch.randn(batch_size, num_heads, action_horizon, head_dim)
    v_action = torch.randn(batch_size, num_heads, action_horizon, head_dim)
    
    # Concatenate (matching JAX behavior with use_cache=False)
    k_combined = torch.cat([k_cached, k_action], dim=2)
    v_combined = torch.cat([v_cached, v_action], dim=2)
    
    print(f"  ✓ Concatenated shape: {k_combined.shape}")
    assert k_combined.shape[2] == prefill_len + action_horizon
    
    # Verify cache wasn't updated
    k_cached_after, _ = cache[0]
    assert k_cached_after.shape[2] == prefill_len, "Cache should not be updated"
    print(f"  ✓ Cache unchanged after concatenation: {k_cached_after.shape[2]}")
    
    print("  ✅ Test 2 passed!\n")


def test_multi_layer():
    """Test that cache works correctly for multiple layers (simulating transformers behavior)."""
    print("Test 3: Multi-layer cache")
    
    num_layers = 3
    cache = StaticKVCache(
        max_batch_size=1,
        max_cache_len=100,
        num_layers=num_layers,
        num_key_value_heads=4,
        head_dim=64,
        dtype=torch.float32,
        device='cpu'
    )
    
    batch_size, num_heads, seq_len, head_dim = 1, 4, 10, 64
    
    # Simulate a single forward pass where all layers write to the same positions
    # (This is how transformers actually works - cache_position is the same for all layers)
    cache_position = torch.arange(0, seq_len)
    cache_kwargs = {"cache_position": cache_position}
    
    for layer_idx in range(num_layers):
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k_out, v_out = cache.update(k, v, layer_idx=layer_idx, cache_kwargs=cache_kwargs)
        
        print(f"  ✓ Layer {layer_idx}: cache updated, shape = {k_out.shape}")
        assert k_out.shape[2] == seq_len, f"Layer {layer_idx}: expected {seq_len}, got {k_out.shape[2]}"
    
    # Verify all layers have the same sequence length
    for layer_idx in range(num_layers):
        k_cached, _ = cache[layer_idx]
        assert k_cached.shape[2] == seq_len
        print(f"  ✓ Layer {layer_idx} verified: cache_len = {k_cached.shape[2]}")
    
    # Now simulate a second forward pass with a single token (autoregressive)
    cache_position = torch.tensor([seq_len])  # Write at position seq_len
    cache_kwargs = {"cache_position": cache_position}
    
    for layer_idx in range(num_layers):
        k = torch.randn(batch_size, num_heads, 1, head_dim)
        v = torch.randn(batch_size, num_heads, 1, head_dim)
        k_out, v_out = cache.update(k, v, layer_idx=layer_idx, cache_kwargs=cache_kwargs)
        
        expected_len = seq_len + 1
        assert k_out.shape[2] == expected_len, f"Layer {layer_idx}: expected {expected_len}, got {k_out.shape[2]}"
    
    print(f"  ✓ Autoregressive token added, final cache_len = {seq_len + 1}")
    print("  ✅ Test 3 passed!\n")


def test_cache_position_from_kwargs():
    """Test that cache_position from kwargs works correctly."""
    print("Test 4: cache_position from kwargs")
    
    cache = StaticKVCache(
        max_batch_size=1,
        max_cache_len=100,
        num_layers=1,
        num_key_value_heads=4,
        head_dim=64,
        dtype=torch.float32,
        device='cpu'
    )
    
    batch_size, num_heads, head_dim = 1, 4, 64
    
    # Update with explicit cache_position
    positions = torch.arange(0, 5)
    k = torch.randn(batch_size, num_heads, 5, head_dim)
    v = torch.randn(batch_size, num_heads, 5, head_dim)
    
    cache_kwargs = {"cache_position": positions}
    k_out, v_out = cache.update(k, v, layer_idx=0, cache_kwargs=cache_kwargs)
    
    print(f"  ✓ Updated with explicit positions: {positions.tolist()}")
    print(f"  ✓ Cache position after update: {cache.cache_position[0].item()}")
    assert cache.cache_position[0].item() == 5
    
    # Add more with explicit position
    positions = torch.tensor([5])
    k = torch.randn(batch_size, num_heads, 1, head_dim)
    v = torch.randn(batch_size, num_heads, 1, head_dim)
    
    cache_kwargs = {"cache_position": positions}
    k_out, v_out = cache.update(k, v, layer_idx=0, cache_kwargs=cache_kwargs)
    
    print(f"  ✓ Added token at position 5")
    print(f"  ✓ Cache position after update: {cache.cache_position[0].item()}")
    assert cache.cache_position[0].item() == 6
    
    print("  ✅ Test 4 passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("StaticKVCache Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_basic_functionality()
        test_getitem_behavior()
        test_multi_layer()
        test_cache_position_from_kwargs()
        
        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise

