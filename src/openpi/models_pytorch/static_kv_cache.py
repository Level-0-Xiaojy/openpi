"""Static KV Cache implementation for pre-allocated memory with indexed updates.

This module provides a StaticKVCache class that matches the JAX implementation's
behavior of pre-allocating cache memory and using indexed updates rather than
concatenation. This is compatible with the transformers Cache interface.
"""

from typing import Optional, Tuple
import torch
from transformers.cache_utils import Cache


class StaticKVCache(Cache):
    """
    Static KV cache with pre-allocated memory and indexed updates.
    
    This cache class pre-allocates memory for the maximum sequence length
    and uses indexed updates to insert new key-value pairs, matching the
    JAX implementation's behavior with jax.lax.dynamic_update_slice.
    
    JAX Reference Behavior (from gemma.py):
    1. _init_cache: Pre-allocates cache by padding k,v to cache_size (attn_mask.shape[-1])
       - Cache shape: [batch, prefill_len + max_decoding_steps, num_heads, head_dim]
       - Initial position idx = prefill_len (after prefill)
    
    2. _update_cache: Updates cache at specific position using dynamic_update_slice
       - Only for single-token updates (k.shape[1] == 1)
       - Writes at position idx, then increments: idx_new = idx + 1
    
    3. During action sampling: Concatenates cached k,v with new k,v WITHOUT updating cache
       - k = concatenate([k_cache, k], axis=1) when k.shape[1] > 1
    
    PyTorch Implementation:
    - Pre-allocates to max_cache_len during initialization (matching step 1)
    - Uses index_copy_ for updates instead of concatenation (matching step 2)
    - Compatible with transformers Cache interface (use_cache=True/False)
    - When use_cache=False in attention, uses __getitem__ + concatenation (matching step 3)
    
    This is backward-compatible with transformers' Cache interface.
    """

    def __init__(
        self,
        max_batch_size: int = 1,
        max_cache_len: int = 4096,
        num_layers: int = 18,
        num_key_value_heads: int = 1,
        head_dim: int = 256,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
    ):
        """Initialize pre-allocated static KV cache.
        
        Args:
            max_batch_size: Maximum batch size
            max_cache_len: Maximum sequence length (prefill_size + max_decoding_steps)
            num_layers: Number of transformer layers
            num_key_value_heads: Number of key-value attention heads
            head_dim: Dimension of each attention head
            dtype: Data type for cache tensors
            device: Device to allocate tensors on
        """
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        # Pre-allocate cache for all layers
        # Shape: [batch, num_heads, max_seq_len, head_dim]
        cache_shape = (max_batch_size, num_key_value_heads, max_cache_len, head_dim)
        
        # Initialize empty cache for each layer
        self.key_cache = []
        self.value_cache = []
        for _ in range(num_layers):
            self.key_cache.append(torch.zeros(cache_shape, dtype=dtype, device=device))
            self.value_cache.append(torch.zeros(cache_shape, dtype=dtype, device=device))
        
        # Track current position in cache for each batch element
        self.cache_position = torch.zeros((max_batch_size,), dtype=torch.long, device=device)
        self._seen_tokens = 0  # For compatibility with Cache interface

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key-value states using indexed update.
        
        Matches JAX implementation:
        - During prefill: Writes all tokens and sets position to end
        - During autoregressive: Writes single token at current position and increments
        
        Args:
            key_states: New key states [batch, num_heads, seq_len, head_dim]
            value_states: New value states [batch, num_heads, seq_len, head_dim]
            layer_idx: Index of the layer being updated
            cache_kwargs: Additional kwargs (contains cache_position from transformers)
        
        Returns:
            Tuple of (updated_keys, updated_values) containing full cache up to current position
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        # Use cache_position from transformers if provided, otherwise use internal counter
        if cache_kwargs is not None and "cache_position" in cache_kwargs:
            cache_position = cache_kwargs["cache_position"]
            if isinstance(cache_position, torch.Tensor):
                write_positions = cache_position
            else:
                # If it's a range, convert to tensor
                write_positions = torch.arange(
                    cache_position, cache_position + seq_len, 
                    device=key_states.device, dtype=torch.long
                )
        else:
            # Use internal counter
            write_pos = self.cache_position[0].item()
            write_positions = torch.arange(
                write_pos, write_pos + seq_len, 
                device=key_states.device, dtype=torch.long
            )
        
        # Handle both single position and range of positions
        if write_positions.numel() == 1:
            # Single position
            start_pos = write_positions.item()
            end_pos = start_pos + seq_len
            write_positions = torch.arange(start_pos, end_pos, device=key_states.device, dtype=torch.long)
        else:
            start_pos = write_positions[0].item()
            end_pos = write_positions[-1].item() + 1
        
        # Check if we exceed max cache length
        if end_pos > self.max_cache_len:
            raise RuntimeError(
                f"Cache position {end_pos} exceeds max cache length {self.max_cache_len}"
            )
        
        # Write to cache using index_copy_ (matching JAX's dynamic_update_slice)
        self.key_cache[layer_idx].index_copy_(2, write_positions, key_states)
        self.value_cache[layer_idx].index_copy_(2, write_positions, value_states)
        
        # Update position counter
        # Note: In normal transformers usage, layers are called sequentially in a single forward pass,
        # so this will be set to the same value by each layer. In testing, we update layers independently,
        # so we need to handle that case too.
        self.cache_position[0] = end_pos
        self._seen_tokens = end_pos
        
        # Return full cache up to current position
        # This matches JAX behavior: return k_cache, v_cache (full cache, not sliced)
        return (
            self.key_cache[layer_idx][:, :, :end_pos, :],
            self.value_cache[layer_idx][:, :, :end_pos, :],
        )

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """Return current sequence length in cache."""
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        """Return maximum cache length."""
        return self.max_cache_len

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = None) -> int:
        """Return usable length considering new sequence length."""
        return min(self._seen_tokens, self.max_cache_len - new_seq_length)

    def reset(self):
        """Reset cache position counter without deallocating memory."""
        self.cache_position.zero_()
        self._seen_tokens = 0

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached key-value pairs for a specific layer.
        
        This maintains compatibility with the pattern:
        key_states = torch.cat([past_key_value[layer_idx][0], key_states], dim=2)
        """
        current_len = self._seen_tokens
        return (
            self.key_cache[layer_idx][:, :, :current_len, :],
            self.value_cache[layer_idx][:, :, :current_len, :],
        )

    def __len__(self) -> int:
        """Return number of layers."""
        return self.num_layers

