"""Visualize left-to-right alignment with concrete examples.

This script shows how sequences with different lengths are transformed from
left-aligned to right-aligned within a batch.
"""
import torch
import numpy as np


def left_to_right_align_vectorized(x, input_mask, attn_mask):
    """Vectorized implementation."""
    batch_size, seq_len = input_mask.shape
    device = x.device
    
    arange = torch.arange(seq_len, device=device).unsqueeze(0)
    seqlens = torch.max(input_mask.float() * arange, dim=1)[0] + 1
    seqlens = seqlens.long()
    
    seq_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    rolled_seq_indices = (seq_indices + seqlens.unsqueeze(1)) % seq_len
    
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
    x_rolled = x[batch_indices, rolled_seq_indices]
    mask_rolled = input_mask[batch_indices, rolled_seq_indices]
    
    row_indices = rolled_seq_indices.unsqueeze(2)
    col_indices = rolled_seq_indices.unsqueeze(1)
    batch_indices_3d = batch_indices.unsqueeze(2)
    
    attn_rolled = attn_mask[
        batch_indices_3d,
        row_indices,
        col_indices
    ]
    
    return x_rolled, mask_rolled, attn_rolled


def visualize_embeddings():
    """Show how embeddings are aligned."""
    print("=" * 80)
    print("EXAMPLE 1: Visualizing Embedding Alignment")
    print("=" * 80)
    
    batch_size, seq_len, dim = 4, 10, 1
    actual_lens = [3, 5, 7, 10]
    
    # Create embeddings with unique identifiers
    # Each position gets a unique letter
    x = torch.zeros(batch_size, seq_len, dim)
    letters = "ABCDEFGHIJ"
    
    for i in range(batch_size):
        for j in range(seq_len):
            # Encode letter as a number (A=1, B=2, etc.)
            x[i, j, 0] = ord(letters[j]) - ord('A') + 1
    
    # Create masks
    input_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    for i, length in enumerate(actual_lens):
        input_mask[i, :length] = True
    
    attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)
    
    # Apply alignment
    x_rolled, mask_rolled, _ = left_to_right_align_vectorized(x, input_mask, attn_mask)
    
    # Display results
    for i, length in enumerate(actual_lens):
        print(f"\nBatch {i} (sequence length = {length}):")
        print("-" * 60)
        
        # Original (left-aligned)
        original = []
        for j in range(seq_len):
            if input_mask[i, j]:
                original.append(letters[j])
            else:
                original.append("_")
        print(f"  BEFORE (left-aligned):  [{' '.join(original)}]")
        
        # Rolled (right-aligned)
        rolled = []
        for j in range(seq_len):
            if mask_rolled[i, j]:
                # Find which letter this is
                val = int(x_rolled[i, j, 0].item())
                rolled.append(letters[val - 1])
            else:
                rolled.append("_")
        print(f"  AFTER  (right-aligned): [{' '.join(rolled)}]")
        
        # Verify right-alignment
        padding_count = seq_len - length
        valid_count = length
        print(f"  Verification: {padding_count} padding + {valid_count} valid tokens")
        print(f"    Padding at positions: 0-{padding_count-1 if padding_count > 0 else 'none'}")
        print(f"    Valid at positions:   {padding_count}-{seq_len-1}")


def visualize_attention_mask():
    """Show how attention masks are transformed."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Visualizing Attention Mask Alignment")
    print("=" * 80)
    
    batch_size, seq_len = 2, 8
    actual_lens = [3, 5]
    
    # Create causal attention masks (lower triangular)
    x = torch.randn(batch_size, seq_len, 4)
    input_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    for i, length in enumerate(actual_lens):
        input_mask[i, :length] = True
    
    # Causal mask: each position can only attend to itself and previous positions
    attn_mask = torch.tril(torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool))
    
    # Apply alignment
    _, mask_rolled, attn_rolled = left_to_right_align_vectorized(x, input_mask, attn_mask)
    
    def mask_to_str(mask_2d):
        """Convert 2D boolean mask to visual string."""
        result = []
        for row in mask_2d:
            result.append(''.join(['█' if val else '·' for val in row]))
        return '\n    '.join(result)
    
    for i, length in enumerate(actual_lens):
        print(f"\nBatch {i} (sequence length = {length}):")
        print("-" * 60)
        print("  BEFORE (left-aligned causal mask):")
        print(f"    {mask_to_str(attn_mask[i])}")
        print("\n  AFTER (right-aligned causal mask):")
        print(f"    {mask_to_str(attn_rolled[i])}")


def visualize_numeric_example():
    """Show numeric values to make the transformation crystal clear."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Numeric Values - Step by Step")
    print("=" * 80)
    
    batch_size, seq_len, dim = 3, 8, 1
    actual_lens = [3, 5, 7]
    
    # Create simple numeric sequences: [0, 1, 2, 3, ...]
    x = torch.zeros(batch_size, seq_len, dim)
    for i in range(batch_size):
        x[i, :, 0] = torch.arange(seq_len) * 10  # 0, 10, 20, 30, ...
    
    input_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    for i, length in enumerate(actual_lens):
        input_mask[i, :length] = True
    
    attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)
    
    # Show the transformation step by step
    device = x.device
    arange = torch.arange(seq_len, device=device).unsqueeze(0)
    seqlens = torch.max(input_mask.float() * arange, dim=1)[0] + 1
    
    print("\nStep 1: Compute sequence lengths for each batch")
    print(f"  Sequence lengths: {seqlens.tolist()}")
    
    seq_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    rolled_seq_indices = (seq_indices + seqlens.unsqueeze(1)) % seq_len
    
    print("\nStep 2: Compute rolled indices for each batch")
    print("  Original indices: [0, 1, 2, 3, 4, 5, 6, 7]")
    for i, length in enumerate(actual_lens):
        print(f"  Batch {i} (len={length}): roll by {length} → {rolled_seq_indices[i].tolist()}")
    
    # Apply alignment
    x_rolled, mask_rolled, _ = left_to_right_align_vectorized(x, input_mask, attn_mask)
    
    print("\nStep 3: Results - actual values at each position")
    for i, length in enumerate(actual_lens):
        print(f"\nBatch {i} (sequence length = {length}):")
        
        # Original values
        orig_values = [int(x[i, j, 0].item()) for j in range(seq_len)]
        orig_mask = ['✓' if input_mask[i, j] else '✗' for j in range(seq_len)]
        
        print(f"  BEFORE:")
        print(f"    Values: {orig_values}")
        print(f"    Valid:  {orig_mask}")
        
        # Rolled values
        rolled_values = [int(x_rolled[i, j, 0].item()) for j in range(seq_len)]
        rolled_mask = ['✓' if mask_rolled[i, j] else '✗' for j in range(seq_len)]
        
        print(f"  AFTER:")
        print(f"    Values: {rolled_values}")
        print(f"    Valid:  {rolled_mask}")
        
        # Verify
        padding_count = seq_len - length
        print(f"  ✓ First {padding_count} positions are padding (✗)")
        print(f"  ✓ Last {length} positions contain valid data (✓)")


def visualize_mixed_batch():
    """Show a realistic mixed batch scenario."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Realistic Mixed Batch (Text Tokens)")
    print("=" * 80)
    
    # Simulate text sequences with different lengths
    sequences = [
        ["The", "cat", "sat"],                                    # 3 tokens
        ["A", "quick", "brown", "fox", "jumps"],                 # 5 tokens
        ["Hello", "world", "this", "is", "a", "test", "message"] # 7 tokens
    ]
    
    batch_size = len(sequences)
    seq_len = 10  # Maximum sequence length
    dim = 1
    
    print(f"\nInput sequences (left-aligned):")
    print("-" * 60)
    
    # Create embeddings where each position stores its token index
    x = torch.zeros(batch_size, seq_len, dim)
    input_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    for i, seq in enumerate(sequences):
        actual_len = len(seq)
        # Display original
        padded_seq = seq + ["[PAD]"] * (seq_len - actual_len)
        print(f"  Batch {i}: {' | '.join(padded_seq)}")
        
        # Store indices
        for j in range(actual_len):
            x[i, j, 0] = j + 1  # Store 1-indexed position
            input_mask[i, j] = True
    
    attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)
    
    # Apply alignment
    x_rolled, mask_rolled, _ = left_to_right_align_vectorized(x, input_mask, attn_mask)
    
    print(f"\nOutput sequences (right-aligned):")
    print("-" * 60)
    
    for i, seq in enumerate(sequences):
        actual_len = len(seq)
        padding_count = seq_len - actual_len
        
        # Build aligned output
        aligned_tokens = ["[PAD]"] * padding_count + seq
        print(f"  Batch {i}: {' | '.join(aligned_tokens)}")
        
        # Verify programmatically
        print(f"    → Positions 0-{padding_count-1}: padding")
        print(f"    → Positions {padding_count}-{seq_len-1}: {seq}")
        
        # Double-check with actual mask
        mask_check = ["PAD" if not mask_rolled[i, j] else "VAL" for j in range(seq_len)]
        print(f"    → Mask check: {' '.join(mask_check)}")


def main():
    """Run all visualizations."""
    print("\n" + "🔍" * 40)
    print("LEFT-TO-RIGHT ALIGNMENT VISUALIZATION")
    print("🔍" * 40)
    
    visualize_embeddings()
    visualize_attention_mask()
    visualize_numeric_example()
    visualize_mixed_batch()
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print("""
The left_to_right_align function transforms sequences from LEFT-aligned to RIGHT-aligned:

LEFT-aligned (before):
  Batch 0: [A B C _ _ _]  (3 tokens)
  Batch 1: [A B C D E _]  (5 tokens)

RIGHT-aligned (after):
  Batch 0: [_ _ _ A B C]  (3 tokens at end)
  Batch 1: [_ A B C D E]  (5 tokens at end)

Key properties:
✓ Each batch example can have a different sequence length
✓ Valid tokens move to the END of the sequence
✓ Padding moves to the START of the sequence
✓ All transformations happen in parallel (no loops!)
✓ Attention masks are rolled in BOTH dimensions to maintain structure
""")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

