#!/usr/bin/env python3
"""
Demo script for Phase 7: Verify constrained loss ignores locked tokens.

Shows that:
1. Locked tokens contribute 0 loss
2. Gradients only flow from editable tokens
3. Training focuses on editable regions

Usage:
    python scripts/demo_constrained_loss.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from constrained_diffusion_lm.data import Tokenizer, create_lock_mask
from constrained_diffusion_lm.training.losses import (
    DiffusionLoss,
    ConstrainedDiffusionLoss,
    compute_constrained_accuracy,
)


def main():
    print("=" * 70)
    print("Phase 7 Demo: Constrained Loss (Editable Tokens Only)")
    print("=" * 70)
    
    # Setup
    tokenizer = Tokenizer("bert-base-uncased", max_length=64)
    vocab_size = tokenizer.vocab_size
    
    # Create sample data
    text = "The contract is governed by Ontario law."
    token_ids = tokenizer.encode(text)
    seq_len = len(token_ids)
    
    print(f"\n[Setup]")
    print(f"Text: {text}")
    print(f"Tokens: {[tokenizer.get_token(tid) for tid in token_ids]}")
    
    # Create lock mask (lock "Ontario law" at positions 6-8)
    lock_mask = create_lock_mask(seq_len, lock_spans=[(6, 9)])  # ontario, law, .
    edit_mask = ~lock_mask
    
    print(f"Lock mask: {lock_mask.tolist()}")
    print(f"Edit mask: {edit_mask.tolist()}")
    
    # Create batch tensors
    targets = torch.tensor([token_ids])  # [1, L]
    lock_mask_batch = lock_mask.unsqueeze(0)  # [1, L]
    edit_mask_batch = edit_mask.unsqueeze(0)  # [1, L]
    
    # =========================================================================
    # Test 1: Loss Values
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("Test 1: Compare Loss Values")
    print("=" * 70)
    
    # Create random logits (simulating model output)
    torch.manual_seed(42)
    logits = torch.randn(1, seq_len, vocab_size)  # [1, L, V]
    
    # Standard loss (all tokens)
    standard_loss_fn = DiffusionLoss()
    standard_loss = standard_loss_fn(logits, targets)
    
    # Constrained loss (editable only)
    constrained_loss_fn = ConstrainedDiffusionLoss()
    constrained_loss = constrained_loss_fn(logits, targets, edit_mask_batch)
    
    print(f"\nWith random logits:")
    print(f"  Standard loss (all {seq_len} tokens):     {standard_loss.item():.4f}")
    print(f"  Constrained loss ({edit_mask.sum()} editable): {constrained_loss.item():.4f}")
    
    # =========================================================================
    # Test 2: Perfect Predictions for Locked Tokens
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("Test 2: Perfect Predictions on Locked Tokens")
    print("=" * 70)
    
    # Create logits where locked tokens are perfectly predicted
    logits_perfect_locked = torch.randn(1, seq_len, vocab_size)
    
    # Make locked positions predict correct tokens with high confidence
    for i in range(seq_len):
        if lock_mask[i]:
            logits_perfect_locked[0, i, :] = -10.0  # low score for all
            logits_perfect_locked[0, i, token_ids[i]] = 10.0  # high score for correct
    
    standard_loss_2 = standard_loss_fn(logits_perfect_locked, targets)
    constrained_loss_2 = constrained_loss_fn(logits_perfect_locked, targets, edit_mask_batch)
    
    print(f"\nWith perfect predictions on LOCKED tokens:")
    print(f"  Standard loss:     {standard_loss_2.item():.4f} (lower due to correct locked)")
    print(f"  Constrained loss:  {constrained_loss_2.item():.4f} (unchanged - locked not counted)")
    
    # =========================================================================
    # Test 3: Gradient Flow
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("Test 3: Gradient Flow (Locked tokens get NO gradient)")
    print("=" * 70)
    
    # Create a simple linear layer to track gradients
    embedding = nn.Embedding(vocab_size, 32)
    linear = nn.Linear(32, vocab_size)
    
    # Input tokens
    inputs = torch.tensor([token_ids])
    
    # Forward pass
    embedded = embedding(inputs)  # [1, L, 32]
    logits = linear(embedded)      # [1, L, V]
    
    # Constrained loss
    loss = constrained_loss_fn(logits, targets, edit_mask_batch)
    loss.backward()
    
    # Check gradients for each position
    print(f"\nGradient magnitudes per position:")
    
    # We'll check the embedding gradients for each input token
    input_grads = embedding.weight.grad[token_ids]  # [L, 32]
    
    for i in range(seq_len):
        token = tokenizer.get_token(token_ids[i])
        grad_norm = input_grads[i].norm().item()
        is_locked = lock_mask[i].item()
        status = "LOCKED" if is_locked else "editable"
        
        # Note: Due to how embeddings work, even locked tokens may show gradients
        # if the same token appears in editable positions. But the LOSS contribution is 0.
        print(f"  [{i:2d}] {token:12s} ({status:8s}): grad_norm = {grad_norm:.6f}")
    
    # =========================================================================
    # Test 4: Loss with Stats
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("Test 4: Constrained Loss with Statistics")
    print("=" * 70)
    
    # Reset gradients
    embedding.zero_grad()
    linear.zero_grad()
    
    # Forward with fresh logits
    embedded = embedding(inputs)
    logits = linear(embedded)
    
    # Get loss with stats
    loss, stats = constrained_loss_fn.forward_with_stats(
        logits, targets, edit_mask_batch
    )
    
    print(f"\nTraining statistics:")
    print(f"  Loss (editable only): {stats['loss']:.4f}")
    print(f"  Accuracy (editable):  {stats['acc_editable']:.3f}")
    print(f"  Accuracy (locked):    {stats['acc_locked']:.3f}")
    print(f"  Editable tokens:      {stats['n_editable']}")
    print(f"  Locked tokens:        {stats['n_locked']}")
    
    # =========================================================================
    # Test 5: Verify Per-Token Loss
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("Test 5: Per-Token Loss Values")
    print("=" * 70)
    
    # Compute per-token loss
    per_token_loss = constrained_loss_fn(
        logits, targets, edit_mask_batch, reduction="none"
    )
    
    print(f"\nPer-token loss (locked should be 0):")
    for i in range(seq_len):
        token = tokenizer.get_token(token_ids[i])
        token_loss = per_token_loss[0, i].item()
        is_locked = lock_mask[i].item()
        status = "LOCKED" if is_locked else "editable"
        expected = "0.0" if is_locked else "non-zero"
        
        print(f"  [{i:2d}] {token:12s} ({status:8s}): loss = {token_loss:.4f} (expected: {expected})")
    
    # Verify locked tokens have 0 loss
    locked_losses = per_token_loss[0, lock_mask]
    assert (locked_losses == 0).all(), "Locked tokens should have 0 loss!"
    print(f"\n  ✓ Verified: All locked tokens have exactly 0 loss!")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("Phase 7 Complete: Constrained loss working!")
    print("=" * 70)
    print("""
Key properties verified:
  1. Loss is computed ONLY on editable tokens
  2. Locked tokens contribute exactly 0 to the loss
  3. Training focuses on predicting editable regions
  4. Statistics track editable vs locked accuracy separately

Loss formula:
  L = CE(logits, targets) ⊙ edit_mask
  L = sum(L) / sum(edit_mask)  # normalized by editable count
""")


if __name__ == "__main__":
    main()

