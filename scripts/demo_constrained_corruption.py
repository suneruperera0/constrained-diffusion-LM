#!/usr/bin/env python3
"""
Demo script for Phase 6: Constrained corruption visualization.

Shows that locked tokens are NEVER masked during forward diffusion.

Usage:
    python scripts/demo_constrained_corruption.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from constrained_diffusion_lm.data import Tokenizer, lock_substring
from constrained_diffusion_lm.data.corruption import MaskCorruptor, ConstrainedMaskCorruptor
from constrained_diffusion_lm.diffusion import get_schedule


def visualize_corruption(tokens: list, original_tokens: list, lock_mask: torch.Tensor, mask_token: str = "[MASK]") -> str:
    """Create colored visualization of corrupted tokens."""
    result = []
    for i, (tok, orig) in enumerate(zip(tokens, original_tokens)):
        if tok == mask_token:
            result.append(f"\033[91m[MASK]\033[0m")  # Red for masked
        elif i < len(lock_mask) and lock_mask[i]:
            result.append(f"\033[92m{tok}\033[0m")   # Green for locked (preserved)
        else:
            result.append(f"\033[93m{tok}\033[0m")   # Yellow for editable (not masked this time)
    return " ".join(result)


def main():
    print("=" * 70)
    print("Phase 6 Demo: Constrained Forward Corruption")
    print("=" * 70)
    
    # Setup
    tokenizer = Tokenizer("bert-base-uncased", max_length=64)
    schedule = get_schedule("cosine", num_timesteps=100)
    corruptor = MaskCorruptor(schedule, tokenizer.mask_token_id)
    
    # Example text with locked substring
    text = "The contract is governed by Ontario law and shall be binding."
    locked_substring = "Ontario law"
    
    # Create lock mask
    lock_mask, edit_mask, spans = lock_substring(text, locked_substring, tokenizer)
    
    # Tokenize
    token_ids = tokenizer.encode(text)
    original_tokens = [tokenizer.get_token(tid) for tid in token_ids]
    
    print(f"\n[Setup]")
    print(f"Text: {text}")
    print(f"Locked: '{locked_substring}' (token spans: {spans})")
    print(f"Tokens: {original_tokens}")
    print(f"Lock mask: {lock_mask.tolist()}")
    
    # Prepare batch
    x_0 = torch.tensor([token_ids])
    lock_mask_batch = lock_mask.unsqueeze(0)
    
    # =========================================================================
    # Comparison: Unconstrained vs Constrained Corruption
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("Comparison: Unconstrained vs Constrained Corruption")
    print("=" * 70)
    
    timesteps = [10, 25, 50, 75, 100]
    
    print(f"\n{'t':>4} | {'Unconstrained':<40} | {'Constrained':<40}")
    print("-" * 90)
    
    for t_val in timesteps:
        t = torch.tensor([t_val])
        gamma = schedule(t).item()
        
        # Unconstrained corruption (no lock_mask)
        x_t_unconstrained, mask_unconstrained = corruptor(x_0, t)
        tokens_unconstrained = [tokenizer.get_token(tid) for tid in x_t_unconstrained[0].tolist()]
        
        # Constrained corruption (with lock_mask)
        x_t_constrained, mask_constrained = corruptor(x_0, t, lock_mask=lock_mask_batch)
        tokens_constrained = [tokenizer.get_token(tid) for tid in x_t_constrained[0].tolist()]
        
        # Count masks
        n_masked_unconstrained = mask_unconstrained.sum().item()
        n_masked_constrained = mask_constrained.sum().item()
        
        print(f"{t_val:>4} | masked={n_masked_unconstrained:>2} {visualize_corruption(tokens_unconstrained, original_tokens, lock_mask)[:40]}")
        print(f"     | masked={n_masked_constrained:>2} {visualize_corruption(tokens_constrained, original_tokens, lock_mask)[:40]}")
        print()
    
    # =========================================================================
    # Verify Locked Tokens are NEVER Corrupted
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("Verification: Locked tokens preserved at ALL timesteps")
    print("=" * 70)
    
    constrained_corruptor = ConstrainedMaskCorruptor(schedule, tokenizer.mask_token_id)
    
    all_preserved = True
    for t_val in range(1, 101):
        t = torch.tensor([t_val])
        x_t, _ = constrained_corruptor(x_0, t, lock_mask_batch)
        
        # Check locked positions
        preserved = constrained_corruptor.verify_constraints(x_0, x_t, lock_mask_batch)
        if not preserved:
            print(f"  ✗ t={t_val}: Locked tokens were corrupted!")
            all_preserved = False
    
    if all_preserved:
        print(f"\n  ✓ All 100 timesteps verified: Locked tokens are NEVER corrupted!")
    
    # =========================================================================
    # Detailed View at High Corruption
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("Detailed View: t=100 (maximum corruption)")
    print("=" * 70)
    
    t = torch.tensor([100])
    
    # Show 5 different random corruptions
    print("\n5 random constrained corruptions at t=100:")
    print(f"\nLegend: \033[92mGreen\033[0m = Locked, \033[91mRed\033[0m = Masked, \033[93mYellow\033[0m = Editable (not masked)")
    print("-" * 70)
    
    for i in range(5):
        x_t, noise_mask = constrained_corruptor(x_0, t, lock_mask_batch)
        tokens = [tokenizer.get_token(tid) for tid in x_t[0].tolist()]
        n_masked = noise_mask.sum().item()
        
        print(f"[{i+1}] {visualize_corruption(tokens, original_tokens, lock_mask)} ({n_masked} masked)")
    
    # Show locked tokens explicitly
    print(f"\nLocked tokens (positions {spans}):")
    for start, end in spans:
        locked_tokens = original_tokens[start:end]
        print(f"  {locked_tokens} — ALWAYS preserved")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("Phase 6 Complete: Constrained corruption working!")
    print("=" * 70)
    print("""
Key property verified:
  x_t[lock_mask] == x_0[lock_mask]  for ALL timesteps t

This ensures:
  - Locked tokens are NEVER replaced with [MASK]
  - During training, the model learns to predict around constraints
  - During inference, locked tokens will remain exactly as specified
""")


if __name__ == "__main__":
    main()

