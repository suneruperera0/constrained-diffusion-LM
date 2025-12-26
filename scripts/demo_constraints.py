#!/usr/bin/env python3
"""
Demo script for Phase 5: Constraint visualization.

Usage:
    python scripts/demo_constraints.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from constrained_diffusion_lm.data import (
    Tokenizer,
    ConstraintSpec,
    create_lock_mask,
    create_edit_mask,
    create_masks_from_spec,
    lock_substring,
    visualize_constraints,
)


def main():
    print("=" * 70)
    print("Phase 5 Demo: Constraint Masks (Lock/Edit)")
    print("=" * 70)
    
    # Initialize tokenizer
    tokenizer = Tokenizer("bert-base-uncased", max_length=64)
    
    # Example 1: Lock by span indices
    print("\n" + "-" * 70)
    print("[Example 1] Lock specific token spans")
    print("-" * 70)
    
    text = "The contract is governed by Ontario law."
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.get_token(tid) for tid in token_ids]
    
    print(f"\nText: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token indices: {list(range(len(tokens)))}")
    
    # Lock tokens 5-8 (indices for "governed by Ontario law")
    lock_mask = create_lock_mask(len(tokens), lock_spans=[(5, 9)])
    edit_mask = ~lock_mask
    
    print(f"\nLocking span (5, 9):")
    print(f"Lock mask: {lock_mask.tolist()}")
    print(f"Edit mask: {edit_mask.tolist()}")
    
    print(f"\n{visualize_constraints(text, lock_mask, tokenizer)}")
    
    # Example 2: Lock prefix and suffix
    print("\n" + "-" * 70)
    print("[Example 2] Lock prefix and suffix")
    print("-" * 70)
    
    text = "Dear Sir, please review the terms at your convenience. Regards."
    token_ids = tokenizer.encode(text)
    
    # Lock first 3 tokens (greeting) and last 2 tokens (closing)
    lock_mask = create_lock_mask(len(token_ids), lock_prefix=4, lock_suffix=3)
    
    print(f"\nText: {text}")
    print(f"Locking: prefix=4, suffix=3")
    print(f"\n{visualize_constraints(text, lock_mask, tokenizer)}")
    
    # Example 3: Lock by substring
    print("\n" + "-" * 70)
    print("[Example 3] Lock by substring matching")
    print("-" * 70)
    
    text = "The contract is governed by Ontario law and shall be binding."
    substring = "Ontario law"
    
    lock_mask, edit_mask, spans = lock_substring(text, substring, tokenizer)
    
    print(f"\nText: {text}")
    print(f"Locking substring: '{substring}'")
    print(f"Found at token spans: {spans}")
    print(f"\n{visualize_constraints(text, lock_mask, tokenizer)}")
    
    # Example 4: Using ConstraintSpec
    print("\n" + "-" * 70)
    print("[Example 4] Using ConstraintSpec dataclass")
    print("-" * 70)
    
    text = "This agreement supersedes all prior agreements between the parties."
    token_ids = tokenizer.encode(text)
    
    spec = ConstraintSpec(
        lock_spans=[(3, 6)],  # "supersedes all prior"
        lock_suffix=2,        # "parties ."
    )
    
    lock_mask, edit_mask = create_masks_from_spec(len(token_ids), spec)
    
    print(f"\nText: {text}")
    print(f"Spec: lock_spans={(3, 6)}, lock_suffix=2")
    print(f"\n{visualize_constraints(text, lock_mask, tokenizer)}")
    
    # Example 5: Edit spans (inverse of lock)
    print("\n" + "-" * 70)
    print("[Example 5] Define editable spans instead")
    print("-" * 70)
    
    text = "Payment is due within thirty days of invoice date."
    token_ids = tokenizer.encode(text)
    
    # Only allow editing the number of days (tokens 5-6)
    spec = ConstraintSpec(edit_spans=[(5, 7)])  # "thirty days"
    lock_mask, edit_mask = create_masks_from_spec(len(token_ids), spec)
    
    print(f"\nText: {text}")
    print(f"Spec: edit_spans=[(5, 7)] (only 'thirty days' is editable)")
    print(f"\n{visualize_constraints(text, lock_mask, tokenizer)}")
    
    # Example 6: Multiple lock regions
    print("\n" + "-" * 70)
    print("[Example 6] Multiple lock regions")
    print("-" * 70)
    
    text = "All disputes shall be resolved through arbitration in New York."
    token_ids = tokenizer.encode(text)
    
    lock_mask = create_lock_mask(
        len(token_ids),
        lock_spans=[
            (1, 3),   # "disputes shall"
            (6, 8),   # "arbitration in"
        ]
    )
    
    print(f"\nText: {text}")
    print(f"Locking multiple spans: (1,3) and (6,8)")
    print(f"\n{visualize_constraints(text, lock_mask, tokenizer)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Phase 5 Complete: Constraint masks working!")
    print("=" * 70)
    print("""
Summary:
- lock_mask: True = token is LOCKED (preserved during diffusion)
- edit_mask: True = token is EDITABLE (can be modified)
- lock_mask = ~edit_mask (they are complements)

Use cases:
- Lock legal boilerplate, edit specific clauses
- Lock names/entities, edit surrounding context  
- Lock structure, edit content
""")


if __name__ == "__main__":
    main()

