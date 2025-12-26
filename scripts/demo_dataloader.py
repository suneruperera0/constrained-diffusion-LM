#!/usr/bin/env python3
"""
Demo script for Phase 1: Verify dataloader works.

Usage:
    python scripts/demo_dataloader.py
    python scripts/demo_dataloader.py --data data/processed/sample.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constrained_diffusion_lm.data import (
    Tokenizer,
    TextDataset,
    InMemoryTextDataset,
    create_dataloader,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo: Dataloader verification")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data file (jsonl, json, or txt)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace tokenizer name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for demo",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Maximum sequence length",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("Phase 1 Demo: Dataset + Tokenization")
    print("=" * 70)
    
    # Initialize tokenizer
    print(f"\n[1] Loading tokenizer: {args.tokenizer}")
    tokenizer = Tokenizer(args.tokenizer, max_length=args.max_length)
    print(f"    Vocab size: {tokenizer.vocab_size}")
    print(f"    PAD token ID: {tokenizer.pad_token_id}")
    print(f"    MASK token ID: {tokenizer.mask_token_id}")
    
    # Create dataset
    if args.data and Path(args.data).exists():
        print(f"\n[2] Loading dataset from: {args.data}")
        dataset = TextDataset(args.data, tokenizer)
    else:
        print("\n[2] Using in-memory sample texts")
        sample_texts = [
            "The contract is governed by Ontario law.",
            "Please review the document at your earliest convenience.",
            "This agreement shall be binding upon both parties.",
            "The defendant was found guilty of all charges.",
            "We hereby acknowledge receipt of your letter dated January 15.",
            "The terms and conditions are subject to change without notice.",
            "All disputes shall be resolved through arbitration.",
            "The company reserves the right to terminate this agreement.",
        ]
        dataset = InMemoryTextDataset(sample_texts, tokenizer)
    
    print(f"    Dataset size: {len(dataset)} examples")
    
    # Show single example
    print("\n[3] Single example:")
    example = dataset[0]
    print(f"    Text: {example['text']}")
    print(f"    Token IDs: {example['input_ids'].tolist()}")
    print(f"    Length: {example['length']}")
    print(f"    Decoded: {tokenizer.decode(example['input_ids'])}")
    
    # Show tokens
    tokens = [tokenizer.get_token(tid) for tid in example['input_ids'].tolist()]
    print(f"    Tokens: {tokens}")
    
    # Create dataloader
    print(f"\n[4] Creating DataLoader (batch_size={args.batch_size})")
    dataloader = create_dataloader(
        dataset,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    # Get a batch
    print("\n[5] Sample batch:")
    batch = next(iter(dataloader))
    
    print(f"    input_ids shape: {batch['input_ids'].shape}")
    print(f"    attention_mask shape: {batch['attention_mask'].shape}")
    print(f"    lengths: {batch['lengths'].tolist()}")
    
    print("\n    Batch contents:")
    for i in range(min(args.batch_size, len(batch['texts']))):
        ids = batch['input_ids'][i]
        mask = batch['attention_mask'][i]
        length = batch['lengths'][i].item()
        text = batch['texts'][i]
        
        print(f"\n    [{i}] Original: {text}")
        print(f"        Token IDs (first {length}): {ids[:length].tolist()}")
        print(f"        Attention mask sum: {mask.sum().item()} (should equal length)")
        print(f"        Decoded: {tokenizer.decode(ids)}")
    
    # Sanity checks
    print("\n[6] Sanity checks:")
    
    # Check padding is correct
    for i in range(len(batch['texts'])):
        length = batch['lengths'][i].item()
        mask = batch['attention_mask'][i]
        ids = batch['input_ids'][i]
        
        # Mask should be 1 for first `length` tokens
        assert mask[:length].sum() == length, "Attention mask mismatch!"
        # Mask should be 0 after `length`
        assert mask[length:].sum() == 0, "Padding mask should be 0!"
        # Padding tokens should be pad_token_id
        if length < len(ids):
            assert (ids[length:] == tokenizer.pad_token_id).all(), "Padding token mismatch!"
    
    print("    ✓ Attention masks are correct")
    print("    ✓ Padding is correct")
    print("    ✓ All sanity checks passed!")
    
    print("\n" + "=" * 70)
    print("Phase 1 Complete: Dataloader is working!")
    print("=" * 70)


if __name__ == "__main__":
    main()

