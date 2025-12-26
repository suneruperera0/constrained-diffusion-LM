#!/usr/bin/env python3
"""
Demo script for Phase 2: Visualize forward corruption over timesteps.

Usage:
    python scripts/demo_corruption.py
    python scripts/demo_corruption.py --schedule cosine --timesteps 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from constrained_diffusion_lm.data import Tokenizer, InMemoryTextDataset, create_dataloader
from constrained_diffusion_lm.data.corruption import MaskCorruptor
from constrained_diffusion_lm.diffusion import get_schedule


def parse_args():
    parser = argparse.ArgumentParser(description="Demo: Corruption visualization")
    parser.add_argument(
        "--schedule",
        type=str,
        choices=["linear", "cosine", "sqrt"],
        default="cosine",
        help="Noise schedule type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="Total number of diffusion timesteps T",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace tokenizer name",
    )
    return parser.parse_args()


def visualize_token_sequence(tokens: list, mask_token: str = "[MASK]") -> str:
    """Create a visual representation of tokens with masks highlighted."""
    result = []
    for tok in tokens:
        if tok == mask_token:
            result.append(f"\033[91m{tok}\033[0m")  # Red for masked
        elif tok in ["[CLS]", "[SEP]", "[PAD]"]:
            result.append(f"\033[90m{tok}\033[0m")  # Gray for special
        else:
            result.append(tok)
    return " ".join(result)


def main():
    args = parse_args()
    
    print("=" * 70)
    print("Phase 2 Demo: Forward Corruption (Mask Diffusion)")
    print("=" * 70)
    
    # Initialize components
    print(f"\n[1] Setup")
    print(f"    Schedule: {args.schedule}")
    print(f"    Timesteps T: {args.timesteps}")
    
    tokenizer = Tokenizer(args.tokenizer, max_length=64)
    schedule = get_schedule(args.schedule, args.timesteps)
    corruptor = MaskCorruptor(schedule, tokenizer.mask_token_id)
    
    # Show schedule curve
    print(f"\n[2] Noise Schedule gamma(t):")
    sample_points = [0, args.timesteps // 4, args.timesteps // 2, 
                     3 * args.timesteps // 4, args.timesteps]
    for t_val in sample_points:
        gamma = schedule(torch.tensor(t_val)).item()
        bar = "█" * int(gamma * 40)
        print(f"    t={t_val:4d}: gamma={gamma:.3f} |{bar}")
    
    # Sample text
    sample_texts = [
        "The contract is governed by Ontario law and shall be binding.",
    ]
    
    dataset = InMemoryTextDataset(sample_texts, tokenizer)
    dataloader = create_dataloader(dataset, tokenizer, batch_size=1, shuffle=False)
    batch = next(iter(dataloader))
    
    x_0 = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    
    print(f"\n[3] Original sequence x₀:")
    original_tokens = [tokenizer.get_token(tid) for tid in x_0[0].tolist()]
    print(f"    {visualize_token_sequence(original_tokens)}")
    print(f"    Decoded: {tokenizer.decode(x_0[0])}")
    
    # Show corruption at different timesteps
    print(f"\n[4] Corruption at different timesteps:")
    
    timesteps_to_show = [1, args.timesteps // 4, args.timesteps // 2, 
                         3 * args.timesteps // 4, args.timesteps]
    
    for t_val in timesteps_to_show:
        t = torch.tensor([t_val])
        gamma = schedule(t).item()
        
        # Do multiple samples to show stochasticity
        x_t, noise_mask = corruptor(x_0, t, attention_mask)
        
        tokens = [tokenizer.get_token(tid) for tid in x_t[0].tolist()]
        num_masked = noise_mask[0].sum().item()
        total_tokens = attention_mask[0].sum().item()
        
        print(f"\n    t={t_val} (gamma={gamma:.3f}, expected {gamma*100:.1f}% masked)")
        print(f"    {visualize_token_sequence(tokens)}")
        print(f"    Actually masked: {num_masked}/{total_tokens} = {100*num_masked/total_tokens:.1f}%")
    
    # Show multiple samples at same timestep to demonstrate stochasticity
    print(f"\n[5] Stochasticity: 5 different corruptions at t={args.timesteps//2}")
    t = torch.tensor([args.timesteps // 2])
    
    for i in range(5):
        x_t, noise_mask = corruptor(x_0, t, attention_mask)
        tokens = [tokenizer.get_token(tid) for tid in x_t[0].tolist()]
        num_masked = noise_mask[0].sum().item()
        print(f"    Sample {i+1}: {visualize_token_sequence(tokens)} ({num_masked} masked)")
    
    # Compare schedules
    print(f"\n[6] Schedule comparison at t=T/2 ({args.timesteps//2}):")
    t_mid = torch.tensor(args.timesteps // 2)
    for sched_type in ["linear", "cosine", "sqrt"]:
        sched = get_schedule(sched_type, args.timesteps)
        gamma = sched(t_mid).item()
        print(f"    {sched_type:8s}: gamma={gamma:.3f}")
    
    print("\n" + "=" * 70)
    print("Phase 2 Complete: Forward corruption working!")
    print("=" * 70)
    print("\nKey insight: As t increases from 0 to T, more tokens get masked.")
    print("The model will learn to predict the original tokens from these corrupted versions.")


if __name__ == "__main__":
    main()

