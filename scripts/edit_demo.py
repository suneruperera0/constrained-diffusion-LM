#!/usr/bin/env python3
"""
Interactive demo for constraint-preserving text editing.

This is the FINAL DELIVERABLE for Phase 8.

Usage:
    python scripts/edit_demo.py --checkpoint checkpoints/best_model.pt
    python scripts/edit_demo.py --checkpoint checkpoints/best_model.pt --examples
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from constrained_diffusion_lm.data import Tokenizer
from constrained_diffusion_lm.diffusion import get_schedule, ConstrainedDiffusionSampler
from constrained_diffusion_lm.models import TransformerDenoiser
from constrained_diffusion_lm.inference import edit_text, edit_text_with_trajectory
from constrained_diffusion_lm.utils.seed import set_seed, get_device
from constrained_diffusion_lm.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo: Constraint-Preserving Text Editing with Diffusion LM",
    )
    
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--lock", type=str, default=None)
    parser.add_argument("--examples", action="store_true")
    parser.add_argument("--show-trajectory", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    
    # Model config (must match training)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=64)
    
    return parser.parse_args()


def visualize_edit(original_tokens, edited_tokens, lock_mask, tokenizer):
    """Create colored visualization of edit."""
    lines = []
    
    # Original
    orig_str = []
    for i, tok in enumerate(original_tokens):
        if lock_mask[i]:
            orig_str.append(f"\033[92m{tok}\033[0m")  # Green = locked
        else:
            orig_str.append(f"\033[93m{tok}\033[0m")  # Yellow = editable
    lines.append("Original: " + " ".join(orig_str))
    
    # Edited
    edit_str = []
    for i, tok in enumerate(edited_tokens):
        if i < len(lock_mask) and lock_mask[i]:
            edit_str.append(f"\033[92m{tok}\033[0m")  # Green = locked (preserved)
        else:
            # Check if changed
            if i < len(original_tokens) and tok == original_tokens[i]:
                edit_str.append(f"\033[93m{tok}\033[0m")  # Yellow = same
            else:
                edit_str.append(f"\033[91m{tok}\033[0m")  # Red = changed
    lines.append("Edited:   " + " ".join(edit_str))
    
    return "\n".join(lines)


def run_example(model, tokenizer, schedule, text, lock_text, device, temperature, show_trajectory=False):
    """Run a single edit example."""
    print(f"\n{'─' * 60}")
    print(f"Input:  \"{text}\"")
    print(f"Locked: \"{lock_text}\"")
    print(f"{'─' * 60}")
    
    if show_trajectory:
        result, trajectory = edit_text_with_trajectory(
            model=model,
            tokenizer=tokenizer,
            schedule=schedule,
            text=text,
            lock_substring_text=lock_text,
            temperature=temperature,
            device=device,
            num_steps_to_show=8,
        )
        
        print("\nDenoising trajectory:")
        for t, text_at_t in trajectory:
            # Highlight [MASK]
            highlighted = text_at_t.replace("[MASK]", "\033[90m[MASK]\033[0m")
            print(f"  t={t:3d}: {highlighted}")
    else:
        result = edit_text(
            model=model,
            tokenizer=tokenizer,
            schedule=schedule,
            text=text,
            lock_substring_text=lock_text,
            temperature=temperature,
            device=device,
            show_progress=False,
        )
    
    print(f"\n{visualize_edit(result.original_tokens, result.edited_tokens, result.lock_mask, tokenizer)}")
    print(f"\nOutput: \"{result.edited_text}\"")
    print(f"\n📊 Metrics:")
    print(f"   Constraint preserved: {'✓ YES' if result.constraint_preserved else '✗ NO'}")
    print(f"   Preservation rate:    {result.preservation_rate * 100:.1f}%")
    
    # Show which tokens were locked
    locked_tokens = [result.original_tokens[i] for i in range(len(result.original_tokens)) if result.lock_mask[i]]
    print(f"   Locked tokens:        {locked_tokens}")
    
    return result


def main():
    args = parse_args()
    
    set_seed(args.seed)
    device = get_device(args.device)
    
    print("=" * 60)
    print("ConstrainedDiffusionLM - Edit Demo")
    print("=" * 60)
    print(f"Device: {device}")
    
    # Load tokenizer
    tokenizer = Tokenizer("bert-base-uncased", max_length=args.max_len)
    
    # Load model
    model = TransformerDenoiser(
        vocab_size=tokenizer.vocab_size,
        dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.model_dim * 2,
        dropout=0.0,
        max_seq_len=args.max_len,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # Create schedule
    schedule = get_schedule("cosine", args.timesteps)
    
    print(f"\nModel: {model.get_num_params():,} parameters")
    print(f"Timesteps: {args.timesteps}")
    print(f"Temperature: {args.temperature}")
    
    if args.examples:
        # Run predefined examples
        examples = [
            {
                "text": "The contract is governed by Ontario law.",
                "lock": "Ontario law",
            },
            {
                "text": "Please review the document at your earliest convenience.",
                "lock": "review the document",
            },
            {
                "text": "All disputes shall be resolved through arbitration.",
                "lock": "arbitration",
            },
            {
                "text": "Payment is due within thirty days of invoice date.",
                "lock": "thirty days",
            },
        ]
        
        print(f"\n{'═' * 60}")
        print("Running {len(examples)} predefined examples...")
        print(f"{'═' * 60}")
        
        all_preserved = True
        for i, ex in enumerate(examples, 1):
            print(f"\n[Example {i}/{len(examples)}]")
            result = run_example(
                model, tokenizer, schedule,
                ex["text"], ex["lock"],
                device, args.temperature,
                show_trajectory=args.show_trajectory,
            )
            if not result.constraint_preserved:
                all_preserved = False
        
        print(f"\n{'═' * 60}")
        print("Summary")
        print(f"{'═' * 60}")
        if all_preserved:
            print("✓ All examples preserved locked tokens exactly!")
        else:
            print("✗ Some examples failed to preserve locked tokens")
            
    elif args.text and args.lock:
        # Run single example from command line
        run_example(
            model, tokenizer, schedule,
            args.text, args.lock,
            device, args.temperature,
            show_trajectory=args.show_trajectory,
        )
    else:
        # Interactive mode or show usage
        print("\nUsage examples:")
        print(f"  python {sys.argv[0]} --checkpoint {args.checkpoint} --examples")
        print(f"  python {sys.argv[0]} --checkpoint {args.checkpoint} --text \"Your text\" --lock \"locked part\"")
        print(f"  python {sys.argv[0]} --checkpoint {args.checkpoint} --examples --show-trajectory")
    
    print(f"\n{'═' * 60}")
    print("Legend:")
    print("  \033[92mGreen\033[0m  = Locked (preserved exactly)")
    print("  \033[93mYellow\033[0m = Editable (unchanged)")
    print("  \033[91mRed\033[0m    = Editable (changed)")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
