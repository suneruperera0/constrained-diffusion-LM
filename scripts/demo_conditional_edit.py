#!/usr/bin/env python3
"""
Demo script for Phase 9: Instruction-conditioned text editing.

Shows how different edit instructions produce different outputs.

Usage:
    python scripts/demo_conditional_edit.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from constrained_diffusion_lm.data import Tokenizer
from constrained_diffusion_lm.diffusion import get_schedule
from constrained_diffusion_lm.models import TransformerDenoiser
from constrained_diffusion_lm.inference import conditional_edit
from constrained_diffusion_lm.conditioning import PromptEncoder
from constrained_diffusion_lm.utils.seed import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo: Instruction-Conditioned Text Editing",
    )
    
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    
    # Model config
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=64)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    set_seed(args.seed)
    device = get_device(args.device)
    
    print("=" * 70)
    print("Phase 9 Demo: Instruction-Conditioned Editing")
    print("=" * 70)
    
    # Load components
    tokenizer = Tokenizer("bert-base-uncased", max_length=args.max_len)
    
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
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    schedule = get_schedule("cosine", args.timesteps)
    
    print(f"\nDevice: {device}")
    print(f"Model: {model.get_num_params():,} parameters")
    
    # Show available instruction templates
    encoder = PromptEncoder(tokenizer)
    print(f"\nAvailable instruction templates:")
    for template in encoder.list_templates():
        print(f"  - {template}: \"{encoder.get_template(template)}\"")
    
    # Test text
    text = "The contract is governed by Ontario law."
    lock = "Ontario law"
    
    print(f"\n{'═' * 70}")
    print(f"Input text:  \"{text}\"")
    print(f"Locked:      \"{lock}\"")
    print(f"{'═' * 70}")
    
    # Try different instructions
    instructions = [
        "formal",
        "aggressive", 
        "shorter",
        "Make this sound more professional",
        "Add more detail",
    ]
    
    print("\nEditing with different instructions:")
    print("-" * 70)
    
    for instruction in instructions:
        # Get full instruction text
        full_instruction = encoder.get_template(instruction) if instruction.lower() in encoder.INSTRUCTION_TEMPLATES else instruction
        
        result = conditional_edit(
            model=model,
            tokenizer=tokenizer,
            schedule=schedule,
            text=text,
            instruction=instruction,
            lock_substring_text=lock,
            temperature=args.temperature,
            device=device,
        )
        
        # Highlight locked portion in output
        edited = result.edited_text
        if lock.lower() in edited.lower():
            edited = edited.replace(lock.lower(), f"\033[92m{lock.lower()}\033[0m")
        
        print(f"\n[{instruction}]")
        print(f"  Instruction: \"{full_instruction}\"")
        print(f"  Output:      \"{edited}\"")
        print(f"  Preserved:   {'✓' if result.constraint_preserved else '✗'} ({result.preservation_rate*100:.0f}%)")
    
    # Show the input format
    print(f"\n{'═' * 70}")
    print("Input Format Explanation")
    print(f"{'═' * 70}")
    
    from constrained_diffusion_lm.conditioning import create_prompted_input
    prompted = create_prompted_input(tokenizer, text, "formal", lock)
    
    tokens = [tokenizer.get_token(tid) for tid in prompted.input_ids.tolist()]
    
    print("\nTokenized input structure:")
    print(f"  Full sequence ({len(tokens)} tokens):")
    
    # Color code the tokens
    token_strs = []
    for i, tok in enumerate(tokens):
        if i < prompted.prompt_length:
            token_strs.append(f"\033[94m{tok}\033[0m")  # Blue = instruction
        elif prompted.full_lock_mask[i]:
            token_strs.append(f"\033[92m{tok}\033[0m")  # Green = locked
        else:
            token_strs.append(f"\033[93m{tok}\033[0m")  # Yellow = editable
    
    print(f"  {' '.join(token_strs)}")
    print(f"\n  Prompt length: {prompted.prompt_length} tokens")
    print(f"  Text starts at: index {prompted.text_start_idx}")
    
    print(f"\nLegend:")
    print(f"  \033[94mBlue\033[0m   = Instruction (always locked)")
    print(f"  \033[92mGreen\033[0m  = Locked text (preserved)")
    print(f"  \033[93mYellow\033[0m = Editable text")
    
    print(f"\n{'═' * 70}")
    print("Phase 9 Complete: Instruction-conditioned editing working!")
    print(f"{'═' * 70}")
    print("""
The model sees: "[CLS] Make this more formal: [SEP] <text> [SEP]"
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 Always locked (never masked)

Different instructions lead to different edit styles.
(Note: Output quality depends on training. Our 3-epoch model
produces gibberish, but the ARCHITECTURE is correct!)
""")


if __name__ == "__main__":
    main()

