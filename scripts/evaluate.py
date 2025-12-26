#!/usr/bin/env python3
"""
Evaluate a trained diffusion LM on constraint-preserving editing.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from constrained_diffusion_lm.data.tokenization import Tokenizer
from constrained_diffusion_lm.models.transformer import TransformerDenoiser
from constrained_diffusion_lm.diffusion.schedule import get_schedule
from constrained_diffusion_lm.diffusion.sampler import ConstrainedDiffusionSampler
from constrained_diffusion_lm.inference.edit import edit_text
from constrained_diffusion_lm.eval.edit_metrics import (
    compute_edit_metrics,
    compute_batch_metrics,
    print_metrics_table,
    EditMetrics,
)
from constrained_diffusion_lm.training.trainer import TrainingConfig


# Test cases for evaluation
TEST_CASES = [
    # (text, locked_spans_description)
    ("The quick brown fox jumps over the lazy dog.", [(0, 3), (4, 9)]),  # Lock "The" and "quick"
    ("Machine learning is transforming artificial intelligence research.", [(0, 16)]),  # Lock "Machine learning"
    ("I love programming in Python because it is so elegant.", [(18, 24)]),  # Lock "Python"
    ("The weather today is absolutely perfect for a picnic.", [(0, 3), (12, 15)]),  # Lock "The" and "is"
    ("Deep neural networks can learn complex patterns from data.", [(0, 4), (21, 26)]),  # Lock "Deep" and "learn"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate constraint-preserving editing")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of test samples to evaluate",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-sample output",
    )
    return parser.parse_args()


def create_lock_mask_from_spans(
    text: str,
    spans: List[Tuple[int, int]],
    tokenizer: Tokenizer,
    max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Create a lock mask from character-level spans.
    
    Returns:
        token_ids: Token IDs
        lock_mask: Boolean mask (True = locked)
        locked_tokens: List of locked token strings
    """
    # Tokenize
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    
    # Truncate if needed
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    
    # Create character-to-token mapping
    # This is simplified - in practice, you'd use the tokenizer's offset_mapping
    tokens = tokenizer.tokenizer.tokenize(text)
    
    # Initialize lock mask
    lock_mask = torch.zeros(len(token_ids), dtype=torch.bool)
    locked_tokens = []
    
    # Lock [CLS] and [SEP] tokens
    lock_mask[0] = True  # CLS
    if len(token_ids) > 1:
        lock_mask[-1] = True  # SEP
    
    # Lock specified spans (convert char positions to approximate token positions)
    char_pos = 0
    for i, token in enumerate(tokens):
        token_text = token.replace("##", "")
        token_start = char_pos
        token_end = char_pos + len(token_text)
        
        # Check if this token falls within any locked span
        for span_start, span_end in spans:
            if token_start >= span_start and token_end <= span_end:
                # Token index + 1 to account for [CLS]
                if i + 1 < len(lock_mask):
                    lock_mask[i + 1] = True
                    locked_tokens.append(token)
        
        char_pos = token_end
        # Skip whitespace
        if char_pos < len(text) and text[char_pos] == " ":
            char_pos += 1
    
    return torch.tensor(token_ids), lock_mask, locked_tokens


def evaluate_single(
    text: str,
    spans: List[Tuple[int, int]],
    model: TransformerDenoiser,
    tokenizer: Tokenizer,
    sampler: ConstrainedDiffusionSampler,
    max_len: int,
    temperature: float,
    device: torch.device,
    verbose: bool = False,
) -> EditMetrics:
    """Evaluate a single editing example."""
    
    # Tokenize and create lock mask
    token_ids, lock_mask, locked_tokens = create_lock_mask_from_spans(
        text, spans, tokenizer, max_len
    )
    
    # Move to device
    token_ids = token_ids.to(device)
    lock_mask = lock_mask.to(device)
    
    # Perform edit
    edited_ids = sampler.edit(
        x_0=token_ids.unsqueeze(0),
        lock_mask=lock_mask.unsqueeze(0),
        temperature=temperature,
    )
    
    # Compute metrics
    metrics = compute_edit_metrics(
        original_ids=token_ids,
        edited_ids=edited_ids.squeeze(0),
        lock_mask=lock_mask,
    )
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"Original:  {text}")
        edited_text = tokenizer.decode(edited_ids.squeeze(0).tolist(), skip_special_tokens=True)
        print(f"Edited:    {edited_text}")
        print(f"Locked:    {locked_tokens}")
        print(f"Fidelity:  {metrics.constraint_fidelity * 100:.1f}%")
        print(f"Edit rate: {metrics.edit_rate * 100:.1f}%")
        print(f"Drift:     {metrics.drift * 100:.1f}%")
    
    return metrics


def main():
    args = parse_args()
    
    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    torch.serialization.add_safe_globals([TrainingConfig])
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Load tokenizer
    model_name = checkpoint.get("model_name", "bert-base-uncased")
    tokenizer = Tokenizer(model_name)
    
    # Get model config
    model_config = checkpoint.get("model_config", {})
    max_seq_len = model_config.get("max_seq_len", 64)
    
    # Create model
    model = TransformerDenoiser(
        vocab_size=tokenizer.vocab_size,
        dim=model_config.get("dim", 128),
        num_heads=model_config.get("num_heads", 4),
        num_layers=model_config.get("num_layers", 2),
        dim_feedforward=model_config.get("dim_feedforward", 256),
        max_seq_len=max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Create schedule and sampler
    schedule = get_schedule("cosine", args.steps)
    sampler = ConstrainedDiffusionSampler(
        model=model,
        schedule=schedule,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    print(f"\nEvaluating {min(args.num_samples, len(TEST_CASES))} test cases...")
    print("=" * 60)
    
    # Evaluate test cases
    all_metrics = []
    
    for i, (text, spans) in enumerate(TEST_CASES[:args.num_samples]):
        metrics = evaluate_single(
            text=text,
            spans=spans,
            model=model,
            tokenizer=tokenizer,
            sampler=sampler,
            max_len=max_seq_len,
            temperature=args.temperature,
            device=device,
            verbose=args.verbose,
        )
        all_metrics.append(metrics)
    
    # Compute aggregate metrics
    batch_metrics = compute_batch_metrics(all_metrics)
    
    # Print results
    print("\n" + "=" * 60)
    print(batch_metrics.summary())
    
    if args.verbose:
        print_metrics_table(all_metrics)
    
    # Print copy-pasteable summary
    print("\n" + "=" * 60)
    print("Copy-pasteable summary for README:")
    print("-" * 60)
    print(f"Constraint Fidelity: {batch_metrics.mean_constraint_fidelity * 100:.1f}% (min: {batch_metrics.min_constraint_fidelity * 100:.1f}%, max: {batch_metrics.max_constraint_fidelity * 100:.1f}%)")
    print(f"Perfect Preservation Rate: {batch_metrics.perfect_constraint_rate * 100:.0f}%")
    print(f"Mean Edit Rate: {batch_metrics.mean_edit_rate * 100:.1f}%")
    print(f"Zero Drift Rate: {batch_metrics.zero_drift_rate * 100:.0f}%")


if __name__ == "__main__":
    main()

