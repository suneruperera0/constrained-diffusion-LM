#!/usr/bin/env python3
"""
Sampling script for ConstrainedDiffusionLM.

Usage:
    python scripts/sample.py --checkpoint checkpoints/best_model.pt
    python scripts/sample.py --checkpoint checkpoints/best_model.pt --show-process
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from constrained_diffusion_lm.data import Tokenizer
from constrained_diffusion_lm.diffusion import get_schedule, DiffusionSampler
from constrained_diffusion_lm.models import TransformerDenoiser
from constrained_diffusion_lm.inference import generate, generate_with_visualization
from constrained_diffusion_lm.utils.seed import set_seed, get_device
from constrained_diffusion_lm.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample from a trained Constraint-Preserving Diffusion Language Model",
    )

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, choices=["generate", "edit"], default="generate")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show-process", action="store_true", help="Show denoising process")
    parser.add_argument("--confidence-sampling", action="store_true", help="Use confidence-based sampling")
    
    # Model config (should match training)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)

    return parser.parse_args()


def infer_model_config_from_state_dict(state_dict):
    """Infer model architecture from saved state dict."""
    # Get dim from token embedding
    dim = state_dict["token_embedding.weight"].shape[1]
    
    # Get max_seq_len from position embedding
    max_seq_len = state_dict["position_embedding.weight"].shape[0]
    
    # Count transformer layers
    num_layers = 0
    while f"transformer.layers.{num_layers}.self_attn.in_proj_weight" in state_dict:
        num_layers += 1
    
    # Get num_heads from in_proj_weight shape (3 * dim * num_heads, dim)
    in_proj_shape = state_dict["transformer.layers.0.self_attn.in_proj_weight"].shape
    # in_proj_weight is (3*dim, dim), so num_heads = dim / head_dim
    # For standard BERT, head_dim = 64, so num_heads = dim / 64
    num_heads = dim // 64 if dim >= 64 else dim // 32
    
    # Get dim_feedforward from linear1
    dim_feedforward = state_dict["transformer.layers.0.linear1.weight"].shape[0]
    
    return {
        "dim": dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dim_feedforward": dim_feedforward,
        "max_seq_len": max_seq_len,
    }


def main():
    args = parse_args()
    
    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load checkpoint first to infer model config
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Try to get model config from checkpoint, otherwise infer from state dict
    if "model_config" in checkpoint and checkpoint["model_config"]:
        model_config = checkpoint["model_config"]
        logger.info(f"Using saved model config: {model_config}")
    else:
        model_config = infer_model_config_from_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Inferred model config: {model_config}")
    
    # Override max_seq_len if user specified
    max_seq_len = model_config.get("max_seq_len", 256)
    
    # Load tokenizer
    tokenizer = Tokenizer("bert-base-uncased", max_length=max_seq_len)
    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    
    # Create model with correct architecture
    model = TransformerDenoiser(
        vocab_size=tokenizer.vocab_size,
        dim=model_config["dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        dim_feedforward=model_config["dim_feedforward"],
        dropout=0.0,  # No dropout during inference
        max_seq_len=max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded: {model.get_num_params():,} parameters")
    
    # Create noise schedule
    schedule = get_schedule("cosine", args.timesteps)
    
    print("\n" + "=" * 60)
    print("ConstrainedDiffusionLM Generation")
    print("=" * 60)
    
    # Use user-specified length or default to something reasonable
    seq_len = min(args.max_len, max_seq_len) if args.max_len else min(64, max_seq_len)
    
    if args.show_process:
        # Show denoising trajectory for one sample
        print("\nGenerating with denoising visualization...\n")
        
        final_text, trajectory = generate_with_visualization(
            model=model,
            tokenizer=tokenizer,
            schedule=schedule,
            seq_len=seq_len,
            temperature=args.temperature,
            device=device,
            num_steps_to_show=10,
        )
        
        print("Denoising trajectory:")
        print("-" * 60)
        for t, text in trajectory:
            # Highlight [MASK] tokens
            highlighted = text.replace("[MASK]", "\033[91m[MASK]\033[0m")
            print(f"t={t:4d}: {highlighted}")
        print("-" * 60)
        print(f"Final: {final_text}")
        
    else:
        # Generate multiple samples
        print(f"\nGenerating {args.num_samples} samples...")
        print(f"Temperature: {args.temperature}")
        print(f"Sequence length: {seq_len}")
        print("-" * 60)
        
        texts = generate(
            model=model,
            tokenizer=tokenizer,
            schedule=schedule,
            num_samples=args.num_samples,
            seq_len=seq_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            use_confidence_sampling=args.confidence_sampling,
            show_progress=True,
        )
        
        print("\nGenerated samples:")
        print("-" * 60)
        for i, text in enumerate(texts, 1):
            print(f"[{i}] {text}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
