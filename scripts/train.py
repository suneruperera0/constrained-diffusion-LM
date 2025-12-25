#!/usr/bin/env python3
"""
Training script for ConstrainedDiffusionLM.

Usage:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --help
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Constraint-Preserving Diffusion Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default config
    python scripts/train.py --config configs/train.yaml

    # Train with custom settings
    python scripts/train.py --config configs/train.yaml --epochs 100 --lr 1e-4

    # Resume from checkpoint
    python scripts/train.py --config configs/train.yaml --resume checkpoints/latest.pt
        """,
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config file (default: configs/train.yaml)",
    )

    # Model
    parser.add_argument(
        "--model-dim",
        type=int,
        default=None,
        help="Model hidden dimension (overrides config)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of transformer layers (overrides config)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Number of attention heads (overrides config)",
    )

    # Diffusion
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Number of diffusion timesteps (overrides config)",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        choices=["linear", "cosine", "sqrt"],
        default=None,
        help="Noise schedule type (overrides config)",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay (overrides config)",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Gradient clipping norm (overrides config)",
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (overrides config)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Maximum sequence length (overrides config)",
    )

    # Checkpointing
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save checkpoint every N epochs (overrides config)",
    )

    # Logging
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="constrained-diffusion-lm",
        help="W&B project name",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Log metrics every N steps (overrides config)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to train on (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (smaller model, fewer steps)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one batch only to verify setup",
    )

    return parser.parse_args()


def main() -> None:
    """Main training entrypoint."""
    args = parse_args()

    # TODO: Phase 1+ will implement actual training
    print("=" * 60)
    print("ConstrainedDiffusionLM Training")
    print("=" * 60)
    print(f"\nConfig: {args.config}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")

    if args.debug:
        print("\n[DEBUG MODE ENABLED]")
    if args.dry_run:
        print("\n[DRY RUN MODE]")

    print("\n✓ CLI is working! Training logic will be implemented in Phase 3.")
    print("=" * 60)


if __name__ == "__main__":
    main()

