#!/usr/bin/env python3
"""
Sampling script for ConstrainedDiffusionLM.

Usage:
    python scripts/sample.py --checkpoint checkpoints/model.pt
    python scripts/sample.py --help
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sample from a trained Constraint-Preserving Diffusion Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Unconditional generation
    python scripts/sample.py --checkpoint checkpoints/model.pt --mode generate

    # Constraint-preserving editing
    python scripts/sample.py --checkpoint checkpoints/model.pt --mode edit \\
        --input "The contract is governed by Ontario law." \\
        --lock-spans "0:10"

    # Interactive editing demo
    python scripts/sample.py --checkpoint checkpoints/model.pt --interactive
        """,
    )

    # Model
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval.yaml",
        help="Path to evaluation config file (default: configs/eval.yaml)",
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generate", "edit"],
        default="generate",
        help="Sampling mode: 'generate' for unconditional, 'edit' for constrained editing",
    )

    # Generation settings
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=128,
        help="Maximum sequence length for generation",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Number of denoising steps (default: use model's training timesteps)",
    )

    # Sampling strategy
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling (None = disabled)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling threshold (None = disabled)",
    )

    # Edit mode inputs
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input text for editing (edit mode only)",
    )
    parser.add_argument(
        "--lock-spans",
        type=str,
        default=None,
        help="Comma-separated lock spans as 'start:end' token indices (e.g., '0:5,10:15')",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Edit instruction prompt (e.g., 'make it more formal')",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: print to stdout)",
    )
    parser.add_argument(
        "--show-process",
        action="store_true",
        help="Show intermediate denoising steps",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run on (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: random)",
    )

    # Interactive
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )

    return parser.parse_args()


def main() -> None:
    """Main sampling entrypoint."""
    args = parse_args()

    print("=" * 60)
    print("ConstrainedDiffusionLM Sampling")
    print("=" * 60)
    print(f"\nMode: {args.mode}")
    print(f"Device: {args.device}")

    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    else:
        print("Checkpoint: [not specified - will need one for actual sampling]")

    if args.mode == "edit":
        print(f"\nInput: {args.input or '[not specified]'}")
        print(f"Lock spans: {args.lock_spans or '[none]'}")
        print(f"Prompt: {args.prompt or '[none]'}")

    print(f"\nNum samples: {args.num_samples}")
    print(f"Temperature: {args.temperature}")

    if args.interactive:
        print("\n[INTERACTIVE MODE]")

    print("\n✓ CLI is working! Sampling logic will be implemented in Phase 4.")
    print("=" * 60)


if __name__ == "__main__":
    main()

