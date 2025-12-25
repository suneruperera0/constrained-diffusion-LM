#!/usr/bin/env python3
"""
Interactive demo for constraint-preserving text editing.

Usage:
    python scripts/edit_demo.py --checkpoint checkpoints/model.pt
    python scripts/edit_demo.py --help
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Demo: Constraint-Preserving Text Editing with Diffusion LM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run demo with a trained model
    python scripts/edit_demo.py --checkpoint checkpoints/model.pt

    # Quick demo with predefined examples
    python scripts/edit_demo.py --checkpoint checkpoints/model.pt --examples

    # Specify input directly
    python scripts/edit_demo.py --checkpoint checkpoints/model.pt \\
        --text "The contract is governed by Ontario law." \\
        --lock "Ontario law" \\
        --prompt "Make the tone more aggressive"
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Input text to edit",
    )
    parser.add_argument(
        "--lock",
        type=str,
        default=None,
        help="Substring to lock (will find and lock this exact text)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Edit instruction",
    )

    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run predefined example edits",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run on",
    )

    return parser.parse_args()


def main() -> None:
    """Main demo entrypoint."""
    args = parse_args()

    print("=" * 60)
    print("ConstrainedDiffusionLM - Edit Demo")
    print("=" * 60)

    if args.examples:
        print("\n[Running predefined examples]")
        examples = [
            {
                "text": "The contract is governed by Ontario law.",
                "lock": "Ontario law",
                "prompt": "Make the tone more aggressive",
            },
            {
                "text": "Please review the document at your earliest convenience.",
                "lock": "review the document",
                "prompt": "Make it more urgent",
            },
        ]
        for i, ex in enumerate(examples, 1):
            print(f"\n--- Example {i} ---")
            print(f"Original: {ex['text']}")
            print(f"Locked: '{ex['lock']}'")
            print(f"Prompt: {ex['prompt']}")
            print(f"Output: [will be generated in Phase 8]")
    else:
        print(f"\nText: {args.text or '[not specified]'}")
        print(f"Lock: {args.lock or '[none]'}")
        print(f"Prompt: {args.prompt or '[none]'}")

    print("\n✓ CLI is working! Edit logic will be implemented in Phase 8.")
    print("=" * 60)


if __name__ == "__main__":
    main()

