#!/usr/bin/env python3
"""
Training script for ConstrainedDiffusionLM.

Usage:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --debug  # Quick test with small model
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from constrained_diffusion_lm.data import Tokenizer, TextDataset, InMemoryTextDataset, create_dataloader
from constrained_diffusion_lm.data.datasets import HuggingFaceDataset
from constrained_diffusion_lm.data.corruption import MaskCorruptor
from constrained_diffusion_lm.diffusion import get_schedule
from constrained_diffusion_lm.models import TransformerDenoiser
from constrained_diffusion_lm.training.trainer import Trainer, TrainingConfig
from constrained_diffusion_lm.utils.seed import set_seed, get_device
from constrained_diffusion_lm.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Constraint-Preserving Diffusion Language Model",
    )

    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Quick test with small model")
    parser.add_argument("--resume", type=str, default=None)
    
    # New: HuggingFace dataset options
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace dataset name (e.g., 'wikitext', 'openwebtext')")
    parser.add_argument("--dataset-config", type=str, default=None,
                        help="Dataset config name (e.g., 'wikitext-2-raw-v1')")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of training samples")
    
    # New: Pretrained initialization
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Initialize from pretrained model (e.g., 'bert-base-uncased')")
    parser.add_argument("--freeze-embeddings", action="store_true",
                        help="Freeze token/position embeddings when using --pretrained")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze entire BERT backbone, only train timestep/output layers")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_sample_texts():
    """Sample texts for testing/demo."""
    return [
        "The contract is governed by Ontario law.",
        "Please review the document at your earliest convenience.",
        "This agreement shall be binding upon both parties.",
        "The defendant was found guilty of all charges.",
        "We hereby acknowledge receipt of your letter dated January 15.",
        "The terms and conditions are subject to change without notice.",
        "All disputes shall be resolved through arbitration.",
        "The company reserves the right to terminate this agreement.",
        "Payment is due within thirty days of invoice date.",
        "The parties agree to maintain confidentiality of all information.",
        "This contract supersedes all prior agreements between the parties.",
        "Neither party may assign this agreement without written consent.",
        "The warranty period shall extend for twelve months from delivery.",
        "Force majeure events shall suspend performance obligations.",
        "Governing law shall be the laws of the State of California.",
        "All notices must be delivered in writing to the registered address.",
    ] * 10  # Repeat for more training data


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load config
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        config = {}
        logger.warning(f"Config file not found: {args.config}, using defaults")
    
    # Debug mode: small model for quick testing
    if args.debug:
        logger.info("DEBUG MODE: Using small model and limited data")
        model_config = {
            "dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "max_seq_len": 64,
        }
        diffusion_config = {
            "timesteps": 100,
            "schedule": "cosine",
        }
        training_config = TrainingConfig(
            epochs=args.epochs or 5,
            learning_rate=args.lr or 1e-3,
            log_every=10,
            save_every=2,
        )
        batch_size = args.batch_size or 8
    else:
        model_config = config.get("model", {
            "dim": 512,
            "num_layers": 6,
            "num_heads": 8,
            "dim_feedforward": 2048,
            "dropout": 0.1,
            "max_seq_len": 256,
        })
        diffusion_config = config.get("diffusion", {
            "timesteps": 1000,
            "schedule": "cosine",
        })
        train_cfg = config.get("training", {})
        training_config = TrainingConfig(
            epochs=args.epochs or train_cfg.get("epochs", 100),
            learning_rate=args.lr or train_cfg.get("learning_rate", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 0.01),
            warmup_steps=train_cfg.get("warmup_steps", 1000),
            grad_clip=train_cfg.get("grad_clip", 1.0),
            log_every=config.get("logging", {}).get("log_every", 100),
            eval_every=config.get("logging", {}).get("eval_every", 1),
            save_every=config.get("checkpoint", {}).get("save_every", 5),
            output_dir=config.get("checkpoint", {}).get("output_dir", "checkpoints"),
        )
        batch_size = args.batch_size or train_cfg.get("batch_size", 32)
    
    # Initialize tokenizer
    tokenizer_name = config.get("data", {}).get("tokenizer", "bert-base-uncased")
    max_seq_len = model_config.get("max_seq_len", 256)
    tokenizer = Tokenizer(tokenizer_name, max_length=max_seq_len)
    logger.info(f"Tokenizer: {tokenizer_name}, vocab_size={tokenizer.vocab_size}")
    
    # Load or create dataset
    data_config = config.get("data", {})
    train_path = data_config.get("train_path")
    val_path = data_config.get("val_path")
    
    if args.dataset:
        # Use HuggingFace dataset
        logger.info(f"Loading HuggingFace dataset: {args.dataset}")
        train_dataset = HuggingFaceDataset(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            split="train",
            max_length=max_seq_len,
            max_samples=args.max_samples,
            config_name=args.dataset_config,
        )
        # Try to load validation split
        try:
            val_dataset = HuggingFaceDataset(
                dataset_name=args.dataset,
                tokenizer=tokenizer,
                split="validation",
                max_length=max_seq_len,
                max_samples=min(1000, args.max_samples or 1000),
                config_name=args.dataset_config,
            )
        except Exception:
            # Fall back to test split or subset of train
            try:
                val_dataset = HuggingFaceDataset(
                    dataset_name=args.dataset,
                    tokenizer=tokenizer,
                    split="test",
                    max_length=max_seq_len,
                    max_samples=1000,
                    config_name=args.dataset_config,
                )
            except Exception:
                logger.warning("No validation split found, using sample texts")
                val_dataset = InMemoryTextDataset(get_sample_texts()[:16], tokenizer)
    elif train_path and Path(train_path).exists():
        logger.info(f"Loading training data from {train_path}")
        train_dataset = TextDataset(train_path, tokenizer)
        if val_path and Path(val_path).exists():
            val_dataset = TextDataset(val_path, tokenizer)
        else:
            val_texts = get_sample_texts()[:16]
            val_dataset = InMemoryTextDataset(val_texts, tokenizer)
    else:
        logger.info("Using sample texts for training")
        train_dataset = InMemoryTextDataset(get_sample_texts(), tokenizer)
        val_texts = get_sample_texts()[:16]
        val_dataset = InMemoryTextDataset(val_texts, tokenizer)
    
    logger.info(f"Train dataset: {len(train_dataset)} examples")
    logger.info(f"Val dataset: {len(val_dataset)} examples")
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset, tokenizer, batch_size=batch_size, shuffle=True
    )
    val_dataloader = create_dataloader(
        val_dataset, tokenizer, batch_size=batch_size, shuffle=False
    )
    
    # Initialize noise schedule and corruptor
    schedule = get_schedule(
        diffusion_config.get("schedule", "cosine"),
        diffusion_config.get("timesteps", 1000),
    )
    corruptor = MaskCorruptor(schedule, tokenizer.mask_token_id)
    logger.info(f"Diffusion: {diffusion_config['timesteps']} timesteps, {diffusion_config['schedule']} schedule")
    
    # Initialize model
    if args.pretrained:
        # Initialize from pretrained BERT
        logger.info(f"Initializing from pretrained: {args.pretrained}")
        model = TransformerDenoiser.from_pretrained_bert(
            bert_model_name=args.pretrained,
            max_seq_len=max_seq_len,
            freeze_embeddings=args.freeze_embeddings,
        )
        # Freeze entire backbone if requested (recommended for small datasets)
        if args.freeze_backbone:
            model.freeze_backbone()
        # Update model_config to match pretrained
        model_config = {
            "dim": model.dim,
            "num_layers": len(model.transformer.layers),
            "num_heads": model.transformer.layers[0].self_attn.num_heads,
            "dim_feedforward": model.transformer.layers[0].linear1.out_features,
            "max_seq_len": max_seq_len,
        }
    else:
        model = TransformerDenoiser(
            vocab_size=tokenizer.vocab_size,
            dim=model_config.get("dim", 512),
            num_layers=model_config.get("num_layers", 6),
            num_heads=model_config.get("num_heads", 8),
            dim_feedforward=model_config.get("dim_feedforward", 2048),
            dropout=model_config.get("dropout", 0.1),
            max_seq_len=model_config.get("max_seq_len", 256),
            pad_token_id=tokenizer.pad_token_id,
        )
    logger.info(f"Model: {model.get_num_trainable_params():,} trainable parameters")
    
    # Initialize trainer with masked loss (aligns training with inference)
    trainer = Trainer(
        model=model,
        corruptor=corruptor,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=training_config,
        device=device,
        model_config=model_config,  # Save architecture info in checkpoints
        use_masked_loss=True,  # Only compute loss on masked tokens
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train!
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    trainer.train()


if __name__ == "__main__":
    main()
