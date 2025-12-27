"""
Training loop and utilities.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from constrained_diffusion_lm.models import TransformerDenoiser
from constrained_diffusion_lm.training.losses import DiffusionLoss, compute_accuracy
from constrained_diffusion_lm.data.corruption import MaskCorruptor
from constrained_diffusion_lm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Training
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    
    # Logging
    log_every: int = 100
    eval_every: int = 1  # epochs
    
    # Checkpointing
    output_dir: str = "checkpoints"
    save_every: int = 5  # epochs
    keep_last: int = 3


class Trainer:
    """
    Trainer for diffusion language model.
    """
    
    def __init__(
        self,
        model: TransformerDenoiser,
        corruptor: MaskCorruptor,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        device: torch.device = None,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: TransformerDenoiser model
            corruptor: MaskCorruptor for forward diffusion
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            config: Training configuration
            device: Device to train on
            model_config: Model architecture config (saved in checkpoints)
        """
        self.model = model
        self.corruptor = corruptor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or TrainingConfig()
        self.device = device or torch.device("cpu")
        self.model_config = model_config or {}
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Loss function
        self.loss_fn = DiffusionLoss()
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Learning rate scheduler
        total_steps = len(train_dataloader) * self.config.epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.1,
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        
        # Output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch from dataloader
            
        Returns:
            Dict of metrics
        """
        self.model.train()
        
        # Move to device
        x_0 = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Sample timesteps
        t = self.corruptor.sample_timesteps(x_0.size(0), self.device)
        
        # Corrupt tokens
        x_t, noise_mask = self.corruptor(x_0, t, attention_mask)
        
        # Forward pass
        logits = self.model(x_t, t, attention_mask)
        
        # Compute loss
        loss = self.loss_fn(logits, x_0, attention_mask)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.grad_clip
            )
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Compute accuracy
        with torch.no_grad():
            acc = compute_accuracy(logits, x_0, attention_mask)
        
        return {
            "loss": loss.item(),
            "accuracy": acc.item(),
            "lr": self.scheduler.get_last_lr()[0],
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set.
        
        Returns:
            Dict of metrics
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        for batch in self.val_dataloader:
            x_0 = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Sample timesteps
            t = self.corruptor.sample_timesteps(x_0.size(0), self.device)
            
            # Corrupt tokens
            x_t, noise_mask = self.corruptor(x_0, t, attention_mask)
            
            # Forward pass
            logits = self.model(x_t, t, attention_mask)
            
            # Compute loss
            loss = self.loss_fn(logits, x_0, attention_mask)
            acc = compute_accuracy(logits, x_0, attention_mask)
            
            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1
        
        return {
            "val_loss": total_loss / num_batches,
            "val_accuracy": total_acc / num_batches,
        }
    
    def save_checkpoint(self, filename: str = "checkpoint.pt"):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "model_config": self.model_config,  # Save model architecture info
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")
    
    def train(self):
        """
        Run full training loop.
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Model has {self.model.get_num_trainable_params():,} trainable parameters")
        
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            # Training epoch
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.config.epochs}",
                leave=True,
            )
            
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            
            for batch in pbar:
                metrics = self.train_step(batch)
                self.global_step += 1
                
                epoch_loss += metrics["loss"]
                epoch_acc += metrics["accuracy"]
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "acc": f"{metrics['accuracy']:.3f}",
                    "lr": f"{metrics['lr']:.2e}",
                })
                
                # Logging
                if self.global_step % self.config.log_every == 0:
                    avg_loss = epoch_loss / num_batches
                    avg_acc = epoch_acc / num_batches
                    logger.info(
                        f"Step {self.global_step}: loss={avg_loss:.4f}, acc={avg_acc:.3f}"
                    )
            
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / num_batches
            avg_epoch_acc = epoch_acc / num_batches
            
            logger.info(
                f"Epoch {epoch+1} complete: loss={avg_epoch_loss:.4f}, "
                f"acc={avg_epoch_acc:.3f}, time={epoch_time:.1f}s"
            )
            
            # Evaluation
            if (epoch + 1) % self.config.eval_every == 0 and self.val_dataloader:
                val_metrics = self.evaluate()
                logger.info(
                    f"Validation: loss={val_metrics['val_loss']:.4f}, "
                    f"acc={val_metrics['val_accuracy']:.3f}"
                )
                
                # Save best model
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best_model.pt")
            
            # Periodic checkpointing
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        logger.info("Training complete!")
