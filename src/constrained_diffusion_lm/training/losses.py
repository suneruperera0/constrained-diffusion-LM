"""
Loss functions for diffusion training.

The model predicts x_0 from x_t, and we compute cross-entropy loss.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """
    Cross-entropy loss for diffusion LM.
    
    Computes CE between predicted x_0 logits and true x_0 tokens.
    """
    
    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize loss.
        
        Args:
            ignore_index: Token ID to ignore in loss (e.g., padding)
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        
        Args:
            logits: Predicted logits [B, L, V]
            targets: True token IDs [B, L]
            attention_mask: Optional mask [B, L], 1 for real tokens
            reduction: "mean", "sum", or "none"
            
        Returns:
            Loss value (scalar if reduction != "none")
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for cross-entropy
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        # If attention mask provided, set padding positions to ignore_index
        if attention_mask is not None:
            mask_flat = attention_mask.view(-1)
            targets_flat = targets_flat.clone()
            targets_flat[mask_flat == 0] = self.ignore_index
        
        # Compute cross-entropy
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=reduction if reduction != "none" else "none",
        )
        
        if reduction == "none":
            loss = loss.view(batch_size, seq_len)
        
        return loss


class MaskedDiffusionLoss(nn.Module):
    """
    Cross-entropy loss computed ONLY on masked (corrupted) tokens.
    
    This focuses the model on predicting the tokens that were actually corrupted,
    which can be more efficient than predicting all tokens.
    """
    
    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        noise_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute loss only on masked positions.
        
        Args:
            logits: Predicted logits [B, L, V]
            targets: True token IDs [B, L]
            noise_mask: Boolean mask of corrupted positions [B, L]
            attention_mask: Optional mask [B, L] for padding
            reduction: "mean", "sum", or "none"
            
        Returns:
            Loss value
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Combine masks: only compute loss where noise_mask is True AND attention_mask is True
        compute_loss_mask = noise_mask.bool()
        if attention_mask is not None:
            compute_loss_mask = compute_loss_mask & attention_mask.bool()
        
        # Set non-masked positions to ignore_index
        targets_masked = targets.clone()
        targets_masked[~compute_loss_mask] = self.ignore_index
        
        # Reshape and compute loss
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets_masked.view(-1)
        
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=reduction if reduction != "none" else "none",
        )
        
        if reduction == "none":
            loss = loss.view(batch_size, seq_len)
        
        return loss


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute token prediction accuracy.
    
    Args:
        logits: Predicted logits [B, L, V]
        targets: True token IDs [B, L]
        mask: Optional mask [B, L] (1 for positions to include)
        
    Returns:
        Accuracy as a scalar tensor
    """
    predictions = logits.argmax(dim=-1)
    correct = (predictions == targets)
    
    if mask is not None:
        correct = correct & mask.bool()
        return correct.sum().float() / mask.sum().float()
    else:
        return correct.float().mean()
