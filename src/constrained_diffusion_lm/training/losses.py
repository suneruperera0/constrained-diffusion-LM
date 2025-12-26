"""
Loss functions for diffusion training.

The model predicts x_0 from x_t, and we compute cross-entropy loss.

This module supports:
- DiffusionLoss: Loss on all tokens
- MaskedDiffusionLoss: Loss only on masked tokens
- ConstrainedDiffusionLoss: Loss only on EDITABLE tokens (locked tokens contribute 0 loss)
"""

from typing import Optional, Tuple

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


class ConstrainedDiffusionLoss(nn.Module):
    """
    Cross-entropy loss computed ONLY on EDITABLE tokens.
    
    Locked tokens contribute ZERO to the loss, meaning:
    - No gradients flow from locked positions
    - The model focuses on predicting editable regions
    - Training is more efficient when many tokens are locked
    
    This is the key loss function for constraint-preserving training.
    """
    
    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize loss.
        
        Args:
            ignore_index: Token ID to ignore in loss
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        edit_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute loss only on editable positions.
        
        The loss formula is:
            L = CE(logits, targets) ⊙ edit_mask
            L = sum(L) / sum(edit_mask)  # normalized by editable token count
        
        Args:
            logits: Predicted logits [B, L, V]
            targets: True token IDs [B, L]
            edit_mask: Boolean mask [B, L] where True = EDITABLE (compute loss)
            attention_mask: Optional mask [B, L] for padding
            reduction: "mean", "sum", or "none"
            
        Returns:
            Loss value (only from editable tokens)
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Combine masks: compute loss where edit_mask AND attention_mask are True
        compute_loss_mask = edit_mask.bool()
        if attention_mask is not None:
            compute_loss_mask = compute_loss_mask & attention_mask.bool()
        
        # Set locked/padding positions to ignore_index
        targets_editable = targets.clone()
        targets_editable[~compute_loss_mask] = self.ignore_index
        
        # Reshape and compute loss
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets_editable.view(-1)
        
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
    
    def forward_with_stats(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        edit_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute loss with additional statistics.
        
        Args:
            logits: Predicted logits [B, L, V]
            targets: True token IDs [B, L]
            edit_mask: Boolean mask [B, L] where True = EDITABLE
            attention_mask: Optional mask [B, L]
            
        Returns:
            Tuple of (loss, stats_dict)
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Get masks
        compute_loss_mask = edit_mask.bool()
        if attention_mask is not None:
            compute_loss_mask = compute_loss_mask & attention_mask.bool()
        
        lock_mask = ~edit_mask.bool()
        if attention_mask is not None:
            lock_mask = lock_mask & attention_mask.bool()
        
        # Compute loss per token
        loss_per_token = self.forward(
            logits, targets, edit_mask, attention_mask, reduction="none"
        )
        
        # Mean loss over editable tokens
        loss = loss_per_token.sum() / compute_loss_mask.sum().float().clamp(min=1)
        
        # Stats
        predictions = logits.argmax(dim=-1)
        
        # Accuracy on editable tokens
        correct_editable = (predictions == targets) & compute_loss_mask
        acc_editable = correct_editable.sum().float() / compute_loss_mask.sum().float().clamp(min=1)
        
        # Accuracy on locked tokens (should be ~1.0 if model learns to copy)
        correct_locked = (predictions == targets) & lock_mask
        acc_locked = correct_locked.sum().float() / lock_mask.sum().float().clamp(min=1) if lock_mask.any() else torch.tensor(1.0)
        
        stats = {
            "loss": loss.item(),
            "acc_editable": acc_editable.item(),
            "acc_locked": acc_locked.item(),
            "n_editable": compute_loss_mask.sum().item(),
            "n_locked": lock_mask.sum().item(),
        }
        
        return loss, stats


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
        return correct.sum().float() / mask.sum().float().clamp(min=1)
    else:
        return correct.float().mean()


def compute_constrained_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    edit_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute accuracy separately for editable and locked tokens.
    
    Args:
        logits: Predicted logits [B, L, V]
        targets: True token IDs [B, L]
        edit_mask: Boolean mask [B, L] where True = editable
        attention_mask: Optional padding mask
        
    Returns:
        Tuple of (editable_accuracy, locked_accuracy)
    """
    predictions = logits.argmax(dim=-1)
    correct = (predictions == targets)
    
    # Editable tokens
    editable = edit_mask.bool()
    if attention_mask is not None:
        editable = editable & attention_mask.bool()
    
    # Locked tokens
    locked = ~edit_mask.bool()
    if attention_mask is not None:
        locked = locked & attention_mask.bool()
    
    editable_correct = (correct & editable).sum().float()
    editable_total = editable.sum().float().clamp(min=1)
    editable_acc = editable_correct / editable_total
    
    locked_correct = (correct & locked).sum().float()
    locked_total = locked.sum().float().clamp(min=1)
    locked_acc = locked_correct / locked_total if locked.any() else torch.tensor(1.0)
    
    return editable_acc, locked_acc
