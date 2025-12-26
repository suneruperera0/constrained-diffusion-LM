"""
Token corruption strategies for the forward diffusion process.

In mask diffusion, we corrupt tokens by replacing them with [MASK].
The corruption probability is determined by the noise schedule.

This module supports both:
- Unconstrained corruption (all tokens can be masked)
- Constrained corruption (locked tokens are NEVER masked)
"""

from typing import Optional, Tuple

import torch

from constrained_diffusion_lm.diffusion.schedule import NoiseSchedule


def corrupt_tokens(
    x_0: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    mask_token_id: int,
    attention_mask: Optional[torch.Tensor] = None,
    lock_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply mask diffusion corruption to tokens.
    
    Each token is independently replaced with [MASK] with probability gamma(t),
    UNLESS it is locked (lock_mask=True) in which case it is never masked.
    
    Args:
        x_0: Clean token IDs [B, L]
        t: Timesteps for each sample in batch [B]
        schedule: Noise schedule that gives masking probability
        mask_token_id: ID of the [MASK] token
        attention_mask: Optional [B, L] mask (1 for real tokens, 0 for padding)
                       If provided, padding tokens are never masked.
        lock_mask: Optional [B, L] boolean mask where True = LOCKED (never mask)
    
    Returns:
        x_t: Corrupted token IDs [B, L]
        noise_mask: Boolean mask indicating which tokens were masked [B, L]
    """
    batch_size, seq_len = x_0.shape
    device = x_0.device
    
    # Get masking probability for each sample's timestep
    gamma = schedule(t)  # [B]
    gamma = gamma.to(device)
    
    # Expand gamma to [B, L] for broadcasting
    gamma = gamma.unsqueeze(1).expand(-1, seq_len)  # [B, L]
    
    # Sample which tokens to mask (Bernoulli with probability gamma)
    noise_mask = torch.rand(batch_size, seq_len, device=device) < gamma
    
    # Don't mask padding tokens if attention_mask is provided
    if attention_mask is not None:
        noise_mask = noise_mask & (attention_mask.bool())
    
    # CRITICAL: Don't mask locked tokens
    if lock_mask is not None:
        noise_mask = noise_mask & (~lock_mask.bool())
    
    # Apply corruption: replace masked tokens with mask_token_id
    x_t = x_0.clone()
    x_t[noise_mask] = mask_token_id
    
    return x_t, noise_mask


def corrupt_tokens_constrained(
    x_0: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    mask_token_id: int,
    lock_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply constrained mask diffusion corruption.
    
    Locked tokens (lock_mask=True) are NEVER corrupted, regardless of timestep.
    This ensures x_t[lock_mask] == x_0[lock_mask] for all t.
    
    Args:
        x_0: Clean token IDs [B, L]
        t: Timesteps for each sample in batch [B]
        schedule: Noise schedule
        mask_token_id: ID of the [MASK] token
        lock_mask: Boolean mask [B, L] where True = LOCKED
        attention_mask: Optional [B, L] mask for padding
    
    Returns:
        x_t: Corrupted token IDs [B, L] (locked positions unchanged)
        noise_mask: Boolean mask of actually masked tokens [B, L]
    """
    return corrupt_tokens(
        x_0=x_0,
        t=t,
        schedule=schedule,
        mask_token_id=mask_token_id,
        attention_mask=attention_mask,
        lock_mask=lock_mask,
    )


def get_corruption_rate(
    t: torch.Tensor,
    schedule: NoiseSchedule,
) -> torch.Tensor:
    """
    Get the expected corruption rate at timestep t.
    
    Args:
        t: Timesteps [B] or scalar
        schedule: Noise schedule
        
    Returns:
        Expected fraction of tokens masked at timestep t
    """
    return schedule(t)


class MaskCorruptor:
    """
    Stateful wrapper for mask corruption.
    
    Convenient for use in training loops.
    """
    
    def __init__(
        self,
        schedule: NoiseSchedule,
        mask_token_id: int,
    ):
        """
        Initialize corruptor.
        
        Args:
            schedule: Noise schedule
            mask_token_id: ID of the [MASK] token
        """
        self.schedule = schedule
        self.mask_token_id = mask_token_id
    
    def __call__(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        lock_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply corruption.
        
        Args:
            x_0: Clean token IDs [B, L]
            t: Timesteps [B]
            attention_mask: Optional attention mask [B, L]
            lock_mask: Optional lock mask [B, L] where True = never mask
            
        Returns:
            x_t: Corrupted tokens [B, L]
            noise_mask: Which tokens were masked [B, L]
        """
        return corrupt_tokens(
            x_0=x_0,
            t=t,
            schedule=self.schedule,
            mask_token_id=self.mask_token_id,
            attention_mask=attention_mask,
            lock_mask=lock_mask,
        )
    
    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Sample random timesteps for training."""
        return self.schedule.sample_timesteps(batch_size, device)


class ConstrainedMaskCorruptor(MaskCorruptor):
    """
    Mask corruptor that respects token constraints.
    
    This is a convenience class that requires lock_mask to be provided,
    ensuring locked tokens are never corrupted.
    """
    
    def __call__(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        lock_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply constrained corruption.
        
        Args:
            x_0: Clean token IDs [B, L]
            t: Timesteps [B]
            lock_mask: Lock mask [B, L] where True = LOCKED (required)
            attention_mask: Optional attention mask [B, L]
            
        Returns:
            x_t: Corrupted tokens [B, L]
            noise_mask: Which tokens were masked [B, L]
        """
        return corrupt_tokens(
            x_0=x_0,
            t=t,
            schedule=self.schedule,
            mask_token_id=self.mask_token_id,
            attention_mask=attention_mask,
            lock_mask=lock_mask,
        )
    
    def verify_constraints(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        lock_mask: torch.Tensor,
    ) -> bool:
        """
        Verify that locked tokens are preserved.
        
        Args:
            x_0: Original tokens [B, L]
            x_t: Corrupted tokens [B, L]
            lock_mask: Lock mask [B, L]
            
        Returns:
            True if all locked tokens are preserved
        """
        locked_original = x_0[lock_mask]
        locked_corrupted = x_t[lock_mask]
        return torch.all(locked_original == locked_corrupted).item()
