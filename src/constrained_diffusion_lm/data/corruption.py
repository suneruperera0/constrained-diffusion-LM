"""
Token corruption strategies for the forward diffusion process.

In mask diffusion, we corrupt tokens by replacing them with [MASK].
The corruption probability is determined by the noise schedule.
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply mask diffusion corruption to tokens.
    
    Each token is independently replaced with [MASK] with probability gamma(t).
    
    Args:
        x_0: Clean token IDs [B, L]
        t: Timesteps for each sample in batch [B]
        schedule: Noise schedule that gives masking probability
        mask_token_id: ID of the [MASK] token
        attention_mask: Optional [B, L] mask (1 for real tokens, 0 for padding)
                       If provided, padding tokens are never masked.
    
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
    
    # Apply corruption: replace masked tokens with mask_token_id
    x_t = x_0.clone()
    x_t[noise_mask] = mask_token_id
    
    return x_t, noise_mask


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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply corruption.
        
        Args:
            x_0: Clean token IDs [B, L]
            t: Timesteps [B]
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
        )
    
    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Sample random timesteps for training."""
        return self.schedule.sample_timesteps(batch_size, device)
