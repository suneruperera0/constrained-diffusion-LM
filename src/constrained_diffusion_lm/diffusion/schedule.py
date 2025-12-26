"""
Noise schedules for the diffusion process.

The schedule defines the probability of masking a token at each timestep t.
At t=0, no tokens are masked (clean data).
At t=T, all tokens are masked (fully noised).
"""

import math
from typing import Literal

import torch


class NoiseSchedule:
    """
    Base class for noise schedules.
    
    A noise schedule defines gamma(t) ∈ [0, 1] for t ∈ [0, T],
    where gamma(t) is the probability that a token is masked at timestep t.
    
    - gamma(0) = 0 (no masking, clean data)
    - gamma(T) = 1 (full masking)
    """
    
    def __init__(self, num_timesteps: int):
        """
        Initialize schedule.
        
        Args:
            num_timesteps: Total number of timesteps T
        """
        self.num_timesteps = num_timesteps
        self._gamma = self._compute_schedule()
    
    def _compute_schedule(self) -> torch.Tensor:
        """Compute gamma values for all timesteps. Override in subclasses."""
        raise NotImplementedError
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get masking probability at timestep t.
        
        Args:
            t: Timestep indices [B] or scalar, values in [0, T]
            
        Returns:
            Masking probabilities gamma(t), same shape as input
        """
        return self._gamma[t]
    
    def sample_timesteps(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """
        Sample random timesteps for training.
        
        Args:
            batch_size: Number of timesteps to sample
            device: Device to place tensor on
            
        Returns:
            Random timesteps in [1, T] (we skip t=0 during training)
        """
        # Sample from [1, T] (inclusive) - we don't train on t=0 (clean data)
        t = torch.randint(1, self.num_timesteps + 1, (batch_size,), device=device)
        return t


class LinearSchedule(NoiseSchedule):
    """
    Linear noise schedule: gamma(t) = t / T
    
    Simple and stable. Masking probability increases linearly with timestep.
    """
    
    def _compute_schedule(self) -> torch.Tensor:
        # t goes from 0 to T
        # gamma(0) = 0, gamma(T) = 1
        t = torch.arange(self.num_timesteps + 1, dtype=torch.float32)
        gamma = t / self.num_timesteps
        return gamma


class CosineSchedule(NoiseSchedule):
    """
    Cosine noise schedule: gamma(t) = 1 - cos(π * t / (2T))
    
    Slower masking at the start, faster at the end.
    This tends to work better in practice as it preserves more signal
    in early timesteps.
    """
    
    def __init__(self, num_timesteps: int, s: float = 0.008):
        """
        Args:
            num_timesteps: Total number of timesteps
            s: Small offset to prevent gamma(0) from being exactly 0
        """
        self.s = s
        super().__init__(num_timesteps)
    
    def _compute_schedule(self) -> torch.Tensor:
        t = torch.arange(self.num_timesteps + 1, dtype=torch.float32)
        
        # f(t) = cos(π/2 * (t/T + s) / (1 + s))^2
        f_t = torch.cos(((t / self.num_timesteps) + self.s) / (1 + self.s) * math.pi / 2) ** 2
        f_0 = f_t[0]
        
        # alpha_bar(t) = f(t) / f(0)
        alpha_bar = f_t / f_0
        
        # gamma(t) = 1 - alpha_bar(t)
        gamma = 1 - alpha_bar
        
        # Clamp to [0, 1]
        gamma = torch.clamp(gamma, 0.0, 1.0)
        
        return gamma


class SqrtSchedule(NoiseSchedule):
    """
    Square root noise schedule: gamma(t) = sqrt(t / T)
    
    Faster masking at the start, slower at the end.
    """
    
    def _compute_schedule(self) -> torch.Tensor:
        t = torch.arange(self.num_timesteps + 1, dtype=torch.float32)
        gamma = torch.sqrt(t / self.num_timesteps)
        return gamma


def get_schedule(
    schedule_type: Literal["linear", "cosine", "sqrt"],
    num_timesteps: int,
    **kwargs,
) -> NoiseSchedule:
    """
    Factory function to create a noise schedule.
    
    Args:
        schedule_type: Type of schedule ("linear", "cosine", or "sqrt")
        num_timesteps: Total number of timesteps
        **kwargs: Additional arguments for specific schedules
        
    Returns:
        NoiseSchedule instance
    """
    schedules = {
        "linear": LinearSchedule,
        "cosine": CosineSchedule,
        "sqrt": SqrtSchedule,
    }
    
    if schedule_type not in schedules:
        raise ValueError(f"Unknown schedule type: {schedule_type}. Choose from {list(schedules.keys())}")
    
    return schedules[schedule_type](num_timesteps, **kwargs)
