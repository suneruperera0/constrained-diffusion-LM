"""
Reverse diffusion sampler for generation and editing.

The sampler iteratively denoises from x_T (fully masked) to x_0 (clean text).
"""

from typing import Optional, Callable, List, Tuple

import torch
import torch.nn.functional as F

from constrained_diffusion_lm.diffusion.schedule import NoiseSchedule
from constrained_diffusion_lm.models.diffusion_head import sample_from_logits, argmax_from_logits


class DiffusionSampler:
    """
    Iterative denoising sampler for diffusion LM.
    
    Starting from x_T (fully masked), iteratively predicts x_0 and
    re-masks according to the schedule until reaching clean tokens.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        schedule: NoiseSchedule,
        mask_token_id: int,
        pad_token_id: int = 0,
    ):
        """
        Initialize sampler.
        
        Args:
            model: Trained TransformerDenoiser
            schedule: Noise schedule used during training
            mask_token_id: ID of [MASK] token
            pad_token_id: ID of [PAD] token
        """
        self.model = model
        self.schedule = schedule
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.num_timesteps = schedule.num_timesteps
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        seq_len: int = 64,
        device: torch.device = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        show_progress: bool = False,
        callback: Optional[Callable[[int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """
        Generate text by iterative denoising.
        
        Args:
            batch_size: Number of samples to generate
            seq_len: Sequence length to generate
            device: Device to run on
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            show_progress: Print progress
            callback: Optional function called at each step with (t, x_t)
            
        Returns:
            Generated token IDs [B, L]
        """
        device = device or next(self.model.parameters()).device
        self.model.eval()
        
        # Initialize x_T as fully masked
        x_t = torch.full(
            (batch_size, seq_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        
        # Attention mask (all ones since no padding in generation)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        
        # Iteratively denoise from t=T to t=1
        timesteps = list(range(self.num_timesteps, 0, -1))
        
        if show_progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Sampling")
        
        for t_val in timesteps:
            t = torch.full((batch_size,), t_val, dtype=torch.long, device=device)
            
            # Get model prediction
            logits = self.model(x_t, t, attention_mask)
            
            # Sample or argmax from logits
            if temperature == 0:
                x_0_pred = argmax_from_logits(logits)
            else:
                x_0_pred = sample_from_logits(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            
            # Determine which tokens to keep vs re-mask for next step
            if t_val > 1:
                # Get target masking rate for t-1
                gamma_next = self.schedule(torch.tensor(t_val - 1, device=device)).item()
                
                # Randomly select positions to keep masked
                keep_mask_prob = gamma_next
                random_mask = torch.rand(batch_size, seq_len, device=device) < keep_mask_prob
                
                # Update x_t: unmask some positions with predictions
                x_t = torch.where(random_mask, self.mask_token_id, x_0_pred)
            else:
                # Final step: use all predictions
                x_t = x_0_pred
            
            # Callback for visualization
            if callback is not None:
                callback(t_val, x_t)
        
        return x_t
    
    @torch.no_grad()
    def sample_with_trajectory(
        self,
        batch_size: int = 1,
        seq_len: int = 64,
        device: torch.device = None,
        temperature: float = 1.0,
        num_steps_to_save: int = 10,
    ) -> Tuple[torch.Tensor, List[Tuple[int, torch.Tensor]]]:
        """
        Generate text and return intermediate states.
        
        Args:
            batch_size: Number of samples
            seq_len: Sequence length
            device: Device
            temperature: Sampling temperature
            num_steps_to_save: Number of intermediate steps to save
            
        Returns:
            Tuple of (final tokens, list of (timestep, tokens) pairs)
        """
        trajectory = []
        save_every = max(1, self.num_timesteps // num_steps_to_save)
        
        def save_callback(t: int, x_t: torch.Tensor):
            if t % save_every == 0 or t == self.num_timesteps or t == 1:
                trajectory.append((t, x_t.clone()))
        
        final = self.sample(
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            temperature=temperature,
            callback=save_callback,
        )
        
        return final, trajectory


class ConfidenceBasedSampler(DiffusionSampler):
    """
    Alternative sampler that unmasks tokens based on prediction confidence.
    
    At each step, unmask the positions where the model is most confident.
    """
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        seq_len: int = 64,
        device: torch.device = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        show_progress: bool = False,
        callback: Optional[Callable[[int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """
        Generate using confidence-based unmasking.
        
        At each step:
        1. Predict x_0 for all masked positions
        2. Compute confidence (max probability) for each position
        3. Unmask the top-k most confident positions
        """
        device = device or next(self.model.parameters()).device
        self.model.eval()
        
        # Initialize x_T as fully masked
        x_t = torch.full(
            (batch_size, seq_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        
        # Track which positions are still masked
        is_masked = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
        # Number of tokens to unmask per step
        total_to_unmask = seq_len
        num_steps = self.num_timesteps
        
        timesteps = list(range(self.num_timesteps, 0, -1))
        if show_progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Sampling")
        
        for step_idx, t_val in enumerate(timesteps):
            t = torch.full((batch_size,), t_val, dtype=torch.long, device=device)
            
            # Get model prediction
            logits = self.model(x_t, t, attention_mask)
            probs = F.softmax(logits, dim=-1)
            
            # Get max probability (confidence) for each position
            confidence, predicted_tokens = probs.max(dim=-1)
            
            # Only consider masked positions
            confidence = confidence.masked_fill(~is_masked, -float('inf'))
            
            # Determine how many to unmask at this step
            # Linear schedule: unmask proportionally
            target_unmasked = int((1 - t_val / self.num_timesteps) * total_to_unmask)
            current_unmasked = (~is_masked).sum(dim=1)
            num_to_unmask = (target_unmasked - current_unmasked).clamp(min=0)
            
            # For each sample, unmask top-k most confident
            for b in range(batch_size):
                if num_to_unmask[b] > 0:
                    _, top_indices = confidence[b].topk(min(num_to_unmask[b].item(), is_masked[b].sum().item()))
                    
                    # Apply temperature when sampling the token
                    if temperature != 0 and temperature != 1.0:
                        sampled = sample_from_logits(
                            logits[b:b+1, top_indices],
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                        )
                        x_t[b, top_indices] = sampled.squeeze(0)
                    else:
                        x_t[b, top_indices] = predicted_tokens[b, top_indices]
                    
                    is_masked[b, top_indices] = False
            
            if callback is not None:
                callback(t_val, x_t)
        
        # Final step: unmask any remaining
        if is_masked.any():
            t = torch.ones(batch_size, dtype=torch.long, device=device)
            logits = self.model(x_t, t, attention_mask)
            final_predictions = argmax_from_logits(logits)
            x_t = torch.where(is_masked, final_predictions, x_t)
        
        return x_t
