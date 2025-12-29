"""
Reverse diffusion sampler for generation and editing.

The sampler iteratively denoises from x_T (fully masked) to x_0 (clean text).

This module supports:
- DiffusionSampler: Unconditional generation
- ConstrainedDiffusionSampler: Editing with locked tokens preserved
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


class ConstrainedDiffusionSampler(DiffusionSampler):
    """
    Constraint-preserving sampler for text editing.
    
    Key properties:
    1. Starts from original text (not all masks)
    2. Only editable positions are initially masked
    3. Locked tokens are CLAMPED at every denoising step
    4. Final output preserves locked tokens exactly
    
    This enables precise, constraint-aware text editing.
    """
    
    @torch.no_grad()
    def edit(
        self,
        x_0: torch.Tensor,
        lock_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        show_progress: bool = False,
        callback: Optional[Callable[[int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """
        Edit text while preserving locked tokens.
        
        Args:
            x_0: Original token IDs [B, L]
            lock_mask: Boolean mask [B, L] where True = LOCKED (preserve)
            attention_mask: Optional padding mask [B, L]
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            show_progress: Show progress bar
            callback: Called at each step with (t, x_t)
            
        Returns:
            Edited token IDs [B, L] with locked positions unchanged
        """
        device = x_0.device
        batch_size, seq_len = x_0.shape
        self.model.eval()
        
        # Store locked tokens
        locked_tokens = x_0.clone()
        
        # Initialize x_T: locked tokens stay, editable tokens become [MASK]
        x_t = x_0.clone()
        edit_mask = ~lock_mask
        x_t[edit_mask] = self.mask_token_id
        
        # Attention mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        
        # Iteratively denoise
        timesteps = list(range(self.num_timesteps, 0, -1))
        
        if show_progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Editing")
        
        for t_val in timesteps:
            t = torch.full((batch_size,), t_val, dtype=torch.long, device=device)
            
            # Get model prediction
            logits = self.model(x_t, t, attention_mask)
            
            # Sample from logits
            if temperature == 0:
                x_0_pred = argmax_from_logits(logits)
            else:
                x_0_pred = sample_from_logits(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            
            # Update x_t
            if t_val > 1:
                # Get target masking rate for t-1
                gamma_next = self.schedule(torch.tensor(t_val - 1, device=device)).item()
                
                # Randomly re-mask some editable positions
                random_mask = torch.rand(batch_size, seq_len, device=device) < gamma_next
                random_mask = random_mask & edit_mask  # Only re-mask editable positions
                
                # Unmask with predictions, but may re-mask some
                x_t = torch.where(random_mask, self.mask_token_id, x_0_pred)
            else:
                # Final step
                x_t = x_0_pred
            
            # CRITICAL: Clamp locked tokens back to original values
            x_t[lock_mask] = locked_tokens[lock_mask]
            
            # Callback for visualization
            if callback is not None:
                callback(t_val, x_t)
        
        # Final verification: locked tokens are preserved
        assert torch.all(x_t[lock_mask] == locked_tokens[lock_mask]), \
            "Locked tokens were modified!"
        
        return x_t
    
    @torch.no_grad()
    def edit_with_trajectory(
        self,
        x_0: torch.Tensor,
        lock_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        num_steps_to_save: int = 10,
    ) -> Tuple[torch.Tensor, List[Tuple[int, torch.Tensor]]]:
        """
        Edit text and return intermediate states.
        
        Args:
            x_0: Original tokens [B, L]
            lock_mask: Lock mask [B, L]
            attention_mask: Optional padding mask
            temperature: Sampling temperature
            num_steps_to_save: Number of intermediate steps to save
            
        Returns:
            Tuple of (final tokens, trajectory)
        """
        trajectory = []
        save_every = max(1, self.num_timesteps // num_steps_to_save)
        
        def save_callback(t: int, x_t: torch.Tensor):
            if t % save_every == 0 or t == self.num_timesteps or t == 1:
                trajectory.append((t, x_t.clone()))
        
        final = self.edit(
            x_0=x_0,
            lock_mask=lock_mask,
            attention_mask=attention_mask,
            temperature=temperature,
            callback=save_callback,
        )
        
        return final, trajectory
    
    def verify_constraints(
        self,
        original: torch.Tensor,
        edited: torch.Tensor,
        lock_mask: torch.Tensor,
    ) -> Tuple[bool, float]:
        """
        Verify that locked tokens are preserved.
        
        Args:
            original: Original tokens [B, L]
            edited: Edited tokens [B, L]
            lock_mask: Lock mask [B, L]
            
        Returns:
            Tuple of (all_preserved, preservation_rate)
        """
        locked_original = original[lock_mask]
        locked_edited = edited[lock_mask]
        
        matches = (locked_original == locked_edited).float()
        preservation_rate = matches.mean().item() if matches.numel() > 0 else 1.0
        all_preserved = preservation_rate == 1.0
        
        return all_preserved, preservation_rate


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


class ImprovedDiffusionSampler(DiffusionSampler):
    """
    Improved sampler with better generation quality.
    
    Improvements over base sampler:
    1. Confidence-based unmasking (unmask most confident first)
    2. Repetition penalty
    3. Annealed temperature (start high, decrease over steps)
    4. Low-confidence resampling
    5. Entropy-based remasking (remask uncertain tokens)
    """
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        seq_len: int = 64,
        device: torch.device = None,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.2,
        temperature_annealing: bool = True,
        confidence_threshold: float = 0.3,
        show_progress: bool = False,
        callback: Optional[Callable[[int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """
        Generate text with improved sampling strategies.
        
        Args:
            batch_size: Number of samples
            seq_len: Sequence length
            device: Device
            temperature: Base sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling
            repetition_penalty: Penalty for repeated tokens (>1.0)
            temperature_annealing: Whether to decrease temp over time
            confidence_threshold: Resample tokens below this confidence
            show_progress: Show progress
            callback: Callback at each step
            
        Returns:
            Generated token IDs [B, L]
        """
        from constrained_diffusion_lm.models.diffusion_head import (
            sample_from_logits, argmax_from_logits, get_confidence_scores
        )
        
        device = device or next(self.model.parameters()).device
        self.model.eval()
        
        # Initialize fully masked
        x_t = torch.full(
            (batch_size, seq_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        
        # Track which positions are still masked
        is_masked = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
        timesteps = list(range(self.num_timesteps, 0, -1))
        
        if show_progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Sampling")
        
        for step_idx, t_val in enumerate(timesteps):
            t = torch.full((batch_size,), t_val, dtype=torch.long, device=device)
            
            # Annealed temperature: start high, end low
            if temperature_annealing:
                progress = step_idx / len(timesteps)
                current_temp = temperature * (1.0 - 0.5 * progress)  # Decrease by 50%
            else:
                current_temp = temperature
            
            # Get model prediction
            logits = self.model(x_t, t, attention_mask)
            
            # Get confidence scores
            confidence = get_confidence_scores(logits)
            
            # Sample with repetition penalty
            x_0_pred = sample_from_logits(
                logits,
                temperature=current_temp,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                prev_tokens=x_t,
            )
            
            if t_val > 1:
                # Calculate how many tokens to unmask this step
                gamma_next = self.schedule(torch.tensor(t_val - 1, device=device)).item()
                gamma_curr = self.schedule(torch.tensor(t_val, device=device)).item()
                
                # For each sample, unmask positions with highest confidence
                for b in range(batch_size):
                    masked_positions = is_masked[b].nonzero(as_tuple=True)[0]
                    
                    if len(masked_positions) == 0:
                        continue
                    
                    # Get confidence at masked positions
                    masked_conf = confidence[b, masked_positions]
                    
                    # Calculate how many to unmask
                    current_masked_ratio = is_masked[b].float().mean().item()
                    target_masked_ratio = gamma_next
                    
                    if current_masked_ratio > target_masked_ratio:
                        num_to_unmask = int((current_masked_ratio - target_masked_ratio) * seq_len)
                        num_to_unmask = max(1, min(num_to_unmask, len(masked_positions)))
                        
                        # Unmask highest confidence positions
                        _, top_conf_indices = masked_conf.topk(num_to_unmask)
                        positions_to_unmask = masked_positions[top_conf_indices]
                        
                        # Update tokens and mask
                        x_t[b, positions_to_unmask] = x_0_pred[b, positions_to_unmask]
                        is_masked[b, positions_to_unmask] = False
                
                # Low-confidence resampling: resample if confidence is too low
                low_conf = (confidence < confidence_threshold) & (~is_masked)
                if low_conf.any():
                    # Re-mask low confidence tokens for resampling
                    x_t[low_conf] = self.mask_token_id
                    is_masked[low_conf] = True
            else:
                # Final step: fill in remaining masks
                x_t = torch.where(is_masked, x_0_pred, x_t)
            
            if callback is not None:
                callback(t_val, x_t)
        
        return x_t


class ImprovedConstrainedSampler(ConstrainedDiffusionSampler):
    """
    Improved constraint-preserving sampler.
    
    Combines constraint enforcement with improved sampling strategies.
    """
    
    @torch.no_grad()
    def edit(
        self,
        x_0: torch.Tensor,
        lock_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.2,
        show_progress: bool = False,
        callback: Optional[Callable[[int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """
        Edit with improved sampling.
        """
        from constrained_diffusion_lm.models.diffusion_head import (
            sample_from_logits, argmax_from_logits, get_confidence_scores
        )
        
        device = x_0.device
        batch_size, seq_len = x_0.shape
        self.model.eval()
        
        locked_tokens = x_0.clone()
        edit_mask = ~lock_mask
        
        # Initialize: locked stays, editable becomes [MASK]
        x_t = x_0.clone()
        x_t[edit_mask] = self.mask_token_id
        
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        
        is_masked = edit_mask.clone()  # Only editable positions start masked
        
        timesteps = list(range(self.num_timesteps, 0, -1))
        
        if show_progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Editing")
        
        for step_idx, t_val in enumerate(timesteps):
            t = torch.full((batch_size,), t_val, dtype=torch.long, device=device)
            
            # Get model prediction
            logits = self.model(x_t, t, attention_mask)
            confidence = get_confidence_scores(logits)
            
            # Sample with improvements
            x_0_pred = sample_from_logits(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                prev_tokens=x_t,
            )
            
            if t_val > 1:
                gamma_next = self.schedule(torch.tensor(t_val - 1, device=device)).item()
                
                for b in range(batch_size):
                    # Only consider editable masked positions
                    editable_masked = is_masked[b] & edit_mask[b]
                    masked_positions = editable_masked.nonzero(as_tuple=True)[0]
                    
                    if len(masked_positions) == 0:
                        continue
                    
                    masked_conf = confidence[b, masked_positions]
                    
                    # Calculate unmask ratio for editable region only
                    editable_count = edit_mask[b].sum().item()
                    current_masked = editable_masked.sum().item()
                    target_masked = int(gamma_next * editable_count)
                    
                    num_to_unmask = max(0, current_masked - target_masked)
                    num_to_unmask = min(num_to_unmask, len(masked_positions))
                    
                    if num_to_unmask > 0:
                        _, top_conf_indices = masked_conf.topk(num_to_unmask)
                        positions_to_unmask = masked_positions[top_conf_indices]
                        
                        x_t[b, positions_to_unmask] = x_0_pred[b, positions_to_unmask]
                        is_masked[b, positions_to_unmask] = False
            else:
                # Final step
                x_t = torch.where(is_masked & edit_mask, x_0_pred, x_t)
            
            # CRITICAL: Always clamp locked tokens
            x_t[lock_mask] = locked_tokens[lock_mask]
            
            if callback is not None:
                callback(t_val, x_t)
        
        return x_t
