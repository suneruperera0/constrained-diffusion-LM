"""
Constraint-preserving text editing.

This module provides the high-level API for editing text while
preserving locked tokens exactly.
"""

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch

from constrained_diffusion_lm.models import TransformerDenoiser
from constrained_diffusion_lm.diffusion.sampler import ConstrainedDiffusionSampler
from constrained_diffusion_lm.diffusion.schedule import NoiseSchedule
from constrained_diffusion_lm.data.tokenization import Tokenizer
from constrained_diffusion_lm.data.constraints import (
    lock_substring,
    create_lock_mask,
    ConstraintSpec,
    create_masks_from_spec,
)


@dataclass
class EditResult:
    """Result of a text editing operation."""
    
    original_text: str
    edited_text: str
    locked_text: str
    lock_mask: torch.Tensor
    original_tokens: List[str]
    edited_tokens: List[str]
    constraint_preserved: bool
    preservation_rate: float


def edit_text(
    model: TransformerDenoiser,
    tokenizer: Tokenizer,
    schedule: NoiseSchedule,
    text: str,
    lock_substring_text: Optional[str] = None,
    lock_spans: Optional[List[Tuple[int, int]]] = None,
    lock_prefix: int = 0,
    lock_suffix: int = 0,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: torch.device = None,
    show_progress: bool = False,
) -> EditResult:
    """
    Edit text while preserving locked regions.
    
    Args:
        model: Trained TransformerDenoiser
        tokenizer: Tokenizer
        schedule: Noise schedule
        text: Original text to edit
        lock_substring_text: Substring to lock (finds and locks all occurrences)
        lock_spans: List of (start, end) token indices to lock
        lock_prefix: Number of tokens to lock at start
        lock_suffix: Number of tokens to lock at end
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        device: Device to run on
        show_progress: Show progress bar
        
    Returns:
        EditResult with original and edited text
    """
    device = device or next(model.parameters()).device
    
    # Tokenize
    token_ids = tokenizer.encode(text)
    seq_len = len(token_ids)
    
    # Create lock mask
    if lock_substring_text:
        lock_mask, _, spans = lock_substring(text, lock_substring_text, tokenizer)
        locked_text = lock_substring_text
    else:
        lock_mask = create_lock_mask(
            seq_len=seq_len,
            lock_spans=lock_spans,
            lock_prefix=lock_prefix,
            lock_suffix=lock_suffix,
        )
        # Get locked text from mask
        locked_tokens = [tokenizer.get_token(token_ids[i]) for i in range(seq_len) if lock_mask[i]]
        locked_text = " ".join(locked_tokens)
    
    # Prepare tensors
    x_0 = torch.tensor([token_ids], device=device)
    lock_mask_batch = lock_mask.unsqueeze(0).to(device)
    
    # Create sampler
    sampler = ConstrainedDiffusionSampler(
        model=model,
        schedule=schedule,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Edit
    edited_ids = sampler.edit(
        x_0=x_0,
        lock_mask=lock_mask_batch,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        show_progress=show_progress,
    )
    
    # Verify constraints
    preserved, rate = sampler.verify_constraints(x_0, edited_ids, lock_mask_batch)
    
    # Decode
    edited_text = tokenizer.decode(edited_ids[0])
    original_tokens = [tokenizer.get_token(tid) for tid in token_ids]
    edited_tokens = [tokenizer.get_token(tid) for tid in edited_ids[0].tolist()]
    
    return EditResult(
        original_text=text,
        edited_text=edited_text,
        locked_text=locked_text,
        lock_mask=lock_mask,
        original_tokens=original_tokens,
        edited_tokens=edited_tokens,
        constraint_preserved=preserved,
        preservation_rate=rate,
    )


def edit_text_with_trajectory(
    model: TransformerDenoiser,
    tokenizer: Tokenizer,
    schedule: NoiseSchedule,
    text: str,
    lock_substring_text: Optional[str] = None,
    lock_spans: Optional[List[Tuple[int, int]]] = None,
    temperature: float = 1.0,
    device: torch.device = None,
    num_steps_to_show: int = 10,
) -> Tuple[EditResult, List[Tuple[int, str]]]:
    """
    Edit text and return intermediate denoising states.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        schedule: Noise schedule
        text: Text to edit
        lock_substring_text: Substring to lock
        lock_spans: Token spans to lock
        temperature: Sampling temperature
        device: Device
        num_steps_to_show: Number of intermediate steps
        
    Returns:
        Tuple of (EditResult, trajectory)
    """
    device = device or next(model.parameters()).device
    
    # Tokenize
    token_ids = tokenizer.encode(text)
    seq_len = len(token_ids)
    
    # Create lock mask
    if lock_substring_text:
        lock_mask, _, _ = lock_substring(text, lock_substring_text, tokenizer)
        locked_text = lock_substring_text
    else:
        lock_mask = create_lock_mask(seq_len=seq_len, lock_spans=lock_spans)
        locked_tokens = [tokenizer.get_token(token_ids[i]) for i in range(seq_len) if lock_mask[i]]
        locked_text = " ".join(locked_tokens)
    
    # Prepare tensors
    x_0 = torch.tensor([token_ids], device=device)
    lock_mask_batch = lock_mask.unsqueeze(0).to(device)
    
    # Create sampler
    sampler = ConstrainedDiffusionSampler(
        model=model,
        schedule=schedule,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Edit with trajectory
    edited_ids, trajectory = sampler.edit_with_trajectory(
        x_0=x_0,
        lock_mask=lock_mask_batch,
        temperature=temperature,
        num_steps_to_save=num_steps_to_show,
    )
    
    # Convert trajectory to text
    trajectory_texts = []
    for t, tokens in trajectory:
        text_at_t = tokenizer.decode(tokens[0], skip_special_tokens=False)
        trajectory_texts.append((t, text_at_t))
    
    # Verify constraints
    preserved, rate = sampler.verify_constraints(x_0, edited_ids, lock_mask_batch)
    
    # Decode
    edited_text = tokenizer.decode(edited_ids[0])
    original_tokens = [tokenizer.get_token(tid) for tid in token_ids]
    edited_tokens = [tokenizer.get_token(tid) for tid in edited_ids[0].tolist()]
    
    result = EditResult(
        original_text=text,
        edited_text=edited_text,
        locked_text=locked_text,
        lock_mask=lock_mask,
        original_tokens=original_tokens,
        edited_tokens=edited_tokens,
        constraint_preserved=preserved,
        preservation_rate=rate,
    )
    
    return result, trajectory_texts


def batch_edit(
    model: TransformerDenoiser,
    tokenizer: Tokenizer,
    schedule: NoiseSchedule,
    texts: List[str],
    lock_substrings: List[str],
    temperature: float = 1.0,
    device: torch.device = None,
    show_progress: bool = True,
) -> List[EditResult]:
    """
    Edit multiple texts in batch.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        schedule: Noise schedule
        texts: List of texts to edit
        lock_substrings: List of substrings to lock (one per text)
        temperature: Sampling temperature
        device: Device
        show_progress: Show progress
        
    Returns:
        List of EditResults
    """
    results = []
    
    iterator = zip(texts, lock_substrings)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(list(iterator), desc="Editing")
    
    for text, lock_sub in iterator:
        result = edit_text(
            model=model,
            tokenizer=tokenizer,
            schedule=schedule,
            text=text,
            lock_substring_text=lock_sub,
            temperature=temperature,
            device=device,
            show_progress=False,
        )
        results.append(result)
    
    return results
