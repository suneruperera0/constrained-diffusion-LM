"""
Unconditional text generation.
"""

from typing import List, Optional, Tuple
from pathlib import Path

import torch

from constrained_diffusion_lm.models import TransformerDenoiser
from constrained_diffusion_lm.diffusion.sampler import DiffusionSampler, ConfidenceBasedSampler
from constrained_diffusion_lm.diffusion.schedule import NoiseSchedule, get_schedule
from constrained_diffusion_lm.data.tokenization import Tokenizer


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device = None,
) -> Tuple[TransformerDenoiser, dict]:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    device = device or torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config from checkpoint or use defaults
    # Note: In a production setting, we'd save model config in checkpoint
    return checkpoint


def generate(
    model: TransformerDenoiser,
    tokenizer: Tokenizer,
    schedule: NoiseSchedule,
    num_samples: int = 1,
    seq_len: int = 64,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: torch.device = None,
    use_confidence_sampling: bool = False,
    show_progress: bool = True,
) -> List[str]:
    """
    Generate text samples.
    
    Args:
        model: Trained TransformerDenoiser
        tokenizer: Tokenizer for decoding
        schedule: Noise schedule
        num_samples: Number of samples to generate
        seq_len: Length of sequences to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        device: Device to run on
        use_confidence_sampling: Use confidence-based sampler
        show_progress: Show progress bar
        
    Returns:
        List of generated text strings
    """
    device = device or next(model.parameters()).device
    
    # Create sampler
    if use_confidence_sampling:
        sampler = ConfidenceBasedSampler(
            model=model,
            schedule=schedule,
            mask_token_id=tokenizer.mask_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        sampler = DiffusionSampler(
            model=model,
            schedule=schedule,
            mask_token_id=tokenizer.mask_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Generate
    token_ids = sampler.sample(
        batch_size=num_samples,
        seq_len=seq_len,
        device=device,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        show_progress=show_progress,
    )
    
    # Decode
    texts = tokenizer.decode_batch(token_ids)
    
    return texts


def generate_with_visualization(
    model: TransformerDenoiser,
    tokenizer: Tokenizer,
    schedule: NoiseSchedule,
    seq_len: int = 64,
    temperature: float = 1.0,
    device: torch.device = None,
    num_steps_to_show: int = 10,
) -> Tuple[str, List[Tuple[int, str]]]:
    """
    Generate text and return intermediate denoising states.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        schedule: Noise schedule
        seq_len: Sequence length
        temperature: Sampling temperature
        device: Device
        num_steps_to_show: Number of intermediate steps
        
    Returns:
        Tuple of (final text, list of (timestep, text) pairs)
    """
    device = device or next(model.parameters()).device
    
    sampler = DiffusionSampler(
        model=model,
        schedule=schedule,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    final_tokens, trajectory = sampler.sample_with_trajectory(
        batch_size=1,
        seq_len=seq_len,
        device=device,
        temperature=temperature,
        num_steps_to_save=num_steps_to_show,
    )
    
    final_text = tokenizer.decode(final_tokens[0])
    
    trajectory_texts = []
    for t, tokens in trajectory:
        text = tokenizer.decode(tokens[0], skip_special_tokens=False)
        trajectory_texts.append((t, text))
    
    return final_text, trajectory_texts
