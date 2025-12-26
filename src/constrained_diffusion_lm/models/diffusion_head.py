"""
Output head for predicting token logits.

The main transformer already includes the output projection.
This module provides utilities for working with the output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Sample token IDs from logits.
    
    Args:
        logits: Vocabulary logits [B, L, V]
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top-k tokens (None = disabled)
        top_p: Nucleus sampling threshold (None = disabled)
        
    Returns:
        Sampled token IDs [B, L]
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature
    
    # Top-k filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
    
    # Top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter back to original order
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
    
    # Sample from distribution
    probs = F.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
    
    return sampled.view(logits.shape[:-1])


def argmax_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Get most likely tokens from logits (greedy decoding).
    
    Args:
        logits: Vocabulary logits [B, L, V]
        
    Returns:
        Token IDs [B, L]
    """
    return logits.argmax(dim=-1)


def get_token_probabilities(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Get probability of specific tokens.
    
    Args:
        logits: Vocabulary logits [B, L, V]
        token_ids: Token IDs to get probabilities for [B, L]
        
    Returns:
        Probabilities [B, L]
    """
    probs = F.softmax(logits, dim=-1)
    token_probs = probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    return token_probs
