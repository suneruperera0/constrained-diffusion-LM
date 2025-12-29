"""
Output head for predicting token logits.

The main transformer already includes the output projection.
This module provides utilities for working with the output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    penalty: float = 1.2,
) -> torch.Tensor:
    """
    Apply repetition penalty to discourage repeated tokens.
    
    Args:
        logits: Vocabulary logits [B, L, V]
        generated_tokens: Previously generated tokens [B, L] 
        penalty: Penalty factor (>1 = discourage repetition)
        
    Returns:
        Modified logits [B, L, V]
    """
    if penalty == 1.0:
        return logits
    
    batch_size, seq_len, vocab_size = logits.shape
    
    for b in range(batch_size):
        # Get unique tokens that have been generated
        unique_tokens = generated_tokens[b].unique()
        
        for token in unique_tokens:
            if token >= 0 and token < vocab_size:
                # Penalize this token across all positions
                logits[b, :, token] = logits[b, :, token] / penalty
    
    return logits


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    prev_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sample token IDs from logits.
    
    Args:
        logits: Vocabulary logits [B, L, V]
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top-k tokens (None = disabled)
        top_p: Nucleus sampling threshold (None = disabled)
        repetition_penalty: Penalty for repeating tokens (>1 = penalize)
        prev_tokens: Previously generated tokens for repetition penalty
        
    Returns:
        Sampled token IDs [B, L]
    """
    # Apply repetition penalty
    if repetition_penalty != 1.0 and prev_tokens is not None:
        logits = apply_repetition_penalty(logits, prev_tokens, repetition_penalty)
    
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


def get_confidence_scores(logits: torch.Tensor) -> torch.Tensor:
    """
    Get confidence score (max probability) for each position.
    
    Args:
        logits: Vocabulary logits [B, L, V]
        
    Returns:
        Confidence scores [B, L]
    """
    probs = F.softmax(logits, dim=-1)
    return probs.max(dim=-1).values


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
