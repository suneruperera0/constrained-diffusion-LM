"""
Constraint handling for locked and editable token regions.

This module provides utilities for creating lock masks that specify
which tokens should be preserved during diffusion and editing.

Key concepts:
- lock_mask: Boolean tensor where True = token is LOCKED (preserved)
- edit_mask: Boolean tensor where True = token is EDITABLE (can change)
- lock_mask = ~edit_mask (they are complements)
"""

from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

import torch

from constrained_diffusion_lm.data.tokenization import Tokenizer


@dataclass
class ConstraintSpec:
    """
    Specification for token constraints.
    
    Attributes:
        lock_spans: List of (start, end) token indices to lock (inclusive start, exclusive end)
        lock_prefix: Number of tokens to lock at the start
        lock_suffix: Number of tokens to lock at the end
        edit_spans: List of (start, end) token indices that are editable
                   (alternative to lock_spans - if provided, everything else is locked)
    """
    lock_spans: Optional[List[Tuple[int, int]]] = None
    lock_prefix: int = 0
    lock_suffix: int = 0
    edit_spans: Optional[List[Tuple[int, int]]] = None


def create_lock_mask(
    seq_len: int,
    lock_spans: Optional[List[Tuple[int, int]]] = None,
    lock_prefix: int = 0,
    lock_suffix: int = 0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a lock mask from span specifications.
    
    Args:
        seq_len: Total sequence length
        lock_spans: List of (start, end) token indices to lock
        lock_prefix: Number of tokens to lock at start
        lock_suffix: Number of tokens to lock at end
        device: Device for the tensor
        
    Returns:
        Boolean tensor [L] where True = locked
    """
    lock_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    
    # Lock prefix
    if lock_prefix > 0:
        lock_mask[:lock_prefix] = True
    
    # Lock suffix
    if lock_suffix > 0:
        lock_mask[-lock_suffix:] = True
    
    # Lock specific spans
    if lock_spans:
        for start, end in lock_spans:
            # Clamp to valid range
            start = max(0, start)
            end = min(seq_len, end)
            if start < end:
                lock_mask[start:end] = True
    
    return lock_mask


def create_edit_mask(
    seq_len: int,
    edit_spans: Optional[List[Tuple[int, int]]] = None,
    lock_spans: Optional[List[Tuple[int, int]]] = None,
    lock_prefix: int = 0,
    lock_suffix: int = 0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create an edit mask (complement of lock mask).
    
    If edit_spans is provided, only those spans are editable.
    Otherwise, uses lock_spans/prefix/suffix and everything else is editable.
    
    Args:
        seq_len: Total sequence length
        edit_spans: List of (start, end) spans that are editable
        lock_spans: List of (start, end) spans that are locked
        lock_prefix: Tokens to lock at start
        lock_suffix: Tokens to lock at end
        device: Device for the tensor
        
    Returns:
        Boolean tensor [L] where True = editable
    """
    if edit_spans is not None:
        # Only edit_spans are editable, everything else is locked
        edit_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        for start, end in edit_spans:
            start = max(0, start)
            end = min(seq_len, end)
            if start < end:
                edit_mask[start:end] = True
        return edit_mask
    else:
        # Use lock mask and invert
        lock_mask = create_lock_mask(
            seq_len=seq_len,
            lock_spans=lock_spans,
            lock_prefix=lock_prefix,
            lock_suffix=lock_suffix,
            device=device,
        )
        return ~lock_mask


def create_masks_from_spec(
    seq_len: int,
    spec: ConstraintSpec,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create both lock and edit masks from a constraint specification.
    
    Args:
        seq_len: Sequence length
        spec: ConstraintSpec object
        device: Device for tensors
        
    Returns:
        Tuple of (lock_mask, edit_mask), both [L] boolean tensors
    """
    edit_mask = create_edit_mask(
        seq_len=seq_len,
        edit_spans=spec.edit_spans,
        lock_spans=spec.lock_spans,
        lock_prefix=spec.lock_prefix,
        lock_suffix=spec.lock_suffix,
        device=device,
    )
    lock_mask = ~edit_mask
    return lock_mask, edit_mask


def find_substring_spans(
    text: str,
    substring: str,
    tokenizer: Tokenizer,
) -> List[Tuple[int, int]]:
    """
    Find token spans corresponding to a substring in the text.
    
    Args:
        text: Full text
        substring: Substring to find
        tokenizer: Tokenizer for encoding
        
    Returns:
        List of (start, end) token indices where substring appears
    """
    # This is a simplified approach - find char positions then map to tokens
    spans = []
    
    # Encode full text
    full_encoding = tokenizer.tokenizer(
        text,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    token_ids = full_encoding["input_ids"]
    offsets = full_encoding["offset_mapping"]
    
    # Find substring positions in original text
    start_char = 0
    while True:
        pos = text.lower().find(substring.lower(), start_char)
        if pos == -1:
            break
        
        end_char = pos + len(substring)
        
        # Map character positions to token positions
        token_start = None
        token_end = None
        
        for i, (char_start, char_end) in enumerate(offsets):
            if char_start is None:  # Special token
                continue
            if char_start <= pos < char_end and token_start is None:
                token_start = i
            if char_start < end_char <= char_end:
                token_end = i + 1
                break
            if char_start >= end_char and token_end is None:
                token_end = i
                break
        
        if token_start is not None and token_end is not None:
            spans.append((token_start, token_end))
        
        start_char = pos + 1
    
    return spans


def lock_substring(
    text: str,
    substring: str,
    tokenizer: Tokenizer,
    seq_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Create lock mask by locking all occurrences of a substring.
    
    Args:
        text: Full text
        substring: Substring to lock
        tokenizer: Tokenizer
        seq_len: Sequence length (if None, uses tokenized length)
        
    Returns:
        Tuple of (lock_mask, edit_mask, spans)
    """
    spans = find_substring_spans(text, substring, tokenizer)
    
    if seq_len is None:
        token_ids = tokenizer.encode(text)
        seq_len = len(token_ids)
    
    lock_mask = create_lock_mask(seq_len=seq_len, lock_spans=spans)
    edit_mask = ~lock_mask
    
    return lock_mask, edit_mask, spans


def visualize_constraints(
    text: str,
    lock_mask: torch.Tensor,
    tokenizer: Tokenizer,
    lock_char: str = "█",
    edit_char: str = "░",
) -> str:
    """
    Create a visual representation of locked vs editable tokens.
    
    Args:
        text: Original text
        lock_mask: Boolean mask [L] where True = locked
        tokenizer: Tokenizer
        lock_char: Character to show for locked tokens
        edit_char: Character to show for editable tokens
        
    Returns:
        Multi-line string visualization
    """
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.get_token(tid) for tid in token_ids]
    
    lines = []
    
    # Token line
    token_strs = []
    for i, tok in enumerate(tokens):
        if i < len(lock_mask):
            if lock_mask[i]:
                token_strs.append(f"\033[92m{tok}\033[0m")  # Green for locked
            else:
                token_strs.append(f"\033[93m{tok}\033[0m")  # Yellow for editable
        else:
            token_strs.append(tok)
    
    lines.append("Tokens: " + " ".join(token_strs))
    
    # Mask visualization
    mask_str = ""
    for i, tok in enumerate(tokens):
        if i < len(lock_mask):
            char = lock_char if lock_mask[i] else edit_char
            mask_str += char * len(tok) + " "
        else:
            mask_str += "?" * len(tok) + " "
    
    lines.append("Mask:   " + mask_str)
    
    # Legend
    lines.append(f"\nLegend: \033[92mGreen/█\033[0m = Locked, \033[93mYellow/░\033[0m = Editable")
    
    return "\n".join(lines)


def batch_create_masks(
    batch_size: int,
    seq_len: int,
    specs: List[ConstraintSpec],
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create batched lock and edit masks.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        specs: List of ConstraintSpec for each item in batch
        device: Device for tensors
        
    Returns:
        Tuple of (lock_masks [B, L], edit_masks [B, L])
    """
    lock_masks = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    edit_masks = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    for i, spec in enumerate(specs):
        lock_mask, edit_mask = create_masks_from_spec(seq_len, spec, device)
        lock_masks[i] = lock_mask
        edit_masks[i] = edit_mask
    
    return lock_masks, edit_masks
