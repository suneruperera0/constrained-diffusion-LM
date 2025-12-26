"""
Tokenization utilities for ConstrainedDiffusionLM.

Wraps HuggingFace tokenizers with a simple interface for diffusion LM training.
"""

from typing import List, Optional, Union
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


class Tokenizer:
    """
    Wrapper around HuggingFace tokenizers for diffusion LM.
    
    Provides consistent access to special tokens and encoding/decoding.
    """
    
    def __init__(
        self,
        name_or_path: str = "bert-base-uncased",
        max_length: int = 256,
    ):
        """
        Initialize tokenizer.
        
        Args:
            name_or_path: HuggingFace tokenizer name or path
            max_length: Maximum sequence length
        """
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = (
            AutoTokenizer.from_pretrained(name_or_path)
        )
        self.max_length = max_length
        
        # Cache special token IDs
        self._pad_token_id = self.tokenizer.pad_token_id
        self._mask_token_id = self.tokenizer.mask_token_id
        self._cls_token_id = self.tokenizer.cls_token_id
        self._sep_token_id = self.tokenizer.sep_token_id
        self._unk_token_id = self.tokenizer.unk_token_id
        
        # Ensure we have a mask token (critical for mask diffusion)
        if self._mask_token_id is None:
            raise ValueError(
                f"Tokenizer {name_or_path} does not have a [MASK] token. "
                "Mask diffusion requires a mask token."
            )
    
    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return self.tokenizer.vocab_size
    
    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return self._pad_token_id
    
    @property
    def mask_token_id(self) -> int:
        """Mask token ID."""
        return self._mask_token_id
    
    @property
    def cls_token_id(self) -> Optional[int]:
        """CLS token ID (if available)."""
        return self._cls_token_id
    
    @property
    def sep_token_id(self) -> Optional[int]:
        """SEP token ID (if available)."""
        return self._sep_token_id
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], "torch.Tensor"]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add [CLS], [SEP] etc.
            truncation: Whether to truncate to max_length
            max_length: Override default max_length
            return_tensors: "pt" for PyTorch tensors, None for list
            
        Returns:
            Token IDs as list or tensor
        """
        max_len = max_length or self.max_length
        
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_len,
        )
        
        if return_tensors == "pt":
            import torch
            return torch.tensor(encoded)
        
        return encoded
    
    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        truncation: bool = True,
        padding: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> dict:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of input texts
            add_special_tokens: Whether to add special tokens
            truncation: Whether to truncate
            padding: Whether to pad to max length in batch
            max_length: Override default max_length
            return_tensors: "pt" for PyTorch tensors
            
        Returns:
            Dict with 'input_ids' and 'attention_mask'
        """
        max_len = max_length or self.max_length
        
        return self.tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding=padding,
            max_length=max_len,
            return_tensors=return_tensors,
        )
    
    def decode(
        self,
        token_ids: Union[List[int], "torch.Tensor"],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip [CLS], [SEP], [PAD] etc.
            
        Returns:
            Decoded text
        """
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def decode_batch(
        self,
        token_ids_batch: Union[List[List[int]], "torch.Tensor"],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode a batch of token IDs.
        
        Args:
            token_ids_batch: Batch of token IDs [B, L]
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded texts
        """
        if hasattr(token_ids_batch, "tolist"):
            token_ids_batch = token_ids_batch.tolist()
        
        return self.tokenizer.batch_decode(
            token_ids_batch, 
            skip_special_tokens=skip_special_tokens
        )
    
    def get_token(self, token_id: int) -> str:
        """Get token string from ID."""
        return self.tokenizer.convert_ids_to_tokens(token_id)
    
    def get_token_id(self, token: str) -> int:
        """Get token ID from string."""
        return self.tokenizer.convert_tokens_to_ids(token)
    
    def __repr__(self) -> str:
        return (
            f"Tokenizer(name={self.tokenizer.name_or_path}, "
            f"vocab_size={self.vocab_size}, max_length={self.max_length})"
        )

