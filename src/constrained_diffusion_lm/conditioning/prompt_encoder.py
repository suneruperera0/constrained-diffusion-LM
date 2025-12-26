"""
Prompt encoding for instruction-guided text editing.

This module implements a simple but effective conditioning strategy:
prepend the edit instruction to the input text with a separator.

Format: "[CLS] <instruction> [SEP] <text to edit> [SEP]"

The model learns to condition its predictions on the instruction prefix.
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch

from constrained_diffusion_lm.data.tokenization import Tokenizer


@dataclass
class PromptedInput:
    """
    Input with prepended instruction prompt.
    
    Attributes:
        input_ids: Full token sequence [prompt + separator + text]
        attention_mask: Attention mask for the sequence
        prompt_length: Length of prompt + separator (these are always "locked")
        text_start_idx: Index where the actual text starts
        text_lock_mask: Lock mask for the TEXT portion only
        full_lock_mask: Lock mask for full sequence (prompt always locked)
    """
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_length: int
    text_start_idx: int
    text_lock_mask: torch.Tensor
    full_lock_mask: torch.Tensor


class PromptEncoder:
    """
    Encodes edit instructions and prepends them to input text.
    
    The instruction is always treated as LOCKED (never masked during diffusion).
    This ensures the model always sees the full instruction when denoising.
    """
    
    # Common edit instruction templates
    INSTRUCTION_TEMPLATES = {
        "formal": "Make this more formal:",
        "casual": "Make this more casual:",
        "aggressive": "Make this more aggressive:",
        "polite": "Make this more polite:",
        "shorter": "Make this shorter:",
        "longer": "Make this longer:",
        "clearer": "Make this clearer:",
        "professional": "Make this more professional:",
        "friendly": "Make this more friendly:",
        "neutral": "Make this more neutral:",
    }
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        separator_token: str = "[SEP]",
        max_prompt_length: int = 32,
    ):
        """
        Initialize prompt encoder.
        
        Args:
            tokenizer: Tokenizer instance
            separator_token: Token to separate prompt from text
            max_prompt_length: Maximum length for prompt tokens
        """
        self.tokenizer = tokenizer
        self.separator_token = separator_token
        self.max_prompt_length = max_prompt_length
        
        # Get separator token ID
        self.sep_token_id = tokenizer.tokenizer.convert_tokens_to_ids(separator_token)
    
    def encode_prompt(
        self,
        instruction: str,
        add_separator: bool = True,
    ) -> Tuple[List[int], int]:
        """
        Encode an instruction prompt.
        
        Args:
            instruction: Edit instruction text
            add_separator: Whether to add separator token at the end
            
        Returns:
            Tuple of (token_ids, length)
        """
        # Handle template shortcuts
        if instruction.lower() in self.INSTRUCTION_TEMPLATES:
            instruction = self.INSTRUCTION_TEMPLATES[instruction.lower()]
        
        # Encode instruction (no special tokens)
        prompt_ids = self.tokenizer.tokenizer.encode(
            instruction,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_length,
        )
        
        # Add separator
        if add_separator:
            prompt_ids.append(self.sep_token_id)
        
        return prompt_ids, len(prompt_ids)
    
    def create_prompted_input(
        self,
        text: str,
        instruction: str,
        text_lock_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
    ) -> PromptedInput:
        """
        Create input with prepended instruction.
        
        Format: [CLS] <instruction> [SEP] <text tokens> [SEP]
        
        Args:
            text: Text to edit
            instruction: Edit instruction
            text_lock_mask: Lock mask for the text portion
            max_length: Maximum total sequence length
            
        Returns:
            PromptedInput with all necessary tensors
        """
        max_length = max_length or self.tokenizer.max_length
        
        # Encode instruction
        prompt_ids, prompt_length = self.encode_prompt(instruction)
        
        # Add CLS at the beginning
        prompt_ids = [self.tokenizer.cls_token_id] + prompt_ids
        prompt_length += 1
        
        # Encode text (without CLS, we already have it)
        text_ids = self.tokenizer.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - prompt_length - 1,  # Leave room for final SEP
        )
        
        # Add final SEP
        text_ids.append(self.sep_token_id)
        text_length = len(text_ids)
        
        # Combine
        full_ids = prompt_ids + text_ids
        seq_len = len(full_ids)
        
        # Create attention mask
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        
        # Create lock mask
        # Prompt is ALWAYS locked (never masked during diffusion)
        full_lock_mask = torch.zeros(seq_len, dtype=torch.bool)
        full_lock_mask[:prompt_length] = True  # Lock prompt
        
        # Apply text lock mask if provided
        if text_lock_mask is not None:
            # Shift text lock mask to account for prompt
            for i in range(min(len(text_lock_mask), text_length)):
                if text_lock_mask[i]:
                    full_lock_mask[prompt_length + i] = True
        
        return PromptedInput(
            input_ids=torch.tensor(full_ids, dtype=torch.long),
            attention_mask=attention_mask,
            prompt_length=prompt_length,
            text_start_idx=prompt_length,
            text_lock_mask=text_lock_mask if text_lock_mask is not None else torch.zeros(text_length, dtype=torch.bool),
            full_lock_mask=full_lock_mask,
        )
    
    def decode_output(
        self,
        output_ids: torch.Tensor,
        prompt_length: int,
    ) -> str:
        """
        Decode output, stripping the prompt portion.
        
        Args:
            output_ids: Full output token IDs
            prompt_length: Length of prompt to skip
            
        Returns:
            Decoded text (without prompt)
        """
        # Get text portion only
        text_ids = output_ids[prompt_length:]
        return self.tokenizer.decode(text_ids)
    
    def get_template(self, style: str) -> str:
        """Get instruction template for a style."""
        return self.INSTRUCTION_TEMPLATES.get(style.lower(), style)
    
    def list_templates(self) -> List[str]:
        """List available instruction templates."""
        return list(self.INSTRUCTION_TEMPLATES.keys())


def encode_with_prompt(
    tokenizer: Tokenizer,
    text: str,
    instruction: str,
    text_lock_mask: Optional[torch.Tensor] = None,
) -> PromptedInput:
    """
    Convenience function to encode text with instruction prompt.
    
    Args:
        tokenizer: Tokenizer
        text: Text to edit
        instruction: Edit instruction
        text_lock_mask: Optional lock mask for text
        
    Returns:
        PromptedInput
    """
    encoder = PromptEncoder(tokenizer)
    return encoder.create_prompted_input(text, instruction, text_lock_mask)


def create_prompted_input(
    tokenizer: Tokenizer,
    text: str,
    instruction: str,
    lock_substring: Optional[str] = None,
) -> PromptedInput:
    """
    Create prompted input with optional substring locking.
    
    Args:
        tokenizer: Tokenizer
        text: Text to edit
        instruction: Edit instruction
        lock_substring: Substring to lock in the text
        
    Returns:
        PromptedInput
    """
    from constrained_diffusion_lm.data.constraints import lock_substring as find_lock
    
    encoder = PromptEncoder(tokenizer)
    
    # Get text lock mask if substring specified
    text_lock_mask = None
    if lock_substring:
        text_lock_mask, _, _ = find_lock(text, lock_substring, tokenizer)
    
    return encoder.create_prompted_input(text, instruction, text_lock_mask)

