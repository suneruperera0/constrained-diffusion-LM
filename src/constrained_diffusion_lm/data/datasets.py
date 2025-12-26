"""
Dataset implementations for ConstrainedDiffusionLM.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader

from constrained_diffusion_lm.data.tokenization import Tokenizer


class TextDataset(Dataset):
    """
    Simple text dataset for diffusion LM training.
    
    Loads text from various formats and tokenizes on-the-fly.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Tokenizer,
        max_length: Optional[int] = None,
        text_field: str = "text",
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to data file (jsonl, txt, or json)
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length (uses tokenizer default if None)
            text_field: Field name for text in JSON/JSONL files
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length or tokenizer.max_length
        self.text_field = text_field
        
        self.texts = self._load_data()
    
    def _load_data(self) -> List[str]:
        """Load text data from file."""
        suffix = self.data_path.suffix.lower()
        
        if suffix == ".jsonl":
            return self._load_jsonl()
        elif suffix == ".json":
            return self._load_json()
        elif suffix == ".txt":
            return self._load_txt()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _load_jsonl(self) -> List[str]:
        """Load from JSONL file (one JSON object per line)."""
        texts = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    texts.append(obj[self.text_field])
        return texts
    
    def _load_json(self) -> List[str]:
        """Load from JSON file (list of objects or list of strings)."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], str):
                return data
            else:
                return [item[self.text_field] for item in data]
        else:
            raise ValueError("JSON file must contain a list")
    
    def _load_txt(self) -> List[str]:
        """Load from plain text file (one text per line)."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single tokenized example.
        
        Returns:
            Dict with:
                - input_ids: [L] token IDs
                - attention_mask: [L] attention mask (1 for real tokens, 0 for padding)
                - length: scalar, actual sequence length before padding
        """
        text = self.texts[idx]
        
        # Encode without padding (we'll pad in collate_fn)
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "length": len(token_ids),
            "text": text,  # Keep original text for debugging
        }


class InMemoryTextDataset(Dataset):
    """
    Dataset from a list of texts in memory.
    
    Useful for quick testing and demos.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: Tokenizer,
        max_length: Optional[int] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length or tokenizer.max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "length": len(token_ids),
            "text": text,
        }


def collate_fn(
    batch: List[Dict],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Pads sequences to the maximum length in the batch.
    
    Args:
        batch: List of dataset items
        pad_token_id: Token ID to use for padding
        
    Returns:
        Dict with:
            - input_ids: [B, L] padded token IDs
            - attention_mask: [B, L] attention mask
            - lengths: [B] original lengths
    """
    # Get max length in this batch
    max_len = max(item["length"] for item in batch)
    
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    texts = []
    
    for i, item in enumerate(batch):
        seq_len = item["length"]
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = 1
        lengths[i] = seq_len
        texts.append(item["text"])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "texts": texts,
    }


def create_dataloader(
    dataset: Dataset,
    tokenizer: Tokenizer,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with proper collation.
    
    Args:
        dataset: Dataset instance
        tokenizer: Tokenizer (for pad_token_id)
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )
