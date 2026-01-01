"""
BERT-based Diffusion LM that leverages pretrained MLM capabilities.

Key insight: BERT's MLM already knows how to predict masked tokens.
We only need to add timestep conditioning to make it a diffusion model.

This model:
1. Uses BERT's pretrained embeddings, encoder, and MLM head
2. Adds timestep embedding that conditions the prediction
3. Only trains the timestep embedding (everything else frozen)

This is fundamentally different from training a denoiser from scratch!
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertConfig


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding projected to model dimension.
    """
    
    def __init__(self, dim: int, max_timesteps: int = 1000):
        super().__init__()
        self.dim = dim
        
        # Sinusoidal position encoding for timesteps
        position = torch.arange(max_timesteps).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe = torch.zeros(max_timesteps, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe)
        
        # Project and transform
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep indices [B]
        Returns:
            Timestep embeddings [B, dim]
        """
        emb = self.pe[t]
        return self.mlp(emb)


class BertDiffusionLM(nn.Module):
    """
    Diffusion LM built on pretrained BERT MLM.
    
    The key innovation: instead of training a denoiser from scratch,
    we leverage BERT's existing MLM capability and only add timestep conditioning.
    
    Architecture:
        Input tokens x_t → BERT embeddings + Timestep embedding 
        → BERT encoder → BERT MLM head → Logits
    
    The timestep embedding is ADDED to token embeddings, allowing the model
    to understand "how noisy is this input" and adjust its predictions accordingly.
    """
    
    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        max_timesteps: int = 1000,
        freeze_bert: bool = True,
    ):
        """
        Initialize from pretrained BERT.
        
        Args:
            bert_model_name: HuggingFace BERT model name
            max_timesteps: Maximum diffusion timesteps
            freeze_bert: If True, freeze BERT weights (recommended)
        """
        super().__init__()
        
        print(f"Loading {bert_model_name} with MLM head...")
        self.bert = BertForMaskedLM.from_pretrained(bert_model_name)
        self.config = self.bert.config
        
        # Timestep embedding (the only new component)
        self.timestep_embedding = TimestepEmbedding(
            dim=self.config.hidden_size,
            max_timesteps=max_timesteps,
        )
        
        # Layer norm for combined embeddings
        self.time_norm = nn.LayerNorm(self.config.hidden_size)
        
        # Freeze BERT if requested
        if freeze_bert:
            self._freeze_bert()
        
        # Store vocab info
        self.vocab_size = self.config.vocab_size
        self.pad_token_id = self.config.pad_token_id
        self.max_timesteps = max_timesteps
        
        print(f"BertDiffusionLM initialized:")
        print(f"  Total params: {self.get_num_params():,}")
        print(f"  Trainable: {self.get_num_trainable_params():,}")
    
    def _freeze_bert(self):
        """Freeze all BERT parameters."""
        for param in self.bert.parameters():
            param.requires_grad = False
        print("  BERT weights frozen")
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: predict clean token logits from noisy input.
        
        Args:
            x_t: Noisy token IDs [B, L]
            t: Timestep for each sample [B]
            attention_mask: Optional mask [B, L]
            
        Returns:
            Logits over vocabulary [B, L, V]
        """
        batch_size, seq_len = x_t.shape
        
        # Get BERT embeddings
        embeddings = self.bert.bert.embeddings(x_t)  # [B, L, D]
        
        # Get timestep embedding and add to all positions
        time_emb = self.timestep_embedding(t)  # [B, D]
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, D]
        
        # Combine: token embeddings + timestep embedding
        embeddings = embeddings + time_emb
        embeddings = self.time_norm(embeddings)
        
        # Extended attention mask for BERT
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=x_t.device)
        
        extended_mask = attention_mask[:, None, None, :]  # [B, 1, 1, L]
        extended_mask = (1.0 - extended_mask) * -10000.0
        
        # Run through BERT encoder
        encoder_output = self.bert.bert.encoder(
            embeddings,
            attention_mask=extended_mask,
        )
        hidden_states = encoder_output.last_hidden_state  # [B, L, D]
        
        # Run through MLM head to get logits
        logits = self.bert.cls(hidden_states)  # [B, L, V]
        
        return logits
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_bert_diffusion():
    """Quick test of the model."""
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertDiffusionLM(freeze_bert=True)
    model.eval()
    
    # Test input
    text = "The weather is [MASK] today."
    inputs = tokenizer(text, return_tensors="pt")
    x_t = inputs["input_ids"]
    
    # Test at different timesteps
    for t_val in [1, 100, 500, 999]:
        t = torch.tensor([t_val])
        
        with torch.no_grad():
            logits = model(x_t, t)
        
        # Get prediction for [MASK] position
        mask_pos = (x_t == tokenizer.mask_token_id).nonzero()[0, 1]
        probs = logits[0, mask_pos].softmax(dim=-1)
        top_token = probs.argmax().item()
        top_prob = probs[top_token].item()
        
        print(f"t={t_val:4d}: '{tokenizer.decode([top_token])}' (prob={top_prob:.3f})")


if __name__ == "__main__":
    test_bert_diffusion()

