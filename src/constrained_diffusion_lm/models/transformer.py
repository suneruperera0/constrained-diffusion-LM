"""
Transformer architecture for the diffusion denoiser.

The model takes corrupted tokens x_t and timestep t, and predicts the clean tokens x_0.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embeddings (fixed, not learned).
    
    Used for both sequence positions and timestep encoding.
    """
    
    def __init__(self, dim: int, max_len: int = 10000):
        super().__init__()
        self.dim = dim
        
        # Precompute position encodings
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: Position indices [B, L] or [B]
            
        Returns:
            Position embeddings, same shape + dim
        """
        return self.pe[positions]


class TimestepEmbedding(nn.Module):
    """
    Timestep embedding using sinusoidal encoding + MLP projection.
    
    Maps scalar timestep t to a vector representation.
    """
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.sinusoidal = SinusoidalPositionEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep indices [B]
            
        Returns:
            Timestep embeddings [B, dim]
        """
        emb = self.sinusoidal(t)
        return self.mlp(emb)


class TransformerDenoiser(nn.Module):
    """
    Transformer-based denoiser for diffusion LM.
    
    Architecture:
    1. Token embedding + position embedding
    2. Add timestep embedding (broadcast to all positions)
    3. Transformer encoder layers
    4. Output projection to vocabulary logits
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        pad_token_id: int = 0,
    ):
        """
        Initialize the denoiser.
        
        Args:
            vocab_size: Size of vocabulary
            dim: Model hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dim_feedforward: FFN intermediate dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            pad_token_id: ID of padding token (for embedding initialization)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        self.pad_token_id = pad_token_id
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        
        # Position embedding (learned)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        
        # Timestep embedding
        self.timestep_embedding = TimestepEmbedding(dim)
        
        # Layer norm after embeddings
        self.embed_norm = nn.LayerNorm(dim)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values."""
        # Token embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        if self.pad_token_id is not None:
            self.token_embedding.weight.data[self.pad_token_id].zero_()
        
        # Position embeddings
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        
        # Output projection (small init for residual)
        nn.init.normal_(self.output_projection.weight, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: predict x_0 logits from x_t.
        
        Args:
            x_t: Corrupted token IDs [B, L]
            t: Timestep for each sample [B]
            attention_mask: Optional mask [B, L], 1 for real tokens, 0 for padding
            
        Returns:
            Logits over vocabulary [B, L, vocab_size]
        """
        batch_size, seq_len = x_t.shape
        device = x_t.device
        
        # Token embeddings [B, L, D]
        tok_emb = self.token_embedding(x_t)
        
        # Position embeddings [B, L, D]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        # Timestep embedding [B, D] -> [B, 1, D] for broadcasting
        time_emb = self.timestep_embedding(t).unsqueeze(1)
        
        # Combine embeddings
        hidden = tok_emb + pos_emb + time_emb
        hidden = self.embed_norm(hidden)
        hidden = self.embed_dropout(hidden)
        
        # Create attention mask for transformer
        # PyTorch expects: True = ignore, False = attend
        if attention_mask is not None:
            # Convert from (1=attend, 0=ignore) to (True=ignore, False=attend)
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Transformer layers
        hidden = self.transformer(hidden, src_key_padding_mask=src_key_padding_mask)
        
        # Output projection
        hidden = self.output_norm(hidden)
        logits = self.output_projection(hidden)
        
        return logits
    
    def get_num_params(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @classmethod
    def from_pretrained_bert(
        cls,
        bert_model_name: str = "bert-base-uncased",
        max_seq_len: int = 256,
        freeze_embeddings: bool = False,
    ) -> "TransformerDenoiser":
        """
        Create a TransformerDenoiser initialized from BERT weights.
        
        This copies:
        - Token embeddings
        - Position embeddings  
        - Transformer encoder layers
        
        The timestep embedding and output projection are randomly initialized.
        
        Args:
            bert_model_name: HuggingFace BERT model name
            max_seq_len: Maximum sequence length
            freeze_embeddings: Whether to freeze embedding layers
            
        Returns:
            TransformerDenoiser with BERT weights
        """
        from transformers import BertModel, BertConfig
        
        print(f"Loading BERT weights from {bert_model_name}...")
        bert = BertModel.from_pretrained(bert_model_name)
        config = bert.config
        
        # Create our model with matching dimensions
        model = cls(
            vocab_size=config.vocab_size,
            dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            max_seq_len=max_seq_len,
            pad_token_id=config.pad_token_id,
        )
        
        # Copy token embeddings
        model.token_embedding.weight.data.copy_(bert.embeddings.word_embeddings.weight.data)
        
        # Copy position embeddings (truncate or pad as needed)
        bert_pos = bert.embeddings.position_embeddings.weight.data
        our_max_len = model.position_embedding.weight.shape[0]
        bert_max_len = bert_pos.shape[0]
        copy_len = min(our_max_len, bert_max_len)
        model.position_embedding.weight.data[:copy_len] = bert_pos[:copy_len]
        
        # Copy LayerNorm after embeddings
        model.embed_norm.weight.data.copy_(bert.embeddings.LayerNorm.weight.data)
        model.embed_norm.bias.data.copy_(bert.embeddings.LayerNorm.bias.data)
        
        # Copy transformer encoder layers
        for i, bert_layer in enumerate(bert.encoder.layer):
            our_layer = model.transformer.layers[i]
            
            # Self-attention weights
            # BERT: separate Q, K, V projections -> PyTorch: combined in_proj
            q_weight = bert_layer.attention.self.query.weight.data
            k_weight = bert_layer.attention.self.key.weight.data
            v_weight = bert_layer.attention.self.value.weight.data
            our_layer.self_attn.in_proj_weight.data.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
            
            q_bias = bert_layer.attention.self.query.bias.data
            k_bias = bert_layer.attention.self.key.bias.data
            v_bias = bert_layer.attention.self.value.bias.data
            our_layer.self_attn.in_proj_bias.data.copy_(torch.cat([q_bias, k_bias, v_bias], dim=0))
            
            # Output projection
            our_layer.self_attn.out_proj.weight.data.copy_(bert_layer.attention.output.dense.weight.data)
            our_layer.self_attn.out_proj.bias.data.copy_(bert_layer.attention.output.dense.bias.data)
            
            # FFN
            our_layer.linear1.weight.data.copy_(bert_layer.intermediate.dense.weight.data)
            our_layer.linear1.bias.data.copy_(bert_layer.intermediate.dense.bias.data)
            our_layer.linear2.weight.data.copy_(bert_layer.output.dense.weight.data)
            our_layer.linear2.bias.data.copy_(bert_layer.output.dense.bias.data)
            
            # Layer norms (note: BERT is post-norm, we use pre-norm, but weights still transfer)
            our_layer.norm1.weight.data.copy_(bert_layer.attention.output.LayerNorm.weight.data)
            our_layer.norm1.bias.data.copy_(bert_layer.attention.output.LayerNorm.bias.data)
            our_layer.norm2.weight.data.copy_(bert_layer.output.LayerNorm.weight.data)
            our_layer.norm2.bias.data.copy_(bert_layer.output.LayerNorm.bias.data)
        
        # Optionally freeze embeddings
        if freeze_embeddings:
            model.token_embedding.weight.requires_grad = False
            model.position_embedding.weight.requires_grad = False
            model.embed_norm.weight.requires_grad = False
            model.embed_norm.bias.requires_grad = False
        
        # Clean up BERT model
        del bert
        
        print(f"Initialized TransformerDenoiser from BERT: {model.get_num_params():,} parameters")
        return model
