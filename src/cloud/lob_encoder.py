"""
LOBERT: Limit Order Book Encoder Representation Transformer

This implements the architectural foundation of the LOBERT model for 
high-frequency trading microstructure analysis.
Ref: "LOBERT: Limit Order Book Encoder Representation Transformer"

Key Innovations:
1. One-Token-Per-Message (OTPM) Tokenizer using Piecewise Linear-Geometric Scaling (PLGS).
2. Continuous-time Rotary Position Embeddings (ROPE) for irregular tick intervals.
3. Masked Message Modeling (MMM) for pre-training.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

# ── Constants ──
D_MODEL = 128
NUM_HEADS = 8
NUM_LAYERS = 4
MAX_SEQ_LEN = 1024


class PLGSTokenizer(nn.Module):
    """
    Piecewise Linear-Geometric Scaling (PLGS) Tokenizer.
    Converts a continuous LOB message (price, volume, direction, type) 
    into a discrete token vocabulary index, resolving the "token explosion" issue.
    """
    def __init__(self, vocab_size: int = 10000):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, D_MODEL)

    def forward(self, message_features: torch.Tensor) -> torch.Tensor:
        """
        message_features: (B, SeqLen, Features)
        Returns: (B, SeqLen, D_MODEL)
        
        Note: In a full implementation, the PLGS mapping function quantizes
        price and volume logarithmically at the tails and linearly near the spread.
        Here we stub the mapping to a random projection for architecture completeness.
        """
        B, S, F = message_features.shape
        # Stub: Hash continuous features to a vocab index
        # In reality, this applies the PLGS bucketing algorithm
        pseudo_indices = (message_features.sum(dim=-1) * 1000).long().abs() % self.vocab_size
        return self.embedding(pseudo_indices)


class ContinuousTimeROPE(nn.Module):
    """
    Continuous-Time Rotary Position Embedding (ROPE).
    Unlike standard ROPE (which assumes integer positions 1, 2, 3...),
    this takes actual timestamp deltas, allowing the Transformer to understand
    microsecond bursts vs quiet periods in the order book.
    """
    def __init__(self, d_model: int, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.base = base
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, timestamps_ms: torch.Tensor) -> torch.Tensor:
        """
        x: (B, SeqLen, D_MODEL)
        timestamps_ms: (B, SeqLen) actual millisecond timestamps
        """
        # timestamps_ms shape: (B, SeqLen)
        # inv_freq shape: (D_MODEL // 2)
        # freqs shape: (B, SeqLen, D_MODEL // 2)
        freqs = torch.einsum("bs,d->bsd", timestamps_ms.float(), self.inv_freq)
        
        # Duplicate freqs to match D_MODEL: (B, SeqLen, D_MODEL)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        
        # Apply Rotary Position Embedding
        x1 = x[..., :self.d_model//2]
        x2 = x[..., self.d_model//2:]
        rotated_x = torch.cat((-x2, x1), dim=-1)
        
        return (x * cos) + (rotated_x * sin)


class LOBERTModel(nn.Module):
    """
    The LOBERT Encoder network.
    Takes limit order book messages with actual timestamps and produces
    rich contextual representations of the market microstructure.
    """
    def __init__(self, d_model: int = D_MODEL, num_heads: int = NUM_HEADS, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.tokenizer = PLGSTokenizer()
        self.rope = ContinuousTimeROPE(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Downstream task head: Predict pattern score (bullish/bearish microstructure)
        self.task_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, messages: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        messages: (B, SeqLen, Features)
        timestamps: (B, SeqLen) in milliseconds
        Returns: Pattern score [0, 1]
        """
        # 1. PLGS Tokenization
        x = self.tokenizer(messages)  # (B, S, D)
        
        # 2. Continuous-time ROPE
        x = self.rope(x, timestamps)  # (B, S, D)
        
        # 3. Transformer Encoder
        x = self.encoder(x)  # (B, S, D)
        
        # 4. Global Average Pooling over the sequence
        x_pooled = x.mean(dim=1)  # (B, D)
        
        # 5. Task Head
        return self.task_head(x_pooled).squeeze(-1)  # (B,)

