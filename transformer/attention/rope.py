"""
Transformer Attention - Rotary Position Embeddings (RoPE)
Su et al. (2021) RoFormer: Enhanced Transformer with Rotary Position Embedding.
Supports precomputed cos/sin frequency caching, dynamic length extension, and half-head complex rotation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    
    Applies rotary position transformations to queries and keys:
        q_rot = q * cos(theta) + rotate_half(q) * sin(theta)
    """
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Inverse frequencies: theta_i = base^(-2(i-1)/dim) for i in [1, ..., dim/2]
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute initial cosine and sine tables
        self._set_cos_sin_cache(max_seq_len, device=device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device], dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        # Outer product: (seq_len, dim/2)
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate to match full dim: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor to derive device and dtype from (e.g. queries)
            seq_len: Sequence length needed
        Returns:
            cos: Tensor of shape (seq_len, dim)
            sin: Tensor of shape (seq_len, dim)
        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
            
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies RoPE to query and key states.
    
    Args:
        q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, n_kv_heads, seq_len, head_dim)
        cos: Cosine frequencies of shape (seq_len, head_dim) or (batch, seq_len, head_dim)
        sin: Sine frequencies of shape (seq_len, head_dim) or (batch, seq_len, head_dim)
        position_ids: Optional tensor of token positions (batch, seq_len)
        
    Returns:
        q_embed: Rotated queries of shape (batch, n_heads, seq_len, head_dim)
        k_embed: Rotated keys of shape (batch, n_kv_heads, seq_len, head_dim)
    """
    # Reshape cos/sin to (1, 1, seq_len, head_dim) for broadcasting over batch and heads
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.ndim == 3:
        cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        sin = sin.unsqueeze(1)
        
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
