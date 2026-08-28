"""
Transformer Attention - Dynamic Key-Value Cache
Supports continuous appending and sliding window caching for autoregressive generation.
"""

import torch
from typing import Tuple, Optional


class KVCache:
    """
    Manages key and value states across autoregressive decoding steps.
    """
    def __init__(self, max_seq_len: int = 4096):
        self.max_seq_len = max_seq_len
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Appends new keys and values to the cache.
        
        Args:
            new_k: Tensor of shape (B, n_kv_heads, T_new, head_dim)
            new_v: Tensor of shape (B, n_kv_heads, T_new, head_dim)
            
        Returns:
            full_k: Tensor of shape (B, n_kv_heads, T_total, head_dim)
            full_v: Tensor of shape (B, n_kv_heads, T_total, head_dim)
        """
        if self.k is None:
            self.k = new_k
            self.v = new_v
        else:
            self.k = torch.cat([self.k, new_k], dim=-2)
            self.v = torch.cat([self.v, new_v], dim=-2)
            
        # Optional sliding window clamp
        if self.k.shape[-2] > self.max_seq_len:
            self.k = self.k[..., -self.max_seq_len :, :]
            self.v = self.v[..., -self.max_seq_len :, :]
            
        return self.k, self.v

    @property
    def seq_len(self) -> int:
        return self.k.shape[-2] if self.k is not None else 0

    def reset(self):
        self.k = None
        self.v = None
