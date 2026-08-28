"""
Transformer Modules - Transformer Decoder Block
Pre-RMSNorm -> Causal Attention (MHA/GQA/FlashAttn) -> Residual -> Pre-RMSNorm -> SwiGLU -> Residual.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

from transformer.config import TransformerConfig
from transformer.attention.causal_attention import CausalSelfAttention
from transformer.attention.kv_cache import KVCache
from transformer.modules.norm import RMSNorm
from transformer.modules.ffn import SwiGLU


class TransformerBlock(nn.Module):
    """
    Standard modern Decoder Transformer Block.
    """
    def __init__(self, config: Optional[TransformerConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = TransformerConfig(**kwargs)
        self.config = config

        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLU(config.d_model, mlp_ratio=config.mlp_ratio, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Union[KVCache, Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[Union[KVCache, Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            kv_cache: Key-value cache state
            start_pos: Token position offset for RoPE during step generation
            
        Returns:
            out: (batch_size, seq_len, d_model)
            new_kv_cache: Updated KV cache
        """
        # Pre-Norm Attention Residual
        normed_x = self.attn_norm(x)
        attn_out, new_kv_cache = self.attn(normed_x, kv_cache=kv_cache, start_pos=start_pos)
        x = x + attn_out

        # Pre-Norm FFN Residual
        x = x + self.ffn(self.ffn_norm(x))

        return x, new_kv_cache
