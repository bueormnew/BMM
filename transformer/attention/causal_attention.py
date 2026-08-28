"""
Transformer Attention - Scalable Causal Self-Attention
Supports Multi-Head (MHA), Grouped-Query (GQA), and Multi-Query Attention (MQA),
FlashAttention (via PyTorch SDPA), Rotary Position Embeddings (RoPE), and KV-Caching.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union

from transformer.config import TransformerConfig
from transformer.attention.rope import RotaryEmbedding, apply_rotary_pos_emb
from transformer.attention.kv_cache import KVCache


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats key/value heads for Grouped-Query Attention (GQA):
    (B, n_kv_heads, seq_len, head_dim) -> (B, n_kv_heads * n_rep, seq_len, head_dim)
    """
    if n_rep == 1:
        return x
    B, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(B, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(B, n_kv_heads * n_rep, seq_len, head_dim)


class CausalSelfAttention(nn.Module):
    """
    Scalable Causal Self-Attention module.
    
    Features:
        - MHA / GQA / MQA support via configurable n_heads and n_kv_heads.
        - FlashAttention & Memory-Efficient Attention via torch.nn.functional.scaled_dot_product_attention.
        - Rotary Position Embeddings (RoPE).
        - KV-Cache management for low-latency generation.
    """
    def __init__(self, config: Optional[TransformerConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = TransformerConfig(**kwargs)
        self.config = config

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.num_queries_per_kv = config.num_queries_per_kv
        self.use_flash_attn = config.use_flash_attn
        self.use_rope = config.use_rope
        self.dropout = config.dropout

        # Projections: Q (d_model -> n_heads * head_dim), K,V (d_model -> n_kv_heads * head_dim)
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=config.bias)

        # RoPE
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                dim=self.head_dim,
                max_seq_len=config.max_seq_len,
                base=config.rope_theta
            )
        else:
            self.rotary_emb = None

        self.reset_parameters()

    def reset_parameters(self):
        # Standard scaled initialization for attention projections
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Union[KVCache, Tuple[torch.Tensor, torch.Tensor]]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[Union[KVCache, Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            kv_cache: Optional KVCache object or tuple of (past_k, past_v)
            attn_mask: Optional explicit attention mask
            start_pos: Starting position offset for RoPE during autoregressive generation
            
        Returns:
            output: Attention output of shape (batch_size, seq_len, d_model)
            new_kv_cache: Updated KV cache
        """
        B, T, _ = x.shape

        # 1. Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)       # (B, H_q, T, D_head)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)    # (B, H_kv, T, D_head)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)    # (B, H_kv, T, D_head)

        # 2. Apply Rotary Position Embedding (RoPE)
        if self.use_rope and self.rotary_emb is not None:
            total_len = start_pos + T
            cos, sin = self.rotary_emb(q, seq_len=total_len)
            cos = cos[start_pos : total_len]
            sin = sin[start_pos : total_len]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. Manage Key-Value Cache
        if kv_cache is not None:
            if isinstance(kv_cache, KVCache):
                k, v = kv_cache.update(k, v)
                new_kv_cache = kv_cache
            else:
                past_k, past_v = kv_cache
                k = torch.cat([past_k, k], dim=-2)
                v = torch.cat([past_v, v], dim=-2)
                new_kv_cache = (k, v)
        else:
            new_kv_cache = None

        # 4. GQA: Repeat KV heads across query groups if needed
        k_rep = repeat_kv(k, self.num_queries_per_kv)  # (B, H_q, T_total, D_head)
        v_rep = repeat_kv(v, self.num_queries_per_kv)  # (B, H_q, T_total, D_head)

        # 5. Compute Attention via PyTorch SDPA (FlashAttention kernel backend)
        is_causal = (T > 1) if (kv_cache is None or start_pos == 0) else False
        
        # When kv_cache has past tokens and we query a single token, causality is implicit by KV length
        dropout_p = self.dropout if self.training else 0.0
        
        out = F.scaled_dot_product_attention(
            q, k_rep, v_rep,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal
        )  # (B, H_q, T, D_head)

        # 6. Transpose, reshape and output projection
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        output = self.out_proj(out)

        return output, new_kv_cache
