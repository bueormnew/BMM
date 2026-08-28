"""
BUEORM Delta Attention (BDA) - Transformer Blocks (BDABlock & HybridBlock)
Spec v2: Section 6.4, 7.0
Residual Transformer blocks with RMSNorm, BDA / Full Attention, and SwiGLU MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, Dict

from BDA.config import BDAConfig
from BDA.layers.bda_layer import BDALayer
from BDA.layers.attention import CausalFullAttention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x.float() ** 2, dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps)
        return normed.to(dtype=x.dtype) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, d_model: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class BDABlock(nn.Module):
    """
    Residual Transformer Block with BDA layer and SwiGLU MLP.
    Structure:
        x = x + BDA(RMSNorm(x))
        x = x + MLP(RMSNorm(x))
    """
    def __init__(self, config: BDAConfig):
        super().__init__()
        self.config = config
        self.norm1 = RMSNorm(config.d_model, eps=config.eps)
        self.bda = BDALayer(config)
        self.norm2 = RMSNorm(config.d_model, eps=config.eps)
        hidden_dim = int(config.d_model * config.mlp_ratio * 2 / 3)
        self.mlp = SwiGLU(config.d_model, hidden_dim, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        normed_x = self.norm1(x)
        bda_out, new_state = self.bda(normed_x, state=state)
        x = x + bda_out
        x = x + self.mlp(self.norm2(x))
        return x, new_state

    def step(
        self,
        x_t: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        normed_x = self.norm1(x_t)
        bda_out, new_state = self.bda.step(normed_x, state=state)
        x_t = x_t + bda_out
        x_t = x_t + self.mlp(self.norm2(x_t))
        return x_t, new_state


class HybridBlock(nn.Module):
    """
    Hybrid Block containing either a BDA layer or a Causal Full Attention layer.
    """
    def __init__(self, config: BDAConfig, is_full_attention: bool = False):
        super().__init__()
        self.config = config
        self.is_full_attention = is_full_attention
        self.norm1 = RMSNorm(config.d_model, eps=config.eps)
        
        if is_full_attention:
            self.layer = CausalFullAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                dropout=config.dropout,
                bias=config.bias
            )
        else:
            self.layer = BDALayer(config)
            
        self.norm2 = RMSNorm(config.d_model, eps=config.eps)
        hidden_dim = int(config.d_model * config.mlp_ratio * 2 / 3)
        self.mlp = SwiGLU(config.d_model, hidden_dim, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        normed_x = self.norm1(x)
        if self.is_full_attention:
            layer_out, new_cache = self.layer(normed_x, kv_cache=cache)
        else:
            layer_out, new_cache = self.layer(normed_x, state=cache)
        x = x + layer_out
        x = x + self.mlp(self.norm2(x))
        return x, new_cache

    def step(
        self,
        x_t: torch.Tensor,
        cache: Optional[Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        normed_x = self.norm1(x_t)
        if self.is_full_attention:
            layer_out, new_cache = self.layer(normed_x.unsqueeze(1), kv_cache=cache)
            layer_out = layer_out.squeeze(1)
        else:
            layer_out, new_cache = self.layer.step(normed_x, state=cache)
        x_t = x_t + layer_out
        x_t = x_t + self.mlp(self.norm2(x_t))
        return x_t, new_cache
