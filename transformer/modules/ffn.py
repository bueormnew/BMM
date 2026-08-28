"""
Transformer Modules - Feed-Forward Networks
SwiGLU (Shazeer, 2020) and standard Gated Linear MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network:
        FFN(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        bias: bool = False
    ):
        super().__init__()
        if hidden_dim is None:
            # 2/3 * 4 * dim standard SwiGLU parameter scaling
            hidden_dim = int(2 * (dim * mlp_ratio) / 3)
            # Round to multiple of 64 or 256 for tensor core efficiency
            hidden_dim = ((hidden_dim + 63) // 64) * 64
            
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_gate.weight)
        nn.init.xavier_uniform_(self.w_up.weight)
        nn.init.xavier_uniform_(self.w_down.weight)
        if self.w_gate.bias is not None:
            nn.init.zeros_(self.w_gate.bias)
            nn.init.zeros_(self.w_up.bias)
            nn.init.zeros_(self.w_down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
