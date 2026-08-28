"""
Bueorm MoE - Expert Network Architectures
Implements specialized SwiGLU and MLP feed-forward expert blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLUExpert(nn.Module):
    """SwiGLU Expert module."""
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MLPExpert(nn.Module):
    """GELU / SiLU MLP Expert module."""
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.act = nn.GELU() if activation == "gelu" else nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
