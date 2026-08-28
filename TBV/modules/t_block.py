"""
T-Bidirectional Vision (TBV) - TBlock Core Module
Maintains shared weights across both forward (+1) and reverse (-1) directions,
modulated by direction embedding.
"""

import torch
import torch.nn as nn
from typing import Union
from TBV.modules.direction import DirectionModulation


class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for 4D spatial tensors (B, C, H, W)."""
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=1, num_channels=num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gn(x)


class TBlock(nn.Module):
    """
    T-Block unit:
    y = x + P(M(C(L(N(x))), g_d))

    N: LayerNorm2d
    L: Depthwise Spatial Conv (3x3)
    C: Channel Transform (1x1 Conv -> Activation -> 1x1 Conv)
    M: Direction Modulation (gamma_d, beta_d)
    P: Residual Projection scaled by res_scale_init
    """
    def __init__(
        self,
        dim: int,
        mlp_expansion: float = 2.0,
        spatial_kernel_size: int = 3,
        dir_emb_dim: int = 16,
        activation: str = "gelu",
        norm_eps: float = 1e-5,
        res_scale_init: float = 0.01,
    ):
        super().__init__()
        self.dim = dim

        # 1. Normalization N
        self.norm = LayerNorm2d(dim, eps=norm_eps)

        # 2. Local Spatial Mix L (Depthwise Conv)
        padding = spatial_kernel_size // 2
        self.local_mix = nn.Conv2d(
            dim,
            dim,
            kernel_size=spatial_kernel_size,
            padding=padding,
            groups=dim,
            bias=True,
        )

        # 3. Channel Transform C (1x1 Conv -> Act -> 1x1 Conv)
        hidden_dim = int(dim * mlp_expansion)
        if activation == "gelu":
            act_layer = nn.GELU()
        elif activation == "silu":
            act_layer = nn.SiLU()
        else:
            act_layer = nn.ReLU()

        self.channel_transform = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            act_layer,
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )

        # 4. Direction Modulation M
        self.dir_modulation = DirectionModulation(dim, emb_dim=dir_emb_dim)

        # 5. Residual Projection P
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

        # Initialization: scale residual projection output to near zero for stability
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.proj.weight, std=res_scale_init)

    def forward(self, x: torch.Tensor, direction: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, D, H_grid, W_grid)
            direction: (B,) tensor or int (0 for +1, 1 for -1)
        Returns:
            Updated spatial tensor of shape (B, D, H_grid, W_grid)
        """
        h = self.norm(x)
        h = self.local_mix(h)
        h = self.channel_transform(h)
        h = self.dir_modulation(h, direction)
        h = self.proj(h)
        return x + h
