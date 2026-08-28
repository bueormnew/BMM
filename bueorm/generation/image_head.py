"""
Bueorm Generation - TextToLatentHead & LatentToImageDecoder
Core modules for image generation via TBV.

Text hidden states (B,T,d_model) --pool--> (B,d_model) --MLP--> Z (B,D,H_g,W_g) --TBV decode--> Image (B,C,H,W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from bueorm.config import BueormConfig, ImageGenConfig
from TBV.model.tbv_network import TBVNetwork
from TBV.config import TBVConfig


class TextToLatentHead(nn.Module):
    """
    Maps pooled language hidden states to TBV latent grid Z.

    Supports any d_model -> latent dim D, any grid size H_g x W_g.
    Pooling: 'mean' | 'last' | 'max'
    Architecture: LN -> Linear(d_model -> hidden) -> GELU -> Linear(hidden -> D*H_g*W_g) -> reshape
    """
    def __init__(
        self,
        d_model: int,
        tbv_dim: int,
        grid_size: int,
        head_hidden_dim: Optional[int] = None,
        pooling: str = "mean",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.tbv_dim = tbv_dim
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.pooling = pooling

        hidden = head_hidden_dim or max(d_model, tbv_dim * 2)

        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, tbv_dim * grid_size * grid_size),
        )
        # init
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (B, T, d_model)
        returns: (B, d_model)
        """
        if self.pooling == "last":
            return hidden_states[:, -1, :]
        elif self.pooling == "max":
            return hidden_states.max(dim=1).values
        else:  # mean
            return hidden_states.mean(dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, d_model)
        Returns:
            Z: (B, D, H_g, W_g)
        """
        pooled = self.pool(hidden_states)  # (B, d_model)
        pooled = self.norm(pooled)
        flat = self.mlp(pooled)  # (B, D*H_g*W_g)
        B = hidden_states.shape[0]
        Z = flat.view(B, self.tbv_dim, self.grid_size, self.grid_size)
        return Z


class LatentToImageDecoder(nn.Module):
    """
    Thin wrapper around TBVNetwork decode path.
    Holds a TBVNetwork and exposes decode(Z) -> Image.
    Weights can be shared with encoder or independent.
    """
    def __init__(self, config: ImageGenConfig, trainable: bool = True):
        super().__init__()
        tbv_cfg = TBVConfig(
            image_size=config.image_size,
            in_channels=config.in_channels,
            patch_size=config.patch_size,
            dim=config.tbv_dim,
            num_blocks=config.tbv_num_blocks,
        )
        self.net = TBVNetwork(tbv_cfg)
        self.config = config
        if not trainable:
            for p in self.net.parameters():
                p.requires_grad = False

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Z: (B, D, H_g, W_g) -> Image (B, C, H, W)"""
        return self.net.decode(Z)

    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        return self.net.reconstruct(images)
