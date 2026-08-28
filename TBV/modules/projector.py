"""
T-Bidirectional Vision (TBV) - Visual Token Projector for VLM & BDA Integration
Converts spatial latent grid Z in R^{B x D x H_g x W_g} into visual token sequence Z_seq in R^{B x N_patches x D_vlm}
with learned 2D positional encodings for seamless multimodal coupling with BDA and Transformer language models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TBVVisualProjector(nn.Module):
    """
    Projects TBV latent representation Z into visual token embeddings for Vision-Language Models (VLM).
    
    Transforms:
        Z (B, D, H_g, W_g) -> Tokens (B, H_g * W_g, D_vlm)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid_size: int,
        use_2d_pos_emb: bool = True,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.use_2d_pos_emb = use_2d_pos_emb
        
        # 2D Positional Embeddings
        if use_2d_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, in_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        # Two-layer MLP projector with GELU (standard VLM adapter architecture)
        proj_hidden = hidden_dim if hidden_dim is not None else max(in_dim, out_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, proj_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent grid from TBV of shape (B, D, H_g, W_g)
        Returns:
            visual_tokens: Sequence of shape (B, N_patches, out_dim)
        """
        B, D, H_g, W_g = z.shape
        N = H_g * W_g
        
        # Flatten spatial dimensions: (B, D, H_g, W_g) -> (B, H_g * W_g, D)
        tokens = z.flatten(2).transpose(1, 2)  # (B, N, D)
        
        # Add 2D spatial positional embedding
        if self.use_2d_pos_emb and self.pos_embed is not None:
            if self.pos_embed.shape[1] == N:
                tokens = tokens + self.pos_embed
            else:
                # Interpolate positional embeddings if grid size changed
                pos_emb_grid = self.pos_embed.view(1, self.grid_size, self.grid_size, D).permute(0, 3, 1, 2)
                pos_emb_resized = F.interpolate(pos_emb_grid, size=(H_g, W_g), mode="bicubic", align_corners=False)
                pos_emb_flat = pos_emb_resized.permute(0, 2, 3, 1).flatten(1, 2)
                tokens = tokens + pos_emb_flat

        tokens = self.norm(tokens)
        visual_tokens = self.mlp(tokens)
        return visual_tokens
