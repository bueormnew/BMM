"""
TBVNetwork - Core neural network module for T-Bidirectional Vision Architecture.
Single differentiable transformation T(input, direction) with shared weights across directions
and direct feature extraction for Vision-Language Models (VLM) & BDA integration.
"""

import torch
import torch.nn as nn
from typing import Union, Optional, Tuple

from TBV.config import TBVConfig
from TBV.modules.patches import PatchProjection, PatchDeprojection
from TBV.modules.t_block import TBlock
from TBV.modules.projector import TBVVisualProjector


class TBVNetwork(nn.Module):
    """
    Single differentiable network T(x, d) implementing:
        - T(Image, +1) -> Latent Z (B, D, H_g, W_g)
        - T(Latent Z, -1) -> Image Reconstructed (B, C, H, W)
        - extract_features(Image) -> Visual Token Sequence (B, N_patches, D_vlm) for VLM/BDA

    All main TBlocks and parameters are strictly shared between both directions.
    """
    def __init__(self, config: Optional[TBVConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = TBVConfig(**kwargs)
        self.config = config

        # Projections
        self.patch_proj = PatchProjection(
            in_channels=config.in_channels,
            dim=config.dim,
            patch_size=config.patch_size,
        )
        self.patch_deproj = PatchDeprojection(
            dim=config.dim,
            out_channels=config.in_channels,
            patch_size=config.patch_size,
        )

        # Sequence of shared T-Blocks
        self.blocks = nn.ModuleList([
            TBlock(
                dim=config.dim,
                mlp_expansion=config.mlp_expansion,
                spatial_kernel_size=config.spatial_kernel_size,
                dir_emb_dim=config.dir_emb_dim,
                activation=config.activation,
                norm_eps=config.norm_eps,
                res_scale_init=config.res_scale_init,
            )
            for _ in range(config.num_blocks)
        ])

        # Optional VLM Token Projector (for multimodal BDA integration)
        if config.vlm_proj_dim is not None:
            self.vlm_projector = TBVVisualProjector(
                in_dim=config.dim,
                out_dim=config.vlm_proj_dim,
                grid_size=config.grid_size,
                use_2d_pos_emb=config.use_2d_pos_emb
            )
        else:
            self.vlm_projector = None

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Convenience method for direction +1: Image (B, C, H, W) -> Latent Z (B, D, N_h, N_w)."""
        return self.forward(images, direction=1)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Convenience method for direction -1: Latent Z (B, D, N_h, N_w) -> Image (B, C, H, W)."""
        return self.forward(latents, direction=-1)

    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        """Full image reconstruction cycle: Image -> Z -> Image Reconstructed."""
        z = self.encode(images)
        return self.decode(z)

    def reverse_cycle(self, latents: torch.Tensor) -> torch.Tensor:
        """Reverse representation cycle: Z -> Image -> Z'."""
        img_rec = self.decode(latents)
        return self.encode(img_rec)

    def extract_features(
        self,
        images: torch.Tensor,
        return_tokens: bool = True
    ) -> torch.Tensor:
        """
        Feature extraction for VLM multimodal conditioning.
        
        Args:
            images: Tensor of shape (B, C, H, W)
            return_tokens: If True, flattens spatial grid into sequence (B, N_patches, D_vlm or D).
                           If False, returns 4D latent grid Z (B, D, H_g, W_g).
                           
        Returns:
            Visual representations for downstream language/BDA models.
        """
        # Direction +1: Image -> Latent Z
        z = self.encode(images)  # (B, D, H_g, W_g)
        
        if not return_tokens:
            return z
            
        if self.vlm_projector is not None:
            return self.vlm_projector(z)  # (B, N_patches, D_vlm)
        else:
            # Flatten spatial grid directly: (B, D, H_g, W_g) -> (B, H_g * W_g, D)
            return z.flatten(2).transpose(1, 2)

    def forward(
        self,
        input_tensor: torch.Tensor,
        direction: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Main unified forward path T(x, d).

        Args:
            input_tensor:
                - If direction == +1 (0): Image tensor of shape (B, C, H, W)
                - If direction == -1 (1): Latent tensor of shape (B, D, N_h, N_w)
            direction: +1 (or 0) for Image->Latent, -1 (or 1) for Latent->Image.
                       Can be integer or 1D tensor.

        Returns:
            - If direction == +1: Latent Z (B, D, N_h, N_w)
            - If direction == -1: Image Reconstructed (B, C, H, W)
        """
        # Determine internal direction index: 0 for +1, 1 for -1
        if isinstance(direction, torch.Tensor):
            if direction.dtype in (torch.float32, torch.float64, torch.int32, torch.int64):
                dir_idx = torch.where(direction > 0, 0, 1)
            else:
                dir_idx = direction
            is_forward = (dir_idx[0].item() == 0) if dir_idx.numel() > 0 else True
        else:
            is_forward = (direction == 1 or direction == 0)
            dir_idx = 0 if is_forward else 1

        state = input_tensor

        if is_forward:
            # Direction +1: Image (B, C, H, W) -> Initial spatial grid state (B, D, N_h, N_w)
            state = self.patch_proj(state)

        # Process state through sequence of shared T-Blocks
        for block in self.blocks:
            state = block(state, dir_idx)

        if not is_forward:
            # Direction -1: Spatial grid state (B, D, N_h, N_w) -> Image (B, C, H, W)
            state = self.patch_deproj(state)

        return state
