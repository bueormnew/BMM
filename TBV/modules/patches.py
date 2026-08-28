"""
T-Bidirectional Vision (TBV) - Patch Projection & Deprojection
Converts between pixel images (B, C, H, W) and latent spatial grids (B, D, H/P, W/P).
"""

import torch
import torch.nn as nn


class PatchProjection(nn.Module):
    """
    Direction +1: Projects Image (B, C, H, W) to Latent Grid (B, D, N_h, N_w).
    Uses non-overlapping Conv2d with kernel=patch_size and stride=patch_size.
    """
    def __init__(self, in_channels: int, dim: int, patch_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="linear")
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (B, C, H, W)
        Returns:
            Latent grid tensor of shape (B, D, H/P, W/P)
        """
        return self.proj(x)


class PatchDeprojection(nn.Module):
    """
    Direction -1: Projects Latent Grid (B, D, N_h, N_w) back to Image (B, C, H, W).
    Uses non-overlapping ConvTranspose2d with kernel=patch_size and stride=patch_size.
    """
    def __init__(self, dim: int, out_channels: int, patch_size: int):
        super().__init__()
        self.dim = dim
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.deproj = nn.ConvTranspose2d(dim, out_channels, kernel_size=patch_size, stride=patch_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.deproj.weight, mode="fan_out", nonlinearity="linear")
        if self.deproj.bias is not None:
            nn.init.zeros_(self.deproj.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent grid tensor of shape (B, D, N_h, N_w)
        Returns:
            Reconstructed image tensor of shape (B, C, H, W)
        """
        return self.deproj(z)
