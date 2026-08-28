"""
T-Bidirectional Vision (TBV) - Configuration
Spec V0.1 + VLM Feature Extraction Extensions
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import json


@dataclass
class TBVConfig:
    """
    Configuration for T-Bidirectional Vision (TBV) Architecture.
    
    A single neural network transformation T(x, d) with shared weights
    across both forward (+1: Image -> Latent) and reverse (-1: Latent -> Image) directions.
    
    Attributes:
        image_size (int): Spatial resolution of input images (H = W). Default: 128.
        in_channels (int): Input image channels (e.g. 3 for RGB). Default: 3.
        patch_size (int): Non-overlapping spatial patch size P. Default: 8.
        dim (int): Latent channel dimension D in the grid state. Default: 64.
        num_blocks (int): Number of shared T-Blocks in the transformation. Default: 4.
        mlp_expansion (float): Channel expansion factor in T-Block MLP. Default: 2.0.
        spatial_kernel_size (int): Kernel size for local spatial depthwise convolution. Default: 3.
        dir_emb_dim (int): Embedding dimension for direction modulation (+1 vs -1). Default: 16.
        activation (str): Activation function ('gelu', 'silu', 'relu'). Default: 'gelu'.
        norm_eps (float): Epsilon for normalization layers. Default: 1e-5.
        res_scale_init (float): Initial standard deviation for residual projection weights. Default: 0.01.
        vlm_proj_dim (Optional[int]): Projection dimension for VLM/BDA token representation. Default: None.
        use_2d_pos_emb (bool): Whether to include learnable 2D positional embeddings for VLM feature tokens. Default: True.
    """
    image_size: int = 128
    in_channels: int = 3
    patch_size: int = 8
    dim: int = 64
    num_blocks: int = 4
    mlp_expansion: float = 2.0
    spatial_kernel_size: int = 3
    dir_emb_dim: int = 16
    activation: str = "gelu"
    norm_eps: float = 1e-5
    res_scale_init: float = 0.01
    vlm_proj_dim: Optional[int] = None
    use_2d_pos_emb: bool = True

    @property
    def grid_size(self) -> int:
        """Spatial grid length (H // patch_size)."""
        assert self.image_size % self.patch_size == 0, (
            f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"
        )
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        """Total number of spatial patches."""
        g = self.grid_size
        return g * g

    @property
    def patch_dim(self) -> int:
        """Dimension of flattened raw patch (P * P * in_channels)."""
        return self.patch_size * self.patch_size * self.in_channels

    @property
    def compression_ratio(self) -> float:
        """Theoretical compression ratio: (in_channels * P^2) / dim."""
        return self.patch_dim / self.dim

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TBVConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "TBVConfig":
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def tiny(cls, **kwargs) -> "TBVConfig":
        """Tiny preset: 128x128 image, 8x8 patch -> 16x16 grid, D=64, 4 blocks."""
        cfg = cls(
            image_size=128,
            in_channels=3,
            patch_size=8,
            dim=64,
            num_blocks=4,
            mlp_expansion=2.0,
            spatial_kernel_size=3,
        )
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def small(cls, **kwargs) -> "TBVConfig":
        """Small preset: 256x256 image, 16x16 patch -> 16x16 grid, D=128, 6 blocks."""
        cfg = cls(
            image_size=256,
            in_channels=3,
            patch_size=16,
            dim=128,
            num_blocks=6,
            mlp_expansion=2.0,
            spatial_kernel_size=3,
        )
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def vlm(cls, vlm_proj_dim: int = 256, **kwargs) -> "TBVConfig":
        """VLM preset: configured for direct feature extraction and token projection for language models."""
        cfg = cls(
            image_size=224,
            in_channels=3,
            patch_size=14,
            dim=128,
            num_blocks=6,
            mlp_expansion=2.5,
            vlm_proj_dim=vlm_proj_dim,
            use_2d_pos_emb=True,
        )
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg
