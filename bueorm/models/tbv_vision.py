"""
Bueorm Models - TBV Vision Model (TBVVisionModel)
Bidirectional vision transformation T(x, d) with shared weights and feature extraction for VLM multimodal tasks.
"""

import torch
import torch.nn as nn
from typing import Union, Optional, Tuple, Dict, Any

from bueorm.config import BueormConfig
from bueorm.core.registry import register_model, register_backbone
from TBV.config import TBVConfig
from TBV.model.tbv_network import TBVNetwork


@register_model("tbv")
@register_model("tbv_vision")
@register_model("TBVVisionModel")
@register_backbone("tbv")
class TBVVisionModel(nn.Module):
    """
    T-Bidirectional Vision (TBV) Model.
    Single unified network with shared weights executing T(Image, +1) -> Z and T(Z, -1) -> Image.
    """
    def __init__(self, config: Optional[BueormConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = BueormConfig(model_type="tbv", **kwargs)
        self.config = config

        tbv_cfg = TBVConfig(
            image_size=config.image_size,
            in_channels=config.in_channels,
            patch_size=config.patch_size,
            dim=config.tbv_dim,
            num_blocks=config.tbv_num_blocks,
            spatial_kernel_size=config.tbv_spatial_kernel,
            vlm_proj_dim=config.d_model if config.is_multimodal else None,
            use_2d_pos_emb=True
        )
        self.net = TBVNetwork(tbv_cfg)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """T(+1): Image -> Latent Grid Z."""
        return self.net.encode(images)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """T(-1): Latent Grid Z -> Reconstructed Image."""
        return self.net.decode(latents)

    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        """Full image reconstruction cycle: Image -> Z -> Image."""
        return self.net.reconstruct(images)

    def reverse_cycle(self, latents: torch.Tensor) -> torch.Tensor:
        """Reverse latent cycle: Z -> Image -> Z'."""
        return self.net.reverse_cycle(latents)

    def extract_features(self, images: torch.Tensor, return_tokens: bool = True) -> torch.Tensor:
        """Extracts spatial visual features or VLM token sequence for downstream language models."""
        return self.net.extract_features(images, return_tokens=return_tokens)

    def forward(self, input_tensor: torch.Tensor, direction: Union[int, torch.Tensor] = 1) -> torch.Tensor:
        return self.net(input_tensor, direction=direction)

    def infer(self, images: torch.Tensor, mode: str = "reconstruct", **kwargs) -> Any:
        from bueorm.core.inference import InferencePipeline
        pipe = InferencePipeline(self)
        return pipe(images=images, mode=mode, **kwargs)
