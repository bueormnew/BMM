"""
T-Bidirectional Vision (TBV) Architecture
A single differentiable neural network for bidirectional image-latent transformations
and visual feature extraction for Vision-Language Models (VLM).
"""

from TBV.config import TBVConfig
from TBV.modules.patches import PatchProjection, PatchDeprojection
from TBV.modules.direction import DirectionModulation
from TBV.modules.t_block import LayerNorm2d, TBlock
from TBV.modules.projector import TBVVisualProjector
from TBV.model.tbv_network import TBVNetwork

__version__ = "0.2.0"

__all__ = [
    "TBVConfig",
    "PatchProjection",
    "PatchDeprojection",
    "DirectionModulation",
    "LayerNorm2d",
    "TBlock",
    "TBVVisualProjector",
    "TBVNetwork",
]
