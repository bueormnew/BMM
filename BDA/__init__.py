"""
BUEORM Delta Attention (BDA)
A Recurrent Associative Memory Architecture for Large Scale Language Models
Author / Design Owner: Gerson Fabian Buenahora Ormaza - BUEORM
Spec v2 Reference Implementation
"""

from BDA.config import BDAConfig
from BDA.ops.state import MemoryState
from BDA.ops.gates import LRFG, DEM
from BDA.ops.normalization import ASN
from BDA.ops.stability import StabilityProjection
from BDA.layers.bda_step import bda_head_step
from BDA.layers.crpt import CRPTFunction, crpt_forward_native
from BDA.layers.bda_layer import BDALayer
from BDA.layers.attention import CausalFullAttention
from BDA.layers.block import RMSNorm, SwiGLU, BDABlock, HybridBlock
from BDA.models.hybrid_model import BDAHybridModel

__version__ = "2.0.0"
__author__ = "Gerson Fabián Buenahora Ormaza (BUEORM)"

__all__ = [
    "BDAConfig",
    "MemoryState",
    "LRFG",
    "DEM",
    "ASN",
    "StabilityProjection",
    "bda_head_step",
    "CRPTFunction",
    "crpt_forward_native",
    "BDALayer",
    "CausalFullAttention",
    "RMSNorm",
    "SwiGLU",
    "BDABlock",
    "HybridBlock",
    "BDAHybridModel",
]
