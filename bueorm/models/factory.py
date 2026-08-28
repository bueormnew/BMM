"""
Bueorm Models - Model Factory and AutoModel Interface
Provides universal instantiation from configs, pretrained files (.bueorm, .safetensors, .gguf, .pt),
and high-level architecture builders.
"""

import os
import torch
import torch.nn as nn
from typing import Union, Optional, Dict, Any

from bueorm.config import BueormConfig
from bueorm.core.registry import MODEL_REGISTRY
from bueorm.models.bda_lm import BDALanguageModel
from bueorm.models.tbv_vision import TBVVisionModel
from bueorm.models.transformer_lm import TransformerLM
from bueorm.models.hybrid_lm import HybridLanguageModel
from bueorm.models.multimodal_vlm import BueormVLM


class BueormModel:
    """
    Universal Factory for building, loading and instantiating any Bueorm model architecture.
    """
    @staticmethod
    def from_config(config: BueormConfig) -> nn.Module:
        """Instantiates a model directly from a BueormConfig."""
        mtype = config.model_type.lower()
        if config.is_multimodal or mtype in ("vlm", "bueorm_vlm"):
            return BueormVLM(config)
        elif mtype in ("bda", "bda_lm"):
            return BDALanguageModel(config)
        elif mtype in ("tbv", "tbv_vision"):
            return TBVVisionModel(config)
        elif mtype in ("transformer", "transformer_lm"):
            return TransformerLM(config)
        elif mtype in ("hybrid", "hybrid_lm"):
            return HybridLanguageModel(config)
        elif mtype in MODEL_REGISTRY:
            cls = MODEL_REGISTRY.get(mtype)
            return cls(config)
        else:
            # Default to Hybrid model
            return HybridLanguageModel(config)

    @staticmethod
    def from_pretrained(
        filepath: str,
        device: Optional[Union[str, torch.device]] = "cpu",
        **kwargs
    ) -> nn.Module:
        """
        Loads a complete pretrained model from a .bueorm, .safetensors, .gguf, or .pt file.
        """
        from bueorm.core.serialization import load_model
        return load_model(filepath, device=device, **kwargs)

    @classmethod
    def create(
        cls,
        model_type: str = "hybrid",
        config: Optional[BueormConfig] = None,
        **kwargs
    ) -> nn.Module:
        """
        High-level builder.
        
        Args:
            model_type: 'bda', 'tbv', 'transformer', 'hybrid', 'vlm'
            config: Optional pre-constructed BueormConfig
        """
        if config is None:
            config = BueormConfig(model_type=model_type, **kwargs)
        else:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        return cls.from_config(config)


def create_model(model_type: str = "hybrid", config: Optional[BueormConfig] = None, **kwargs) -> nn.Module:
    """Convenience alias for BueormModel.create."""
    return BueormModel.create(model_type=model_type, config=config, **kwargs)
