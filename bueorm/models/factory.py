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

# Generative imports (lazy for registry side-effects)
try:
    from bueorm.models.text_to_image import TextToImageModel
    from bueorm.models.generative_language import BDAWithImageGen, TransformerWithImageGen, HybridWithImageGen
    from bueorm.models.generative_vlm import GenerativeVLM
except Exception:
    TextToImageModel = None
    BDAWithImageGen = None
    TransformerWithImageGen = None
    HybridWithImageGen = None
    GenerativeVLM = None


class BueormModel:
    """
    Universal Factory for building, loading and instantiating any Bueorm model architecture.
    """
    @staticmethod
    def from_config(config: BueormConfig) -> nn.Module:
        """Instantiates a model directly from a BueormConfig."""
        mtype = config.model_type.lower()
        is_gen = getattr(config, "enable_image_gen", False) and getattr(config, "image_gen_config", None) is not None and config.image_gen_config.enabled

        # Text-to-Image standalone (punto 1) — always generative
        if mtype in ("tti", "text_to_image", "text2image"):
            if TextToImageModel is not None:
                return TextToImageModel(config)
            raise ValueError("TextToImageModel not available")

        # Any-to-Any / VLM with generation (punto 2-3, multimodal)
        if is_gen and (config.is_multimodal or mtype in ("vlm", "bueorm_vlm", "vlm_with_image", "vlm_gen", "generative_vlm", "any_to_any")):
            if GenerativeVLM is not None:
                return GenerativeVLM(config)
            return BueormVLM(config)

        # Language models with image generation (puntos 2-3)
        if is_gen:
            if mtype in ("bda", "bda_lm", "bda_with_image", "bda_gen"):
                if BDAWithImageGen is not None:
                    return BDAWithImageGen(config)
                return BDALanguageModel(config)
            elif mtype in ("transformer", "transformer_lm", "transformer_with_image", "transformer_gen"):
                if TransformerWithImageGen is not None:
                    return TransformerWithImageGen(config)
                return TransformerLM(config)
            elif mtype in ("hybrid", "hybrid_lm", "hybrid_with_image", "hybrid_gen"):
                if HybridWithImageGen is not None:
                    return HybridWithImageGen(config)
                return HybridLanguageModel(config)

        # Standard non-generative routing
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
                # Also propagate vision fields to image_gen_config if enabled
                if getattr(config, "enable_image_gen", False) and getattr(config, "image_gen_config", None) is not None and hasattr(config.image_gen_config, k):
                    setattr(config.image_gen_config, k, v)
                if k in ("image_size", "patch_size", "tbv_dim", "tbv_num_blocks", "in_channels") and getattr(config, "image_gen_config", None) is not None:
                    setattr(config.image_gen_config, k, v)
        # Sync heads if n_heads/d_model changed without explicit n_kv_heads
        if "n_heads" in kwargs and "n_kv_heads" not in kwargs:
            config.n_kv_heads = config.n_heads
        if ("d_model" in kwargs or "n_heads" in kwargs) and config.n_heads > 0:
            if "head_dim" not in kwargs:
                config.head_dim = config.d_model // config.n_heads
            if config.model_type in ("bda", "hybrid", "tti") or getattr(config, "enable_image_gen", False):
                if "d_k" not in kwargs:
                    config.d_k = config.d_model // config.n_heads
                if "d_v" not in kwargs:
                    config.d_v = config.d_model // config.n_heads
        # Final sync: ensure image_gen_config backbone matches model_type if tti
        if getattr(config, "enable_image_gen", False) and config.image_gen_config is not None:
            if config.model_type == "tti" and config.image_gen_config.backbone == "bda" and getattr(config, "tti_backbone", "bda") != "bda":
                config.image_gen_config.backbone = config.tti_backbone
        return cls.from_config(config)


def create_model(model_type: str = "hybrid", config: Optional[BueormConfig] = None, **kwargs) -> nn.Module:
    """Convenience alias for BueormModel.create."""
    return BueormModel.create(model_type=model_type, config=config, **kwargs)
