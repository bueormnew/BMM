"""
Bueorm - Modular Deep Learning Architecture & Model Framework
Author: Gerson Fabián Buenahora Ormaza (BUEORM)

Empowers creation, hybrid composition, multimodal coupling (VLM), Mixture of Experts (MoE),
multi-format serialization (.bueorm, .safetensors, .gguf, .pt), quantization, and training.
"""

from bueorm.config import BueormConfig, MoEConfig, VLMConfig, ImageGenConfig
from bueorm.models.bda_lm import BDALanguageModel
from bueorm.models.tbv_vision import TBVVisionModel
from bueorm.models.transformer_lm import TransformerLM
from bueorm.models.hybrid_lm import HybridLanguageModel
from bueorm.models.multimodal_vlm import BueormVLM
from bueorm.models.text_to_image import TextToImageModel
from bueorm.models.generative_language import BDAWithImageGen, TransformerWithImageGen, HybridWithImageGen
from bueorm.models.generative_vlm import GenerativeVLM
from bueorm.models.factory import BueormModel, create_model
from bueorm.utils.builder import ModelBuilder, calculate_active_vs_total_params

from bueorm.core.registry import (
    register_model,
    register_layer,
    register_backbone,
    register_router,
    MODEL_REGISTRY,
    LAYER_REGISTRY,
    BACKBONE_REGISTRY,
    ROUTER_REGISTRY,
)
from bueorm.core.quantization import quantize_model, get_model_memory_footprint, Int8Linear
from bueorm.core.serialization import save_model, load_model
from bueorm.core.inference import InferencePipeline, pipeline
from bueorm.trainer.trainer import Trainer, TrainingArguments
from bueorm.hub.hub import export_model

__version__ = "1.0.0"
__author__ = "Gerson Fabián Buenahora Ormaza (BUEORM)"

__all__ = [
    # Configs
    "BueormConfig",
    "MoEConfig",
    "VLMConfig",
    "ImageGenConfig",
    # Builder & Tools
    "ModelBuilder",
    "calculate_active_vs_total_params",
    # Models
    "BueormModel",
    "create_model",
    "BDALanguageModel",
    "TBVVisionModel",
    "TransformerLM",
    "HybridLanguageModel",
    "BueormVLM",
    "TextToImageModel",
    "BDAWithImageGen",
    "TransformerWithImageGen",
    "HybridWithImageGen",
    "GenerativeVLM",
    # Core & Serialization
    "save_model",
    "load_model",
    "export_model",
    "quantize_model",
    "get_model_memory_footprint",
    "InferencePipeline",
    "pipeline",
    # Registry
    "register_model",
    "register_layer",
    "register_backbone",
    "register_router",
    "MODEL_REGISTRY",
    "LAYER_REGISTRY",
    "BACKBONE_REGISTRY",
    "ROUTER_REGISTRY",
    # Trainer
    "Trainer",
    "TrainingArguments",
]
