"""
Bueorm Utilities - Fluent Model Builder & Sizing Calculator
Provides a lego-like block construction interface and parameter scaling estimation.
"""

from typing import Union, Optional, List, Dict, Any
import torch
import torch.nn as nn

from bueorm.config import BueormConfig, MoEConfig, VLMConfig
from bueorm.models.factory import BueormModel


class ModelBuilder:
    """
    Fluent, lego-like model builder for Bueorm architectures.
    
    Example:
        model = (
            bueorm.ModelBuilder()
            .with_vision(image_size=256, patch_size=16, tbv_dim=128)
            .with_language_hybrid(pattern="4bda:1attn", d_model=512, n_heads=8, n_layers=10, max_seq_len=16384)
            .with_moe(num_experts=8, top_k=2)
            .build()
        )
    """
    def __init__(self, model_name: str = "custom-bueorm-model"):
        self.config = BueormConfig(model_name=model_name)

    def with_vision(
        self,
        image_size: int = 128,
        patch_size: int = 8,
        tbv_dim: int = 64,
        num_blocks: int = 4,
        freeze: bool = False
    ) -> "ModelBuilder":
        """Adds TBV visual perception backbone."""
        self.config.is_multimodal = True
        self.config.image_size = image_size
        self.config.patch_size = patch_size
        self.config.tbv_dim = tbv_dim
        self.config.tbv_num_blocks = num_blocks
        self.config.vlm_config = VLMConfig(freeze_vision=freeze)
        return self

    def with_language_bda(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        n_layers: int = 8,
        max_seq_len: int = 16384,
        vocab_size: int = 32000
    ) -> "ModelBuilder":
        """Configures pure BDA recurrent language model."""
        self.config.model_type = "bda" if not self.config.is_multimodal else "vlm"
        self.config.d_model = d_model
        self.config.n_heads = n_heads
        self.config.n_kv_heads = n_kv_heads or n_heads
        self.config.head_dim = d_model // n_heads
        self.config.d_k = d_model // n_heads
        self.config.d_v = d_model // n_heads
        self.config.n_layers = n_layers
        self.config.max_seq_len = max_seq_len
        self.config.vocab_size = vocab_size
        return self

    def with_language_transformer(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        n_layers: int = 8,
        max_seq_len: int = 16384,
        vocab_size: int = 32000,
        use_flash_attn: bool = True
    ) -> "ModelBuilder":
        """Configures pure Transformer language model."""
        self.config.model_type = "transformer" if not self.config.is_multimodal else "vlm"
        self.config.d_model = d_model
        self.config.n_heads = n_heads
        self.config.n_kv_heads = n_kv_heads or n_heads
        self.config.head_dim = d_model // n_heads
        self.config.n_layers = n_layers
        self.config.max_seq_len = max_seq_len
        self.config.vocab_size = vocab_size
        self.config.use_flash_attn = use_flash_attn
        return self

    def with_language_hybrid(
        self,
        pattern: str = "4bda:1attn",
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        n_layers: int = 10,
        max_seq_len: int = 16384,
        vocab_size: int = 32000
    ) -> "ModelBuilder":
        """Configures Hybrid BDA + FlashAttention language model in desired proportions."""
        self.config.model_type = "hybrid" if not self.config.is_multimodal else "vlm"
        self.config.hybrid_pattern = pattern
        self.config.d_model = d_model
        self.config.n_heads = n_heads
        self.config.n_kv_heads = n_kv_heads or n_heads
        self.config.head_dim = d_model // n_heads
        self.config.d_k = d_model // n_heads
        self.config.d_v = d_model // n_heads
        self.config.n_layers = n_layers
        self.config.max_seq_len = max_seq_len
        self.config.vocab_size = vocab_size
        return self

    def with_moe(
        self,
        num_experts: int = 8,
        top_k: int = 2,
        layer_interval: int = 1,
        aux_loss_coef: float = 0.01
    ) -> "ModelBuilder":
        """Enables Mixture of Experts (MoE) across feed-forward blocks."""
        self.config.use_moe = True
        self.config.moe_layer_interval = layer_interval
        self.config.moe_config = MoEConfig(
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_coef=aux_loss_coef
        )
        return self

    def build(self) -> nn.Module:
        """Instantiates and returns the configured neural network."""
        return BueormModel.from_config(self.config)

    def get_config(self) -> BueormConfig:
        return self.config


def calculate_active_vs_total_params(config: BueormConfig) -> Dict[str, Any]:
    """
    Estimates total parameters and active parameters per token (especially relevant for MoE).
    """
    model = BueormModel.from_config(config)
    total_params = sum(p.numel() for p in model.parameters())

    if config.use_moe and config.moe_config is not None:
        E = config.moe_config.num_experts
        K = config.moe_config.top_k
        ffn_per_expert = int(2 * (config.d_model * config.mlp_ratio) / 3) * config.d_model * 3
        inactive_expert_params = config.n_layers * (E - K) * ffn_per_expert
        active_params = total_params - inactive_expert_params
    else:
        active_params = total_params

    return {
        "total_parameters": total_params,
        "total_parameters_m": total_params / 1e6,
        "active_parameters": active_params,
        "active_parameters_m": active_params / 1e6,
        "memory_mb_fp32": (total_params * 4) / (1024 ** 2),
        "memory_mb_fp16": (total_params * 2) / (1024 ** 2),
        "memory_mb_int8": (total_params * 1) / (1024 ** 2),
    }
