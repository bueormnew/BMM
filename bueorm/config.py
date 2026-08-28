"""
Bueorm Framework - Configuration Architecture
Universal, extensible configuration system for Language Models, Vision Models,
Hybrid Architectures, Multimodal VLMs, and Mixture of Experts (MoE).
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union
import json


@dataclass
class MoEConfig:
    """
    Configuration for Mixture of Experts (MoE) modules.
    
    Attributes:
        num_experts (int): Total number of expert networks E. Default: 8.
        top_k (int): Number of active experts routed per token. Default: 2.
        router_type (str): Type of gating network ('top_k', 'noisy_top_k', 'soft'). Default: 'top_k'.
        aux_loss_coef (float): Multiplier for auxiliary load balancing loss. Default: 0.01.
        expert_capacity_factor (float): Optional capacity limit factor for expert dispatching. Default: 1.25.
        drop_tokens (bool): Whether to drop tokens exceeding expert capacity. Default: False.
        router_jitter_noise (float): Multiplicative jitter noise added to router during training. Default: 0.01.
    """
    num_experts: int = 8
    top_k: int = 2
    router_type: str = "top_k"
    aux_loss_coef: float = 0.01
    expert_capacity_factor: float = 1.25
    drop_tokens: bool = False
    router_jitter_noise: float = 0.01

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MoEConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class ImageGenConfig:
    """
    Configuration for Text-to-Image & Unified Image Generation.
    Enables any BDA / Transformer / Hybrid / MoE / VLM backbone to generate
    TBV latents Z and decode them to images via shared TBV decoder.

    Attributes:
        enabled (bool): Master switch. Default False (opt-in, no breakage).
        image_size (int): Target image resolution (H=W). Default 64.
        patch_size (int): TBV patch size. Default 8 -> grid 8x8 for 64px.
        tbv_dim (int): Latent channel dim D for Z. Default 32.
        tbv_num_blocks (int): Number of shared T-Blocks in TBV decoder. Default 4.
        in_channels (int): Output image channels. Default 3.
        head_hidden_dim (Optional[int]): Hidden dim of text->latent MLP. Default d_model.
        pooling (str): How to pool text hidden states: 'mean', 'last', 'max'. Default 'mean'.
        loss_type (str): 'mse' or 'l1'. Default 'mse'.
        loss_weight (float): Weight for image loss when co-training with text. Default 1.0.
        train_decoder (bool): Whether TBV decoder weights are trainable. Default True.
        backbone (str): For TTI standalone: which text backbone to use ('bda','transformer','hybrid'). Default 'bda'.
    """
    enabled: bool = False
    image_size: int = 64
    patch_size: int = 8
    tbv_dim: int = 32
    tbv_num_blocks: int = 4
    in_channels: int = 3
    head_hidden_dim: Optional[int] = None
    pooling: str = "mean"
    loss_type: str = "mse"
    loss_weight: float = 1.0
    train_decoder: bool = True
    backbone: str = "bda"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageGenConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class VLMConfig:
    """
    Configuration for Vision-Language Models (VLM).
    
    Attributes:
        vision_backbone (str): Name of visual encoder ('tbv', 'vit'). Default: 'tbv'.
        projector_type (str): Cross-modal adapter type ('mlp', 'linear', 'resampler'). Default: 'mlp'.
        projector_hidden_dim (Optional[int]): Hidden dimension of projector MLP. Default: None.
        use_2d_pos_emb (bool): Whether to inject 2D spatial coordinates into visual tokens. Default: True.
        freeze_vision (bool): Whether to freeze vision backbone weights during language tuning. Default: False.
    """
    vision_backbone: str = "tbv"
    projector_type: str = "mlp"
    projector_hidden_dim: Optional[int] = None
    use_2d_pos_emb: bool = True
    freeze_vision: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VLMConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class BueormConfig:
    """
    Master unified configuration dataclass for the Bueorm Framework.
    
    Supports:
        - BDA Models
        - TBV Vision Models
        - Transformer Models
        - Arbitrary Hybrid Models (e.g. 4 BDA : 1 FlashAttention)
        - Mixture of Experts (MoE)
        - Multimodal VLMs
    """
    # Model Identity
    model_type: str = "bda"  # "bda", "transformer", "hybrid", "tbv", "vlm"
    model_name: str = "bueorm-base"
    version: str = "1.0.0"
    
    # Core Architecture Dimensions
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    n_layers: int = 6
    vocab_size: int = 32000
    max_seq_len: int = 4096
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    norm_eps: float = 1e-6
    bias: bool = False
    
    # BDA Specific Hyperparameters
    d_k: int = 64
    d_v: int = 64
    rank_r: int = 16
    ema_lambda: float = 0.01
    stability_margin: float = 1.0
    chunk_size: int = 16
    
    # Transformer Specific Hyperparameters
    rope_theta: float = 10000.0
    use_rope: bool = True
    use_flash_attn: bool = True
    
    # Hybrid Architecture Specification
    # Examples: "4bda:1attn", "3bda:1attn", [4, 1], ["bda", "bda", "attn"]
    hybrid_pattern: Union[str, List[Union[str, int]]] = "4bda:1attn"
    
    # Vision (TBV) Parameters
    image_size: int = 128
    in_channels: int = 3
    patch_size: int = 8
    tbv_dim: int = 64
    tbv_num_blocks: int = 4
    tbv_spatial_kernel: int = 3
    
    # MoE (Mixture of Experts) Integration
    use_moe: bool = False
    moe_config: Optional[MoEConfig] = None
    moe_layer_interval: int = 1  # 1 = every layer is MoE, 2 = every 2nd layer
    
    # VLM Integration
    is_multimodal: bool = False
    vlm_config: Optional[VLMConfig] = None

    # Image Generation (text -> Z -> TBV -> image) – opt-in, backward compatible
    enable_image_gen: bool = False
    image_gen_config: Optional[ImageGenConfig] = None
    # For standalone TTI models: which backbone to use as text encoder
    tti_backbone: str = "bda"  # 'bda' | 'transformer' | 'hybrid'

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.head_dim is None and self.n_heads > 0:
            self.head_dim = self.d_model // self.n_heads
        if self.use_moe and self.moe_config is None:
            self.moe_config = MoEConfig()
        elif isinstance(self.moe_config, dict):
            self.moe_config = MoEConfig.from_dict(self.moe_config)
            
        if self.is_multimodal and self.vlm_config is None:
            self.vlm_config = VLMConfig()
        elif isinstance(self.vlm_config, dict):
            self.vlm_config = VLMConfig.from_dict(self.vlm_config)

        if self.enable_image_gen and self.image_gen_config is None:
            self.image_gen_config = ImageGenConfig(
                enabled=True,
                image_size=self.image_size,
                patch_size=self.patch_size,
                tbv_dim=self.tbv_dim,
                tbv_num_blocks=self.tbv_num_blocks,
                in_channels=self.in_channels,
            )
        elif isinstance(self.image_gen_config, dict):
            self.image_gen_config = ImageGenConfig.from_dict(self.image_gen_config)
        # Ensure enabled flag mirrors config if set
        if self.image_gen_config is not None and self.image_gen_config.enabled:
            self.enable_image_gen = True
        # Sync top-level vision params to image gen if still at defaults (avoid mismatch)
        if self.image_gen_config is not None and self.enable_image_gen:
            # If top-level was customized but image_gen stayed at default 64/8/32, sync it
            if self.image_gen_config.image_size == 64 and self.image_size != 64:
                self.image_gen_config.image_size = self.image_size
            if self.image_gen_config.patch_size == 8 and self.patch_size != 8:
                self.image_gen_config.patch_size = self.patch_size
            if self.image_gen_config.tbv_dim == 32 and self.tbv_dim != 32:
                # Only sync if tbv_dim default 32 and parent differs
                # Parent default is 64, so only sync when parent was explicitly set to non-64
                if self.tbv_dim != 64:
                    self.image_gen_config.tbv_dim = self.tbv_dim
            if self.image_gen_config.in_channels == 3 and self.in_channels != 3:
                self.image_gen_config.in_channels = self.in_channels

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.moe_config is not None:
            data["moe_config"] = self.moe_config.to_dict()
        if self.vlm_config is not None:
            data["vlm_config"] = self.vlm_config.to_dict()
        if self.image_gen_config is not None:
            data["image_gen_config"] = self.image_gen_config.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BueormConfig":
        raw = data.copy()
        moe_raw = raw.pop("moe_config", None)
        vlm_raw = raw.pop("vlm_config", None)
        img_raw = raw.pop("image_gen_config", None)
        
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in valid_keys}
        
        obj = cls(**filtered)
        if moe_raw:
            obj.moe_config = MoEConfig.from_dict(moe_raw) if isinstance(moe_raw, dict) else moe_raw
        if vlm_raw:
            obj.vlm_config = VLMConfig.from_dict(vlm_raw) if isinstance(vlm_raw, dict) else vlm_raw
        if img_raw:
            obj.image_gen_config = ImageGenConfig.from_dict(img_raw) if isinstance(img_raw, dict) else img_raw
        return obj

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "BueormConfig":
        return cls.from_dict(json.loads(json_str))

    # --- Preset Factory Methods ---

    @classmethod
    def bda_small(cls, **kwargs) -> "BueormConfig":
        cfg = cls(model_type="bda", d_model=256, n_heads=4, d_k=32, d_v=32, n_layers=4)
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def tbv_small(cls, **kwargs) -> "BueormConfig":
        cfg = cls(model_type="tbv", image_size=128, patch_size=8, tbv_dim=64, tbv_num_blocks=4)
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def transformer_small(cls, **kwargs) -> "BueormConfig":
        cfg = cls(model_type="transformer", d_model=256, n_heads=4, n_layers=4, use_flash_attn=True)
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def hybrid_4_to_1(cls, **kwargs) -> "BueormConfig":
        """4 BDA layers for every 1 FlashAttention layer."""
        cfg = cls(
            model_type="hybrid",
            d_model=256,
            n_heads=4,
            d_k=32,
            d_v=32,
            n_layers=10,
            hybrid_pattern="4bda:1attn"
        )
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def vlm_bda(cls, **kwargs) -> "BueormConfig":
        """Multimodal VLM with TBV vision backbone and BDA/Hybrid language model."""
        cfg = cls(
            model_type="vlm",
            is_multimodal=True,
            d_model=256,
            n_heads=4,
            d_k=32,
            d_v=32,
            n_layers=6,
            image_size=128,
            patch_size=8,
            tbv_dim=64,
            tbv_num_blocks=4,
            vlm_config=VLMConfig(vision_backbone="tbv", projector_type="mlp")
        )
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def moe_hybrid(cls, num_experts: int = 8, top_k: int = 2, **kwargs) -> "BueormConfig":
        """Hybrid BDA-FlashAttention model with Mixture of Experts."""
        cfg = cls(
            model_type="hybrid",
            use_moe=True,
            moe_config=MoEConfig(num_experts=num_experts, top_k=top_k),
            d_model=256,
            n_heads=4,
            d_k=32,
            d_v=32,
            n_layers=6,
            hybrid_pattern="4bda:1attn"
        )
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    # --- Image Generation Presets ---

    @classmethod
    def tti_small(cls, backbone: str = "bda", **kwargs) -> "BueormConfig":
        """Text-to-Image standalone (punto 1): texto -> Z -> TBV -> imagen."""
        cfg = cls(
            model_type="tti",
            tti_backbone=backbone,
            d_model=128,
            n_heads=4,
            n_layers=4,
            vocab_size=1000,
            image_size=64,
            patch_size=8,
            tbv_dim=32,
            tbv_num_blocks=4,
            enable_image_gen=True,
            image_gen_config=ImageGenConfig(
                enabled=True,
                image_size=kwargs.get("image_size", 64),
                patch_size=kwargs.get("patch_size", 8),
                tbv_dim=kwargs.get("tbv_dim", 32),
                backbone=backbone,
            ),
        )
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            elif hasattr(cfg.image_gen_config, k):
                setattr(cfg.image_gen_config, k, v)
        # Sync heads if d_model/n_heads changed without explicit n_kv_heads
        if "n_heads" in kwargs and "n_kv_heads" not in kwargs:
            cfg.n_kv_heads = cfg.n_heads
        if ("d_model" in kwargs or "n_heads" in kwargs) and cfg.n_heads > 0:
            if "head_dim" not in kwargs:
                cfg.head_dim = cfg.d_model // cfg.n_heads
            if "d_k" not in kwargs:
                cfg.d_k = cfg.d_model // cfg.n_heads
            if "d_v" not in kwargs:
                cfg.d_v = cfg.d_model // cfg.n_heads
        return cfg

    @classmethod
    def bda_with_image_gen(cls, **kwargs) -> "BueormConfig":
        """BDA nativo con generación de imagen (punto 2-3)."""
        cfg = cls(model_type="bda", d_model=256, n_heads=4, n_layers=6, enable_image_gen=True,
                  image_gen_config=ImageGenConfig(enabled=True, backbone="bda"))
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            if hasattr(cfg.image_gen_config, k):
                setattr(cfg.image_gen_config, k, v)
            # Sync common vision fields
            if k in ("image_size", "patch_size", "tbv_dim", "tbv_num_blocks", "in_channels"):
                setattr(cfg.image_gen_config, k, v)
        if "n_heads" in kwargs and "n_kv_heads" not in kwargs:
            cfg.n_kv_heads = cfg.n_heads
        if ("d_model" in kwargs or "n_heads" in kwargs) and cfg.n_heads > 0:
            if "head_dim" not in kwargs:
                cfg.head_dim = cfg.d_model // cfg.n_heads
            if "d_k" not in kwargs:
                cfg.d_k = cfg.d_model // cfg.n_heads
            if "d_v" not in kwargs:
                cfg.d_v = cfg.d_model // cfg.n_heads
        return cfg

    @classmethod
    def transformer_with_image_gen(cls, **kwargs) -> "BueormConfig":
        cfg = cls(model_type="transformer", d_model=256, n_heads=4, n_layers=6, enable_image_gen=True,
                  image_gen_config=ImageGenConfig(enabled=True, backbone="transformer"))
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            if hasattr(cfg.image_gen_config, k):
                setattr(cfg.image_gen_config, k, v)
            if k in ("image_size", "patch_size", "tbv_dim", "tbv_num_blocks", "in_channels"):
                setattr(cfg.image_gen_config, k, v)
        if "n_heads" in kwargs and "n_kv_heads" not in kwargs:
            cfg.n_kv_heads = cfg.n_heads
        if ("d_model" in kwargs or "n_heads" in kwargs) and cfg.n_heads > 0:
            if "head_dim" not in kwargs:
                cfg.head_dim = cfg.d_model // cfg.n_heads
        return cfg

    @classmethod
    def hybrid_with_image_gen(cls, pattern: str = "4bda:1attn", **kwargs) -> "BueormConfig":
        cfg = cls(model_type="hybrid", d_model=256, n_heads=4, n_layers=6, hybrid_pattern=pattern,
                  enable_image_gen=True, image_gen_config=ImageGenConfig(enabled=True, backbone="hybrid"))
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            if hasattr(cfg.image_gen_config, k):
                setattr(cfg.image_gen_config, k, v)
            if k in ("image_size", "patch_size", "tbv_dim", "tbv_num_blocks", "in_channels"):
                setattr(cfg.image_gen_config, k, v)
        if "n_heads" in kwargs and "n_kv_heads" not in kwargs:
            cfg.n_kv_heads = cfg.n_heads
        if ("d_model" in kwargs or "n_heads" in kwargs) and cfg.n_heads > 0:
            if "head_dim" not in kwargs:
                cfg.head_dim = cfg.d_model // cfg.n_heads
            if "d_k" not in kwargs:
                cfg.d_k = cfg.d_model // cfg.n_heads
            if "d_v" not in kwargs:
                cfg.d_v = cfg.d_model // cfg.n_heads
        return cfg

    @classmethod
    def vlm_with_image_gen(cls, **kwargs) -> "BueormConfig":
        """VLM bidireccional con generación (visión<->texto<->imagen)."""
        cfg = cls.vlm_bda(enable_image_gen=True,
                          image_gen_config=ImageGenConfig(enabled=True, backbone="hybrid"))
        cfg.is_multimodal = True
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            if hasattr(cfg.image_gen_config, k):
                setattr(cfg.image_gen_config, k, v)
            if k in ("image_size", "patch_size", "tbv_dim", "tbv_num_blocks", "in_channels"):
                setattr(cfg.image_gen_config, k, v)
                # For VLM, also keep top-level in sync for encoder
                if k == "image_size":
                    cfg.image_size = v
                if k == "patch_size":
                    cfg.patch_size = v
                if k == "tbv_dim":
                    cfg.tbv_dim = v
        if "n_heads" in kwargs and "n_kv_heads" not in kwargs:
            cfg.n_kv_heads = cfg.n_heads
        if ("d_model" in kwargs or "n_heads" in kwargs) and cfg.n_heads > 0:
            if "head_dim" not in kwargs:
                cfg.head_dim = cfg.d_model // cfg.n_heads
            if "d_k" not in kwargs:
                cfg.d_k = cfg.d_model // cfg.n_heads
            if "d_v" not in kwargs:
                cfg.d_v = cfg.d_model // cfg.n_heads
        return cfg
