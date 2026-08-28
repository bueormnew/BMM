"""
Bueorm Models - TextToImageModel (Punto 1)
Sistema simple: texto -> Z latente -> TBV decode -> imagen.
Backbone texto configurable: bda | transformer | hybrid, con soporte MoE nativo.
No rompe nada existente: modelo independiente registrado como 'tti' / 'text_to_image'.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Any

from bueorm.config import BueormConfig
from bueorm.core.registry import register_model
from bueorm.generation.image_head import TextToLatentHead, LatentToImageDecoder


def _build_text_backbone(config: BueormConfig) -> nn.Module:
    """Instantiates text backbone per tti_backbone + MoE config."""
    backbone = getattr(config, "tti_backbone", "bda").lower()
    # Reuse existing LMs but without image-gen flag to avoid recursion
    # Create shallow copy of config for backbone with enable_image_gen=False
    import copy
    cfg_copy = copy.copy(config)
    cfg_copy.enable_image_gen = False
    # Keep image_gen_config but disabled for backbone build
    if backbone == "bda":
        from bueorm.models.bda_lm import BDALanguageModel
        return BDALanguageModel(cfg_copy)
    elif backbone == "transformer":
        from bueorm.models.transformer_lm import TransformerLM
        return TransformerLM(cfg_copy)
    elif backbone in ("hybrid", "hybrid_lm"):
        from bueorm.models.hybrid_lm import HybridLanguageModel
        return HybridLanguageModel(cfg_copy)
    else:
        from bueorm.models.bda_lm import BDALanguageModel
        return BDALanguageModel(cfg_copy)


@register_model("tti")
@register_model("text_to_image")
@register_model("text2image")
@register_model("TextToImageModel")
class TextToImageModel(nn.Module):
    """
    Standalone Text-to-Image model.
    - Text encoder: BDA / Transformer / Hybrid (+MoE si config.use_moe)
    - Head: TextToLatentHead (pooled hidden -> Z)
    - Decoder: TBV LatentToImageDecoder (Z -> Image)

    Uso:
        cfg = BueormConfig.tti_small(backbone="hybrid")
        model = create_model("tti", config=cfg)
        img = model.generate_image(prompt_ids)  # (B,C,H,W)
        # Entrenamiento:
        img_pred, loss, z = model(input_ids, target_images=images)
    """
    def __init__(self, config: Optional[BueormConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = BueormConfig.tti_small(**kwargs)
        # Ensure image gen config exists — sync with parent vision params
        if config.image_gen_config is None:
            from bueorm.config import ImageGenConfig
            config.image_gen_config = ImageGenConfig(
                enabled=True,
                image_size=config.image_size if config.image_size != 128 else 64,
                patch_size=config.patch_size,
                tbv_dim=config.tbv_dim if config.tbv_dim != 64 else 32,
                tbv_num_blocks=config.tbv_num_blocks,
                in_channels=config.in_channels,
                backbone=getattr(config, "tti_backbone", "bda"),
            )
            config.enable_image_gen = True
        else:
            # Sync if top-level was customized but image_gen stayed at default
            if config.image_gen_config.image_size == 64 and config.image_size not in (128, 64):
                config.image_gen_config.image_size = config.image_size
            if config.image_gen_config.patch_size == 8 and config.patch_size != 8:
                config.image_gen_config.patch_size = config.patch_size
            if config.image_gen_config.tbv_dim == 32 and config.tbv_dim not in (32, 64):
                # Only sync if parent was explicitly set to non-default
                if config.tbv_dim != 64:
                    config.image_gen_config.tbv_dim = config.tbv_dim
        igc = config.image_gen_config
        # Ensure enabled
        igc.enabled = True
        config.enable_image_gen = True
        self.config = config
        self.igc = igc

        grid_size = igc.image_size // igc.patch_size
        assert igc.image_size % igc.patch_size == 0, "image_size must be divisible by patch_size"

        # 1) Text backbone
        self.text_backbone = _build_text_backbone(config)

        # 2) Head: hidden -> Z
        # Need d_model from backbone config
        d_model = config.d_model
        self.image_head = TextToLatentHead(
            d_model=d_model,
            tbv_dim=igc.tbv_dim,
            grid_size=grid_size,
            head_hidden_dim=igc.head_hidden_dim,
            pooling=igc.pooling,
            dropout=config.dropout,
        )

        # 3) TBV decoder
        self.decoder = LatentToImageDecoder(igc, trainable=igc.train_decoder)

        # For returning hidden convenience
        self.vocab_size = config.vocab_size

    def _encode_text_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Clean unified implementation using backbone's own forward internals."""
        backbone = self.text_backbone
        B, T = input_ids.shape
        device = input_ids.device

        # BDA
        if backbone.__class__.__name__ == "BDALanguageModel":
            positions = torch.arange(0, T, device=device).unsqueeze(0)
            x = backbone.tok_embeddings(input_ids) + backbone.pos_embeddings(positions)
            x = backbone.drop(x)
            states = [None] * backbone.n_layers
            for i, layer in enumerate(backbone.layers):
                x, _, _ = layer(x, state=states[i])
            return backbone.final_norm(x)

        # Hybrid
        elif backbone.__class__.__name__ == "HybridLanguageModel":
            positions = torch.arange(0, T, device=device).unsqueeze(0)
            x = backbone.tok_embeddings(input_ids) + backbone.pos_embeddings(positions)
            x = backbone.drop(x)
            caches = [None] * backbone.n_layers
            for i, layer in enumerate(backbone.layers):
                ltype = backbone.layer_types[i]
                if ltype in ("bda", "delta"):
                    x, _, _ = layer(x, state=caches[i])
                else:
                    x, _, _ = layer(x, kv_cache=caches[i], start_pos=0)
            return backbone.final_norm(x)

        # Transformer
        elif backbone.__class__.__name__ == "TransformerLM":
            x = backbone.tok_embeddings(input_ids)
            x = backbone.drop(x)
            for layer in backbone.layers:
                x, _, _ = layer(x, kv_cache=None, start_pos=0)
            return backbone.norm(x)

        else:
            # Generic fallback: try to call internal method if exists
            raise RuntimeError(f"Unknown backbone {backbone.__class__.__name__}")

    def forward(
        self,
        input_ids: torch.Tensor,
        target_images: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            input_ids: (B,T) texto
            target_images: (B,C,H,W) opcional para loss
        Returns:
            image_pred: (B,C,H,W)
            loss: scalar or None
            z_pred: (B,D,H_g,W_g)
        """
        hidden = self._encode_text_hidden(input_ids)  # (B,T,d_model)
        z_pred = self.image_head(hidden)  # (B,D,H_g,W_g)
        image_pred = self.decoder(z_pred)  # (B,C,H,W)

        loss = None
        if target_images is not None:
            if self.igc.loss_type == "l1":
                loss = F.l1_loss(image_pred, target_images)
            else:
                loss = F.mse_loss(image_pred, target_images)

        return image_pred, loss, z_pred

    @torch.no_grad()
    def generate_image(
        self,
        prompt_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Genera imagen desde texto: prompt_ids (B,T) -> Image (B,C,H,W)"""
        self.eval()
        image_pred, _, _ = self.forward(prompt_ids)
        return image_pred

    # Alias for compatibility with pipeline expectations
    def generate(self, prompt_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.generate_image(prompt_ids, **kwargs)

    def infer(self, prompt: Union[str, torch.Tensor], **kwargs) -> Any:
        from bueorm.core.inference import InferencePipeline
        pipe = InferencePipeline(self)
        return pipe(prompt, **kwargs)
