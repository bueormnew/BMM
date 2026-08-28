"""
Bueorm Models - Generative Language Models (Puntos 2-3)
Modelos de texto nativos con generación de imagen opcional.

- BDAWithImageGen: BDA + TextToLatentHead + TBV decoder
- TransformerWithImageGen: Transformer + head + decoder
- HybridWithImageGen: Hybrid + head + decoder

Todos soportan MoE nativo (config.use_moe), versátiles por diseño.
Opt-in vía config.enable_image_gen=True; si False, usar clases base puras sin daño.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Any

from bueorm.config import BueormConfig
from bueorm.core.registry import register_model
from bueorm.generation.image_head import TextToLatentHead, LatentToImageDecoder

from bueorm.models.bda_lm import BDALanguageModel
from bueorm.models.transformer_lm import TransformerLM
from bueorm.models.hybrid_lm import HybridLanguageModel, parse_hybrid_pattern
from transformer.modules.norm import RMSNorm
from BDA.config import BDAConfig
from BDA.layers.bda_layer import BDALayer
from BDA.layers.block import SwiGLU
from transformer.config import TransformerConfig
from transformer.attention.causal_attention import CausalSelfAttention
from transformer.attention.kv_cache import KVCache
from transformer.modules.ffn import SwiGLU as TransformerSwiGLU
from bueorm.moe.moe_layer import SparseMoELayer


def _build_image_components(config: BueormConfig):
    igc = config.image_gen_config
    assert igc is not None and igc.enabled, "ImageGenConfig must be enabled"
    grid_size = igc.image_size // igc.patch_size
    assert igc.image_size % igc.patch_size == 0
    head = TextToLatentHead(
        d_model=config.d_model,
        tbv_dim=igc.tbv_dim,
        grid_size=grid_size,
        head_hidden_dim=igc.head_hidden_dim,
        pooling=igc.pooling,
        dropout=config.dropout,
    )
    decoder = LatentToImageDecoder(igc, trainable=igc.train_decoder)
    return head, decoder, grid_size


# ---------------------------------------------------------------------------
# BDA with Image Generation
# ---------------------------------------------------------------------------

@register_model("bda_with_image")
@register_model("bda_gen")
@register_model("BDAWithImageGen")
class BDAWithImageGen(BDALanguageModel):
    """
    BDA Language Model + TBV Image Generation.
    Usa BDA O(1) memory + MoE opcional + head texto->Z.
    """
    def __init__(self, config: Optional[BueormConfig] = None, **kwargs):
        if config is None:
            config = BueormConfig.bda_with_image_gen(**kwargs)
        if config.image_gen_config is None or not config.image_gen_config.enabled:
            from bueorm.config import ImageGenConfig
            config.image_gen_config = ImageGenConfig(enabled=True, backbone="bda")
            config.enable_image_gen = True
        super().__init__(config)
        self.image_head, self.image_decoder, self.grid_size = _build_image_components(config)
        self.igc = config.image_gen_config

    def _hidden_from_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Helper: input_ids -> hidden (B,T,d_model) before lm_head."""
        B, T = input_ids.shape
        device = input_ids.device
        positions = torch.arange(0, T, device=device).unsqueeze(0)
        x = self.tok_embeddings(input_ids) + self.pos_embeddings(positions)
        x = self.drop(x)
        states = [None] * self.n_layers
        for i, layer in enumerate(self.layers):
            x, _, _ = layer(x, state=states[i])
        hidden = self.final_norm(x)
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_images: Optional[torch.Tensor] = None,
        states: Optional[List[Any]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Any]]:
        # Base BDA forward with hidden retention
        B, T = input_ids.shape
        device = input_ids.device
        positions = torch.arange(0, T, device=device).unsqueeze(0)
        x = self.tok_embeddings(input_ids) + self.pos_embeddings(positions)
        x = self.drop(x)
        if states is None:
            states = [None] * self.n_layers
        new_states = []
        total_aux_loss = torch.tensor(0.0, device=device)
        for i, layer in enumerate(self.layers):
            x, state_i, aux_loss_i = layer(x, state=states[i])
            new_states.append(state_i)
            total_aux_loss = total_aux_loss + aux_loss_i
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)

        # Text loss
        loss = None
        ce_loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
            aux_coef = self.config.moe_config.aux_loss_coef if self.config.use_moe and self.config.moe_config else 0.0
            loss = ce_loss + aux_coef * total_aux_loss
        else:
            if self.config.use_moe and self.config.moe_config:
                loss = total_aux_loss * self.config.moe_config.aux_loss_coef if total_aux_loss.requires_grad else None
                # If no targets but aux loss exists, still may need to propagate? Keep None unless aux
                if loss is not None and loss.item() == 0.0:
                    loss = None

        # Image loss branch (opt-in)
        if target_images is not None:
            z_pred = self.image_head(hidden)  # (B,D,H_g,W_g)
            image_pred = self.image_decoder(z_pred)
            if self.igc.loss_type == "l1":
                img_loss = F.l1_loss(image_pred, target_images)
            else:
                img_loss = F.mse_loss(image_pred, target_images)
            img_loss = img_loss * self.igc.loss_weight
            if loss is None:
                loss = img_loss
            else:
                loss = loss + img_loss

        return logits, loss, new_states

    @torch.no_grad()
    def generate_image(self, prompt_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        self.eval()
        hidden = self._hidden_from_ids(prompt_ids)
        z_pred = self.image_head(hidden)
        image = self.image_decoder(z_pred)
        return image

    # generate() inherited from BDALanguageModel for text generation
    # generate_image() added for text->image


# ---------------------------------------------------------------------------
# Transformer with Image Generation
# ---------------------------------------------------------------------------

@register_model("transformer_with_image")
@register_model("transformer_gen")
@register_model("TransformerWithImageGen")
class TransformerWithImageGen(TransformerLM):
    def __init__(self, config: Optional[BueormConfig] = None, **kwargs):
        if config is None:
            config = BueormConfig.transformer_with_image_gen(**kwargs)
        if config.image_gen_config is None or not config.image_gen_config.enabled:
            from bueorm.config import ImageGenConfig
            config.image_gen_config = ImageGenConfig(enabled=True, backbone="transformer")
            config.enable_image_gen = True
        super().__init__(config)
        self.image_head, self.image_decoder, self.grid_size = _build_image_components(config)
        self.igc = config.image_gen_config

    def _hidden_from_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_embeddings(input_ids)
        x = self.drop(x)
        for layer in self.layers:
            x, _, _ = layer(x, kv_cache=None, start_pos=0)
        return self.norm(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_images: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Any]] = None,
        start_pos: int = 0,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Any]]:
        B, T = input_ids.shape
        device = input_ids.device
        x = self.tok_embeddings(input_ids)
        x = self.drop(x)
        if kv_caches is None:
            kv_caches = [None] * self.n_layers
        new_caches = []
        total_aux_loss = torch.tensor(0.0, device=device)
        for i, layer in enumerate(self.layers):
            x, cache_i, aux_loss_i = layer(x, kv_cache=kv_caches[i], start_pos=start_pos)
            new_caches.append(cache_i)
            total_aux_loss = total_aux_loss + aux_loss_i
        hidden = self.norm(x)
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
            aux_coef = self.config.moe_config.aux_loss_coef if self.config.use_moe and self.config.moe_config else 0.0
            loss = ce_loss + aux_coef * total_aux_loss

        if target_images is not None:
            z_pred = self.image_head(hidden)
            image_pred = self.image_decoder(z_pred)
            img_loss = F.l1_loss(image_pred, target_images) if self.igc.loss_type == "l1" else F.mse_loss(image_pred, target_images)
            img_loss = img_loss * self.igc.loss_weight
            loss = img_loss if loss is None else loss + img_loss

        return logits, loss, new_caches

    @torch.no_grad()
    def generate_image(self, prompt_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        self.eval()
        hidden = self._hidden_from_ids(prompt_ids)
        z_pred = self.image_head(hidden)
        return self.image_decoder(z_pred)


# ---------------------------------------------------------------------------
# Hybrid with Image Generation
# ---------------------------------------------------------------------------

@register_model("hybrid_with_image")
@register_model("hybrid_gen")
@register_model("HybridWithImageGen")
class HybridWithImageGen(HybridLanguageModel):
    def __init__(self, config: Optional[BueormConfig] = None, **kwargs):
        if config is None:
            config = BueormConfig.hybrid_with_image_gen(**kwargs)
        if config.image_gen_config is None or not config.image_gen_config.enabled:
            from bueorm.config import ImageGenConfig
            config.image_gen_config = ImageGenConfig(enabled=True, backbone="hybrid")
            config.enable_image_gen = True
        super().__init__(config)
        self.image_head, self.image_decoder, self.grid_size = _build_image_components(config)
        self.igc = config.image_gen_config

    def _hidden_from_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
        positions = torch.arange(0, T, device=device).unsqueeze(0)
        x = self.tok_embeddings(input_ids) + self.pos_embeddings(positions)
        x = self.drop(x)
        caches = [None] * self.n_layers
        for i, layer in enumerate(self.layers):
            ltype = self.layer_types[i]
            if ltype in ("bda", "delta"):
                x, _, _ = layer(x, state=caches[i])
            else:
                x, _, _ = layer(x, kv_cache=caches[i], start_pos=0)
        return self.final_norm(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_images: Optional[torch.Tensor] = None,
        caches: Optional[List[Any]] = None,
        start_pos: int = 0,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Any]]:
        B, T = input_ids.shape
        device = input_ids.device
        positions = torch.arange(0, T, device=device).unsqueeze(0)
        x = self.tok_embeddings(input_ids) + self.pos_embeddings(positions)
        x = self.drop(x)
        if caches is None:
            caches = [None] * self.n_layers
        new_caches = []
        total_aux_loss = torch.tensor(0.0, device=device)
        for i, layer in enumerate(self.layers):
            ltype = self.layer_types[i]
            if ltype in ("bda", "delta"):
                x, cache_i, aux_loss_i = layer(x, state=caches[i])
            else:
                x, cache_i, aux_loss_i = layer(x, kv_cache=caches[i], start_pos=start_pos)
            new_caches.append(cache_i)
            total_aux_loss = total_aux_loss + aux_loss_i
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
            aux_coef = self.config.moe_config.aux_loss_coef if self.config.use_moe and self.config.moe_config else 0.0
            loss = ce_loss + aux_coef * total_aux_loss

        if target_images is not None:
            z_pred = self.image_head(hidden)
            image_pred = self.image_decoder(z_pred)
            img_loss = F.l1_loss(image_pred, target_images) if self.igc.loss_type == "l1" else F.mse_loss(image_pred, target_images)
            img_loss = img_loss * self.igc.loss_weight
            loss = img_loss if loss is None else loss + img_loss

        return logits, loss, new_caches

    @torch.no_grad()
    def generate_image(self, prompt_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        self.eval()
        hidden = self._hidden_from_ids(prompt_ids)
        z_pred = self.image_head(hidden)
        return self.image_decoder(z_pred)
