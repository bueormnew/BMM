"""
Bueorm Models - Generative VLM (Any-to-Any)
Extiende BueormVLM para soportar:
- image+text -> texto (heredado)
- texto -> imagen (nuevo)
- image+text -> imagen (nuevo, ej. edición condicionada)
Mantiene compatibilidad 100% con VLM existente cuando enable_image_gen=False no se usa.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Any, Union

from bueorm.config import BueormConfig
from bueorm.core.registry import register_model
from bueorm.models.multimodal_vlm import BueormVLM
from bueorm.generation.image_head import TextToLatentHead, LatentToImageDecoder


@register_model("vlm_with_image")
@register_model("vlm_gen")
@register_model("generative_vlm")
@register_model("any_to_any")
@register_model("GenerativeVLM")
class GenerativeVLM(BueormVLM):
    """
    VLM generativo Any-to-Any.

    Arquitectura:
      Vision (TBV encode) -> projector -> tokens visuales \
      Texto -> tok_embeddings --------------------------------> concat -> language backbone (BDA/Hybrid/Transformer + MoE) -> hidden -> lm_head (texto)
                                                                                              |
                                                                                              +-> TextToLatentHead -> Z -> TBV decode -> imagen

    Versátil: backbone híbrido/BDA/Transformer con MoE, cambia vía config.model_type / hybrid_pattern / use_moe.

    Uso:
        cfg = BueormConfig.vlm_with_image_gen(image_size=64, tbv_dim=32, d_model=128, hybrid_pattern="3bda:1attn", use_moe=True, moe_config=...)
        model = create_model("vlm_with_image", config=cfg)
        # texto condicionado por imagen
        logits, loss, _ = model(input_ids, images=imgs, targets=targets)
        # generar texto
        txt = model.generate(prompt_ids, images=imgs)
        # generar imagen desde texto (con opcional contexto visual)
        img = model.generate_image(prompt_ids, images=imgs)
        # forward conjunto
        logits, loss, caches, img_pred = model.forward_with_image(input_ids, images=imgs, targets=targets, target_images=target_imgs)
    """
    def __init__(self, config: Optional[BueormConfig] = None, **kwargs):
        if config is None:
            config = BueormConfig.vlm_with_image_gen(**kwargs)
        if config.image_gen_config is None or not config.image_gen_config.enabled:
            from bueorm.config import ImageGenConfig
            config.image_gen_config = ImageGenConfig(enabled=True, backbone=config.model_type)
            config.enable_image_gen = True
        # Ensure is_multimodal
        config.is_multimodal = True
        super().__init__(config)
        # Image generation head on top of language hidden
        igc = config.image_gen_config
        grid_size = igc.image_size // igc.patch_size
        assert igc.image_size % igc.patch_size == 0
        self.igc = igc
        self.grid_size = grid_size
        self.image_head = TextToLatentHead(
            d_model=config.d_model,
            tbv_dim=igc.tbv_dim,
            grid_size=grid_size,
            head_hidden_dim=igc.head_hidden_dim,
            pooling=igc.pooling,
            dropout=config.dropout,
        )
        self.image_decoder = LatentToImageDecoder(igc, trainable=igc.train_decoder)

    def _get_norm(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.language_model, "final_norm"):
            return self.language_model.final_norm(x)
        elif hasattr(self.language_model, "norm"):
            return self.language_model.norm(x)
        return x

    def _hidden_from_combined(self, input_ids: torch.Tensor, images: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Helper to get language hidden before lm_head, including visual tokens if images provided."""
        B, T_text = input_ids.shape
        device = input_ids.device
        text_emb = self.language_model.tok_embeddings(input_ids)
        if images is not None:
            vis_emb = self.extract_visual_tokens(images)  # (B,N_patches,d_model)
            combined_emb = torch.cat([vis_emb, text_emb], dim=1)
        else:
            combined_emb = text_emb

        # Pos embeddings handling: BDA/Hybrid have pos, Transformer does not
        if hasattr(self.language_model, "pos_embeddings"):
            T_total = combined_emb.shape[1]
            positions = torch.arange(0, T_total, device=device).unsqueeze(0)
            pos_emb = self.language_model.pos_embeddings(positions)
            x = self.language_model.drop(combined_emb + pos_emb)
        else:
            # TransformerLM: no positional addition
            x = self.language_model.drop(combined_emb)

        lm = self.language_model
        lm_type = lm.__class__.__name__
        if lm_type == "BDALanguageModel":
            states = [None] * lm.n_layers
            for i, layer in enumerate(lm.layers):
                x, _, _ = layer(x, state=states[i])
        elif lm_type == "TransformerLM":
            for layer in lm.layers:
                x, _, _ = layer(x, kv_cache=None, start_pos=0)
        elif lm_type == "HybridLanguageModel":
            caches = [None] * lm.n_layers
            for i, layer in enumerate(lm.layers):
                ltype = lm.layer_types[i]
                if ltype in ("bda", "delta"):
                    x, _, _ = layer(x, state=caches[i])
                else:
                    x, _, _ = layer(x, kv_cache=caches[i], start_pos=0)
        else:
            # Generic fallback using layer_types if present
            caches = [None] * lm.n_layers
            for i, layer in enumerate(lm.layers):
                ltype = getattr(lm, "layer_types", ["bda"] * lm.n_layers)[i]
                if ltype in ("bda", "delta"):
                    x, _, _ = layer(x, state=caches[i])
                else:
                    x, _, _ = layer(x, kv_cache=caches[i], start_pos=0)
        hidden = self._get_norm(x)
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        target_images: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Any]]:
        """
        Forward compatible con BueormVLM.
        Si target_images se provee, añade loss de imagen.
        Returns: (logits, loss, caches) — loss incluye texto + aux MoE + imagen si aplica.
        Para obtener image_pred, usar forward_with_image().
        """
        B, T_text = input_ids.shape
        device = input_ids.device
        text_emb = self.language_model.tok_embeddings(input_ids)
        if images is not None:
            vis_emb = self.extract_visual_tokens(images)
            combined_emb = torch.cat([vis_emb, text_emb], dim=1)
            N_patches = vis_emb.shape[1]
        else:
            combined_emb = text_emb
            N_patches = 0

        if hasattr(self.language_model, "pos_embeddings"):
            T_total = combined_emb.shape[1]
            positions = torch.arange(0, T_total, device=device).unsqueeze(0)
            pos_emb = self.language_model.pos_embeddings(positions)
            x = self.language_model.drop(combined_emb + pos_emb)
        else:
            x = self.language_model.drop(combined_emb)

        lm = self.language_model
        lm_type = lm.__class__.__name__
        caches = [None] * lm.n_layers
        new_caches = []
        total_aux_loss = torch.tensor(0.0, device=device)
        if lm_type == "BDALanguageModel":
            for i, layer in enumerate(lm.layers):
                x, cache_i, aux_loss_i = layer(x, state=caches[i])
                new_caches.append(cache_i)
                total_aux_loss = total_aux_loss + aux_loss_i
        elif lm_type == "TransformerLM":
            for i, layer in enumerate(lm.layers):
                x, cache_i, aux_loss_i = layer(x, kv_cache=caches[i], start_pos=0)
                new_caches.append(cache_i)
                total_aux_loss = total_aux_loss + aux_loss_i
        elif lm_type == "HybridLanguageModel":
            for i, layer in enumerate(lm.layers):
                ltype = lm.layer_types[i]
                if ltype in ("bda", "delta"):
                    x, cache_i, aux_loss_i = layer(x, state=caches[i])
                else:
                    x, cache_i, aux_loss_i = layer(x, kv_cache=caches[i], start_pos=0)
                new_caches.append(cache_i)
                total_aux_loss = total_aux_loss + aux_loss_i
        else:
            for i, layer in enumerate(lm.layers):
                ltype = getattr(lm, "layer_types", ["bda"] * lm.n_layers)[i]
                if ltype in ("bda", "delta"):
                    x, cache_i, aux_loss_i = layer(x, state=caches[i])
                else:
                    x, cache_i, aux_loss_i = layer(x, kv_cache=caches[i], start_pos=0)
                new_caches.append(cache_i)
                total_aux_loss = total_aux_loss + aux_loss_i
        hidden = self._get_norm(x)
        logits = self.language_model.lm_head(hidden)

        # Text loss
        loss = None
        if targets is not None:
            text_logits = logits[:, N_patches:, :].contiguous()
            ce_loss = F.cross_entropy(text_logits.view(-1, self.config.vocab_size), targets.view(-1))
            aux_coef = self.config.moe_config.aux_loss_coef if self.config.use_moe and self.config.moe_config else 0.0
            loss = ce_loss + aux_coef * total_aux_loss

        # Image loss branch
        if target_images is not None:
            # Use hidden's text part only for pooling? Use full hidden for richer context
            # For image generation we pool over the hidden sequence corresponding to text
            # Take hidden's text segment if multimodal, else full
            if N_patches > 0:
                text_hidden = hidden[:, N_patches:, :]
            else:
                text_hidden = hidden
            z_pred = self.image_head(text_hidden)
            image_pred = self.image_decoder(z_pred)
            img_loss = F.l1_loss(image_pred, target_images) if self.igc.loss_type == "l1" else F.mse_loss(image_pred, target_images)
            img_loss = img_loss * self.igc.loss_weight
            loss = img_loss if loss is None else loss + img_loss

        return logits, loss, new_caches

    def forward_with_image(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        target_images: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Any], Optional[torch.Tensor]]:
        """Like forward but also returns image_pred for training inspection."""
        logits, loss, caches = self.forward(input_ids, images=images, targets=targets, target_images=target_images)
        image_pred = None
        if target_images is not None or True:  # always compute pred for inspection if requested
            # need hidden recompute for pred; reuse forward internal hidden path would be more efficient,
            # for now recompute hidden to produce pred
            hidden = self._hidden_from_combined(input_ids, images)
            # Extract text hidden slice for head
            if images is not None:
                N_patches = (self.config.image_size // self.config.patch_size) ** 2
                text_hidden = hidden[:, N_patches:, :]
            else:
                text_hidden = hidden
            z_pred = self.image_head(text_hidden)
            image_pred = self.image_decoder(z_pred)
        return logits, loss, caches, image_pred

    @torch.no_grad()
    def generate_image(
        self,
        prompt_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Genera imagen desde prompt texto, opcionalmente condicionada por imagen contexto."""
        self.eval()
        hidden = self._hidden_from_combined(prompt_ids, images)
        # For image generation, pool over text hidden part only
        if images is not None:
            N_patches = (self.config.image_size // self.config.patch_size) ** 2
            text_hidden = hidden[:, N_patches:, :]
        else:
            text_hidden = hidden
        z_pred = self.image_head(text_hidden)
        return self.image_decoder(z_pred)

    # generate() inherited from BueormVLM for text generation
