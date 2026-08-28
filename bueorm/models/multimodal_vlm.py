"""
Bueorm Models - Multimodal Vision-Language Model (BueormVLM)
Unifies TBV bidirectional vision backbone, 2D spatial feature projection,
and BDA/Transformer/Hybrid/MoE language reasoning into a cohesive VLM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Union, Dict, Any

from bueorm.config import BueormConfig
from bueorm.core.registry import register_model
from bueorm.models.tbv_vision import TBVVisionModel
from bueorm.models.hybrid_lm import HybridLanguageModel
from bueorm.models.bda_lm import BDALanguageModel
from bueorm.models.transformer_lm import TransformerLM
from TBV.modules.projector import TBVVisualProjector


@register_model("vlm")
@register_model("bueorm_vlm")
@register_model("BueormVLM")
class BueormVLM(nn.Module):
    """
    Multimodal Vision-Language Model (VLM).
    Connects TBV visual features to BDA/Hybrid/Transformer language modeling.
    """
    def __init__(self, config: Optional[BueormConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = BueormConfig.vlm_bda(**kwargs)
        self.config = config

        # 1. Vision Backbone (TBV)
        self.vision_backbone = TBVVisionModel(config)
        if config.vlm_config and config.vlm_config.freeze_vision:
            for p in self.vision_backbone.parameters():
                p.requires_grad = False

        # 2. Visual Projector (Spatial Grid Z -> Language Token Embedding Space)
        self.projector = TBVVisualProjector(
            in_dim=config.tbv_dim,
            out_dim=config.d_model,
            grid_size=config.image_size // config.patch_size,
            use_2d_pos_emb=True
        )

        # 3. Language Backbone (Hybrid BDA-Transformer or Pure BDA/Transformer)
        if config.model_type == "bda":
            self.language_model = BDALanguageModel(config)
        elif config.model_type == "transformer":
            self.language_model = TransformerLM(config)
        else:
            self.language_model = HybridLanguageModel(config)

    def extract_visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts visual token representations for language conditioning.
        Args:
            images: Tensor of shape (B, C, H, W)
        Returns:
            visual_tokens: (B, N_patches, d_model)
        """
        z = self.vision_backbone.encode(images)  # (B, D, H_g, W_g)
        visual_tokens = self.projector(z)        # (B, N_patches, d_model)
        return visual_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Any]]:
        B, T_text = input_ids.shape
        device = input_ids.device

        text_emb = self.language_model.tok_embeddings(input_ids)  # (B, T_text, d_model)

        if images is not None:
            vis_emb = self.extract_visual_tokens(images)  # (B, N_patches, d_model)
            combined_emb = torch.cat([vis_emb, text_emb], dim=1)  # (B, N_patches + T_text, d_model)
            N_patches = vis_emb.shape[1]
        else:
            combined_emb = text_emb
            N_patches = 0

        T_total = combined_emb.shape[1]
        positions = torch.arange(0, T_total, device=device).unsqueeze(0)
        pos_emb = self.language_model.pos_embeddings(positions)
        
        x = self.language_model.drop(combined_emb + pos_emb)

        caches = [None] * self.language_model.n_layers
        new_caches = []
        total_aux_loss = torch.tensor(0.0, device=device)

        for i, layer in enumerate(self.language_model.layers):
            ltype = getattr(self.language_model, "layer_types", ["bda"] * self.language_model.n_layers)[i]
            if ltype in ("bda", "delta"):
                x, cache_i, aux_loss_i = layer(x, state=caches[i])
            else:
                x, cache_i, aux_loss_i = layer(x, kv_cache=caches[i], start_pos=0)
            new_caches.append(cache_i)
            total_aux_loss = total_aux_loss + aux_loss_i

        x = self.language_model.final_norm(x)
        logits = self.language_model.lm_head(x)

        loss = None
        if targets is not None:
            text_logits = logits[:, N_patches:, :].contiguous()
            ce_loss = F.cross_entropy(text_logits.view(-1, self.config.vocab_size), targets.view(-1))
            aux_coef = self.config.moe_config.aux_loss_coef if self.config.use_moe and self.config.moe_config else 0.0
            loss = ce_loss + aux_coef * total_aux_loss

        return logits, loss, new_caches

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        self.eval()
        B, prompt_len = prompt_ids.shape
        device = prompt_ids.device

        logits, _, caches = self.forward(input_ids=prompt_ids, images=images)
        next_token_logits = logits[:, -1, :]

        if temperature > 0.0:
            next_token_logits = next_token_logits / temperature
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = [prompt_ids, next_token]
        curr_token = next_token

        N_patches = (self.config.image_size // self.config.patch_size) ** 2 if images is not None else 0
        pos = N_patches + prompt_len

        for _ in range(max_new_tokens - 1):
            if pos >= self.config.max_seq_len:
                break

            pos_tensor = torch.tensor([[pos]], device=device).expand(B, 1)
            x = self.language_model.tok_embeddings(curr_token) + self.language_model.pos_embeddings(pos_tensor)
            x_t = x.squeeze(1)

            new_caches = []
            for i, layer in enumerate(self.language_model.layers):
                ltype = getattr(self.language_model, "layer_types", ["bda"] * self.language_model.n_layers)[i]
                if ltype in ("bda", "delta"):
                    x_t, cache_i = layer.step(x_t, state=caches[i])
                else:
                    out_attn, cache_i, _ = layer(x_t.unsqueeze(1), kv_cache=caches[i], start_pos=pos)
                    x_t = out_attn.squeeze(1)
                new_caches.append(cache_i)
            caches = new_caches

            x_norm = self.language_model.final_norm(x_t)
            step_logits = self.language_model.lm_head(x_norm)

            if temperature > 0.0:
                step_logits = step_logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(step_logits, min(top_k, step_logits.size(-1)))
                    step_logits[step_logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(step_logits, dim=-1)
                curr_token = torch.multinomial(probs, num_samples=1)
            else:
                curr_token = torch.argmax(step_logits, dim=-1, keepdim=True)

            generated.append(curr_token)
            pos += 1

        return torch.cat(generated, dim=1)

    def infer(self, prompt: Union[str, torch.Tensor], images: Optional[torch.Tensor] = None, **kwargs) -> Any:
        from bueorm.core.inference import InferencePipeline
        pipe = InferencePipeline(self)
        return pipe(inputs=prompt, images=images, **kwargs)
