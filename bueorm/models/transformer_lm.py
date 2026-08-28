"""
Bueorm Models - Transformer Language Model (TransformerLM)
Scalable causal transformer with FlashAttention/SDPA, GQA, RoPE, and native MoE scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Union, Dict, Any

from bueorm.config import BueormConfig
from bueorm.core.registry import register_model
from bueorm.moe.moe_layer import SparseMoELayer

from transformer.config import TransformerConfig
from transformer.attention.causal_attention import CausalSelfAttention
from transformer.attention.kv_cache import KVCache
from transformer.modules.norm import RMSNorm
from transformer.modules.ffn import SwiGLU


class TransformerLMBlock(nn.Module):
    """Transformer block with optional MoE."""
    def __init__(self, config: BueormConfig, use_moe: bool = False):
        super().__init__()
        self.config = config
        self.use_moe = use_moe

        tf_cfg = TransformerConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            mlp_ratio=config.mlp_ratio,
            norm_eps=config.norm_eps,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
            use_rope=config.use_rope,
            use_flash_attn=config.use_flash_attn,
            bias=config.bias
        )

        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttention(tf_cfg)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)

        if use_moe and config.moe_config is not None:
            self.ffn = SparseMoELayer(
                dim=config.d_model,
                num_experts=config.moe_config.num_experts,
                top_k=config.moe_config.top_k,
                mlp_ratio=config.mlp_ratio,
                jitter_noise=config.moe_config.router_jitter_noise,
                bias=config.bias
            )
        else:
            self.ffn = SwiGLU(config.d_model, mlp_ratio=config.mlp_ratio, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Union[KVCache, Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[Union[KVCache, Tuple[torch.Tensor, torch.Tensor]]], torch.Tensor]:
        # Attention
        normed_x = self.attn_norm(x)
        attn_out, new_kv_cache = self.attn(normed_x, kv_cache=kv_cache, start_pos=start_pos)
        x = x + attn_out

        # FFN / MoE
        normed_ffn = self.ffn_norm(x)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(normed_ffn)
        else:
            ffn_out = self.ffn(normed_ffn)
            aux_loss = torch.tensor(0.0, device=x.device)

        x = x + ffn_out
        return x, new_kv_cache, aux_loss


@register_model("transformer")
@register_model("transformer_lm")
@register_model("TransformerLM")
class TransformerLM(nn.Module):
    """
    Scalable Causal Transformer Language Model.
    Supports FlashAttention/SDPA, GQA, RoPE, and native MoE scaling.
    """
    def __init__(self, config: Optional[BueormConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = BueormConfig(model_type="transformer", **kwargs)
        self.config = config

        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.max_seq_len = config.max_seq_len

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            is_moe = config.use_moe and ((i + 1) % config.moe_layer_interval == 0)
            self.layers.append(TransformerLMBlock(config, use_moe=is_moe))

        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Optional[KVCache]]] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Optional[KVCache]]]:
        B, T = input_ids.shape
        device = input_ids.device

        x = self.tok_embeddings(input_ids)
        x = self.drop(x)

        if kv_caches is None:
            kv_caches = [None] * self.n_layers

        new_kv_caches = []
        total_aux_loss = torch.tensor(0.0, device=device)

        for i, layer in enumerate(self.layers):
            x, cache_i, aux_loss_i = layer(x, kv_cache=kv_caches[i], start_pos=start_pos)
            new_kv_caches.append(cache_i)
            total_aux_loss = total_aux_loss + aux_loss_i

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            aux_coef = self.config.moe_config.aux_loss_coef if self.config.use_moe and self.config.moe_config else 0.0
            loss = ce_loss + aux_coef * total_aux_loss

        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        self.eval()
        B, prompt_len = prompt_ids.shape
        device = prompt_ids.device

        kv_caches = [KVCache(max_seq_len=self.max_seq_len) for _ in range(self.n_layers)]

        logits, _, kv_caches = self.forward(prompt_ids, kv_caches=kv_caches, start_pos=0)
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
        curr_pos = prompt_len

        for _ in range(max_new_tokens - 1):
            if curr_pos >= self.max_seq_len:
                break

            logits, _, kv_caches = self.forward(curr_token, kv_caches=kv_caches, start_pos=curr_pos)
            step_logits = logits[:, -1, :]

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
            curr_pos += 1

        return torch.cat(generated, dim=1)

    def infer(self, prompt: Union[str, torch.Tensor], **kwargs) -> Any:
        from bueorm.core.inference import InferencePipeline
        pipe = InferencePipeline(self)
        return pipe(prompt, **kwargs)
