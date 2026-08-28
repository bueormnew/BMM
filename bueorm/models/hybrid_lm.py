"""
Bueorm Models - Hybrid Language Model (HybridLanguageModel)
Interleaves BDA linear recurrence blocks and FlashAttention blocks in arbitrary configurable ratios and patterns,
with optional Mixture of Experts (MoE) scaling.
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Union, Dict, Any

from bueorm.config import BueormConfig
from bueorm.core.registry import register_model
from bueorm.models.bda_lm import BDALMBlock
from bueorm.models.transformer_lm import TransformerLMBlock
from transformer.modules.norm import RMSNorm


def parse_hybrid_pattern(pattern: Union[str, List[str]], n_layers: int) -> List[str]:
    """
    Parses a hybrid specification into a list of layer types of length n_layers.
    
    Examples:
        - "4bda:1attn" -> ['bda', 'bda', 'bda', 'bda', 'attn', 'bda', ...]
        - "3:1" -> 3 BDA to 1 Attn
        - ['bda', 'attn'] -> alternating
    """
    if isinstance(pattern, list):
        if len(pattern) == n_layers:
            return pattern
        repeated = (pattern * ((n_layers // len(pattern)) + 1))[:n_layers]
        return repeated

    pattern_str = str(pattern).lower().strip()
    
    match = re.match(r"(\d+)\s*bda\s*:\s*(\d+)\s*attn", pattern_str)
    if not match:
        match = re.match(r"(\d+)\s*:\s*(\d+)", pattern_str)
        
    if match:
        n_bda = int(match.group(1))
        n_attn = int(match.group(2))
        cycle = (["bda"] * n_bda) + (["attn"] * n_attn)
        repeated = (cycle * ((n_layers // len(cycle)) + 1))[:n_layers]
        return repeated
    else:
        cycle = ["bda", "bda", "bda", "bda", "attn"]
        return (cycle * ((n_layers // len(cycle)) + 1))[:n_layers]


@register_model("hybrid")
@register_model("hybrid_lm")
@register_model("HybridLanguageModel")
class HybridLanguageModel(nn.Module):
    """
    Universal Hybrid Language Model combining BDA and FlashAttention in configurable ratios.
    """
    def __init__(self, config: Optional[BueormConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = BueormConfig(model_type="hybrid", **kwargs)
        self.config = config

        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.max_seq_len = config.max_seq_len

        self.layer_types = parse_hybrid_pattern(config.hybrid_pattern, config.n_layers)

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Build heterogeneous stack of blocks
        self.layers = nn.ModuleList()
        for i, ltype in enumerate(self.layer_types):
            is_moe = config.use_moe and ((i + 1) % config.moe_layer_interval == 0)
            if ltype in ("bda", "delta"):
                self.layers.append(BDALMBlock(config, use_moe=is_moe))
            else:
                self.layers.append(TransformerLMBlock(config, use_moe=is_moe))

        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embeddings.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        caches: Optional[List[Any]] = None,
        start_pos: int = 0
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

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            aux_coef = self.config.moe_config.aux_loss_coef if self.config.use_moe and self.config.moe_config else 0.0
            loss = ce_loss + aux_coef * total_aux_loss

        return logits, loss, new_caches

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

        # Warm up on prompt
        logits, _, caches = self.forward(prompt_ids, start_pos=0)
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
        pos = prompt_len

        for _ in range(max_new_tokens - 1):
            if pos >= self.max_seq_len:
                break

            pos_tensor = torch.tensor([[pos]], device=device).expand(B, 1)
            x = self.tok_embeddings(curr_token) + self.pos_embeddings(pos_tensor)
            x_t = x.squeeze(1)

            new_caches = []
            for i, layer in enumerate(self.layers):
                ltype = self.layer_types[i]
                if ltype in ("bda", "delta"):
                    x_t, cache_i = layer.step(x_t, state=caches[i])
                else:
                    # FlashAttention step
                    out_attn, cache_i, _ = layer(x_t.unsqueeze(1), kv_cache=caches[i], start_pos=pos)
                    x_t = out_attn.squeeze(1)
                new_caches.append(cache_i)
            caches = new_caches

            x_norm = self.final_norm(x_t)
            step_logits = self.lm_head(x_norm)

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

    def infer(self, prompt: Union[str, torch.Tensor], **kwargs) -> Any:
        from bueorm.core.inference import InferencePipeline
        pipe = InferencePipeline(self)
        return pipe(prompt, **kwargs)
