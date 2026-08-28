"""
Bueorm Models - BDA Language Model (BDALanguageModel)
Recurrent associative memory language model with O(1) inference latency and optional MoE scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Union, Dict

from bueorm.config import BueormConfig
from bueorm.core.registry import register_model
from bueorm.moe.moe_layer import SparseMoELayer

from BDA.config import BDAConfig
from BDA.layers.bda_layer import BDALayer
from BDA.layers.block import RMSNorm, SwiGLU


class BDALMBlock(nn.Module):
    """
    BDA Transformer block with optional Mixture of Experts (MoE) FFN.
    """
    def __init__(self, config: BueormConfig, use_moe: bool = False):
        super().__init__()
        self.config = config
        self.use_moe = use_moe

        # BDA Layer
        bda_cfg = BDAConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_k=config.d_k,
            d_v=config.d_v,
            rank_r=config.rank_r,
            ema_lambda=config.ema_lambda,
            stability_margin=config.stability_margin,
            chunk_size=config.chunk_size,
            eps=config.norm_eps,
            bias=config.bias
        )
        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.bda = BDALayer(bda_cfg)
        self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)

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
            hidden_dim = int(2 * (config.d_model * config.mlp_ratio) / 3)
            self.ffn = SwiGLU(config.d_model, hidden_dim, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        # BDA Attention
        normed_x = self.norm1(x)
        bda_out, new_state = self.bda(normed_x, state=state)
        x = x + bda_out

        # FFN / MoE
        normed_ffn = self.norm2(x)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(normed_ffn)
        else:
            ffn_out = self.ffn(normed_ffn)
            aux_loss = torch.tensor(0.0, device=x.device)

        x = x + ffn_out
        return x, new_state, aux_loss

    def step(
        self,
        x_t: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        normed_x = self.norm1(x_t)
        bda_out, new_state = self.bda.step(normed_x, state=state)
        x_t = x_t + bda_out

        normed_ffn = self.norm2(x_t)
        if self.use_moe:
            ffn_out, _ = self.ffn(normed_ffn.unsqueeze(1))
            ffn_out = ffn_out.squeeze(1)
        else:
            ffn_out = self.ffn(normed_ffn)

        x_t = x_t + ffn_out
        return x_t, new_state


@register_model("bda")
@register_model("bda_lm")
@register_model("BDALanguageModel")
class BDALanguageModel(nn.Module):
    """
    BUEORM Delta Attention (BDA) Language Model.
    Supports pure BDA recurrent state memory, dense SwiGLU, and Mixture of Experts (MoE).
    """
    def __init__(self, config: Optional[BueormConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = BueormConfig(model_type="bda", **kwargs)
        self.config = config

        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.max_seq_len = config.max_seq_len

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Build layers with optional MoE
        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            is_moe = config.use_moe and ((i + 1) % config.moe_layer_interval == 0)
            self.layers.append(BDALMBlock(config, use_moe=is_moe))

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
        states: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
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

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            aux_coef = self.config.moe_config.aux_loss_coef if self.config.use_moe and self.config.moe_config else 0.0
            loss = ce_loss + aux_coef * total_aux_loss

        return logits, loss, new_states

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
        logits, _, states = self.forward(prompt_ids)
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

            new_states = []
            for i, layer in enumerate(self.layers):
                x_t, state_i = layer.step(x_t, state=states[i])
                new_states.append(state_i)
            states = new_states

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
