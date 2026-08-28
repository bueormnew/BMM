"""
Transformer Models - Causal Transformer Language Model (CausalTransformerLM)
Complete modern decoder-only language model with FlashAttention/GQA, RoPE, KV-cache, and generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Union

from transformer.config import TransformerConfig
from transformer.attention.kv_cache import KVCache
from transformer.modules.norm import RMSNorm
from transformer.modules.block import TransformerBlock


class CausalTransformerLM(nn.Module):
    """
    Scalable Causal Transformer Language Model.
    """
    def __init__(self, config: Optional[TransformerConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = TransformerConfig(**kwargs)
        self.config = config

        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.max_seq_len = config.max_seq_len

        # Token embedding
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm & LM head
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
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
        """
        Args:
            input_ids: (batch_size, seq_len)
            targets: Optional (batch_size, seq_len) ground truth tokens
            kv_caches: Optional list of KVCache objects per layer
            start_pos: Starting position for RoPE
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: Cross entropy loss if targets provided
            new_kv_caches: Updated KV cache list
        """
        B, T = input_ids.shape
        x = self.tok_embeddings(input_ids)
        x = self.drop(x)

        if kv_caches is None:
            kv_caches = [None] * self.n_layers

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            x, cache_i = layer(x, kv_cache=kv_caches[i], start_pos=start_pos)
            new_kv_caches.append(cache_i)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

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
        """
        Fast autoregressive generation with KV-cache.
        
        Args:
            prompt_ids: (batch_size, prompt_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering threshold
            top_p: Nucleus filtering threshold
            
        Returns:
            generated_ids: (batch_size, prompt_len + max_new_tokens)
        """
        self.eval()
        B, prompt_len = prompt_ids.shape
        device = prompt_ids.device

        # Initialize KV caches for each layer
        kv_caches = [KVCache(max_seq_len=self.max_seq_len) for _ in range(self.n_layers)]

        # 1. Prefill prompt
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

        # 2. Step by step autoregressive decoding
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
