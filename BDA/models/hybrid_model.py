"""
BUEORM Delta Attention (BDA) - Hybrid Language Model
Spec v2: Section 3.1, 6.4
Hybrid Language Model interleaving BDA memory blocks with Causal Full Attention blocks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Union, Dict

from BDA.config import BDAConfig
from BDA.layers.block import RMSNorm, HybridBlock


class BDAHybridModel(nn.Module):
    """
    BUEORM Delta Attention Hybrid Language Model.
    
    Combines BDA layers for O(1) state memory inference and linear compute with
    interleaved Causal Full Attention layers for exact long-range recall.
    """
    def __init__(self, config: Optional[BDAConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = BDAConfig(**kwargs)
        self.config = config
        
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.max_seq_len = config.max_seq_len
        
        # Token embedding
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # Build hybrid layers
        self.blocks = nn.ModuleList()
        for layer_idx in range(config.n_layers):
            # Interleave Full Attention every `hybrid_interval` layers (1-indexed check)
            is_full_attn = (config.hybrid_interval > 0) and ((layer_idx + 1) % config.hybrid_interval == 0)
            self.blocks.append(HybridBlock(config, is_full_attention=is_full_attn))
            
        self.final_norm = RMSNorm(config.d_model, eps=config.eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying (optional standard Transformer practice)
        self.lm_head.weight = self.tok_embeddings.weight
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embeddings.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass for training and sequence evaluation.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            targets: Optional ground truth labels of shape (batch_size, seq_len)
            caches: Optional list of layer states/caches
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: Cross entropy loss if targets provided, else None
            new_caches: Updated layer caches
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        positions = torch.arange(0, T, device=device).unsqueeze(0)
        x = self.tok_embeddings(input_ids) + self.pos_embeddings(positions)
        x = self.drop(x)
        
        if caches is None:
            caches = [None] * self.n_layers
            
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, cache_i = block(x, cache=caches[i])
            new_caches.append(cache_i)
            
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            
        return logits, loss, new_caches

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressive generation using streaming step-by-step state caching.
        
        Args:
            prompt_ids: Prompt token IDs (batch_size, prompt_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Optional top-k filtering
            
        Returns:
            generated_ids: Tensor of shape (batch_size, prompt_len + max_new_tokens)
        """
        self.eval()
        B, prompt_len = prompt_ids.shape
        device = prompt_ids.device
        
        # 1. Warm up states on the prompt sequence
        logits, _, caches = self.forward(prompt_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
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
            for i, block in enumerate(self.blocks):
                x_t, cache_i = block.step(x_t, cache=caches[i])
                new_caches.append(cache_i)
            caches = new_caches
            
            x_norm = self.final_norm(x_t)
            logits_step = self.lm_head(x_norm)  # (B, vocab_size)
            
            if temperature > 0.0:
                logits_step = logits_step / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits_step, min(top_k, logits_step.size(-1)))
                    logits_step[logits_step < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits_step, dim=-1)
                curr_token = torch.multinomial(probs, num_samples=1)
            else:
                curr_token = torch.argmax(logits_step, dim=-1, keepdim=True)
                
            generated.append(curr_token)
            pos += 1
            
        return torch.cat(generated, dim=1)
