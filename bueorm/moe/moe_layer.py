"""
Bueorm MoE - Sparse Mixture of Experts Layer
Routes tokens dynamically to top-k expert networks with parallel dispatching and gradient tracking.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List

from bueorm.moe.router import TopKRouter
from bueorm.moe.experts import SwiGLUExpert, MLPExpert


class SparseMoELayer(nn.Module):
    """
    Sparse Mixture of Experts (MoE) Layer.
    
    Replaces a monolithic feed-forward layer with E expert networks,
    activating only top-k experts per token for sub-linear compute scaling.
    """
    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        mlp_ratio: float = 4.0,
        expert_type: str = "swiglu",
        jitter_noise: float = 0.01,
        bias: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Router
        self.router = TopKRouter(
            d_model=dim,
            num_experts=num_experts,
            top_k=top_k,
            jitter_noise=jitter_noise
        )
        
        # Expert modules
        hidden_dim = int(2 * (dim * mlp_ratio) / 3) if expert_type == "swiglu" else int(dim * mlp_ratio)
        
        self.experts = nn.ModuleList([
            SwiGLUExpert(dim, hidden_dim, bias=bias) if expert_type == "swiglu"
            else MLPExpert(dim, hidden_dim, bias=bias)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            out: Expert output of shape (batch_size, seq_len, dim)
            aux_loss: Auxiliary load-balancing loss for router training
        """
        orig_shape = x.shape
        x_flat = x.view(-1, self.dim)  # (N_tokens, dim)
        N = x_flat.shape[0]

        # Route tokens: topk_weights (N, top_k), topk_indices (N, top_k)
        topk_weights, topk_indices, aux_loss = self.router(x_flat)

        # Output accumulator
        out_flat = torch.zeros_like(x_flat)

        # Dispatch tokens to respective experts
        for expert_idx, expert in enumerate(self.experts):
            # Mask of tokens routed to this expert
            token_mask = (topk_indices == expert_idx)  # (N, top_k)
            if not token_mask.any():
                continue

            # Token row indices and rank positions
            row_indices, rank_indices = torch.where(token_mask)
            tokens_for_expert = x_flat[row_indices]  # (M, dim)
            
            # Forward through expert
            expert_out = expert(tokens_for_expert)  # (M, dim)
            
            # Scale by routing weights
            weights = topk_weights[row_indices, rank_indices].unsqueeze(-1)  # (M, 1)
            weighted_out = expert_out * weights
            
            # Accumulate into output
            out_flat.index_add_(0, row_indices, weighted_out)

        out = out_flat.view(orig_shape)
        return out, aux_loss
