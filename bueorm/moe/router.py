"""
Bueorm MoE - Expert Routing & Load Balancing Engine
Implements Top-K Gating, Noisy Routers, and GShard / Switch Transformer Auxiliary Load Balancing Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TopKRouter(nn.Module):
    """
    Top-K Gating Router for Mixture of Experts (MoE).
    
    Routes tokens to top-k expert networks and computes auxiliary load-balancing loss:
        L_aux = E * sum_{i=1}^E (f_i * P_i)
    where:
        f_i = fraction of tokens routed to expert i
        P_i = average routing probability allocated to expert i
    """
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        jitter_noise: float = 0.01
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.jitter_noise = jitter_noise
        
        # Router projection W_g: d_model -> num_experts
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.gate.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input token representations of shape (batch_size, seq_len, d_model) or (N_tokens, d_model)
            
        Returns:
            topk_weights: Normalized routing weights for selected experts, shape (N_tokens, top_k)
            topk_indices: Indices of selected experts (0..E-1), shape (N_tokens, top_k)
            aux_loss: Scalar auxiliary load balancing loss for expert dispatch balance
        """
        orig_shape = x.shape
        x_flat = x.view(-1, self.d_model)  # (N, d_model)
        N = x_flat.shape[0]

        # 1. Compute gating logits
        logits = self.gate(x_flat)  # (N, num_experts)
        
        if self.training and self.jitter_noise > 0.0:
            # Multiplicative noise for uniform expert exploration
            noise = torch.empty_like(logits).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
            logits = logits * noise

        # 2. Softmax probabilities over all experts
        probs = F.softmax(logits, dim=-1)  # (N, num_experts)

        # 3. Select top-k experts per token
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)  # (N, top_k)

        # Re-normalize top-k weights so they sum to 1.0 per token
        topk_weights = topk_weights / torch.clamp(topk_weights.sum(dim=-1, keepdim=True), min=1e-8)

        # 4. Compute Auxiliary Load Balancing Loss
        if self.training:
            # P_i = average probability for expert i across batch tokens
            P_i = probs.mean(dim=0)  # (num_experts,)
            
            # f_i = fraction of tokens assigned to expert i as top-1
            top1_indices = topk_indices[:, 0]
            mask_1 = F.one_hot(top1_indices, num_classes=self.num_experts).float()
            f_i = mask_1.mean(dim=0)  # (num_experts,)
            
            # Loss = E * sum(f_i * P_i)
            aux_loss = self.num_experts * torch.sum(f_i * P_i)
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        return topk_weights, topk_indices, aux_loss
