"""
BUEORM Delta Attention (BDA) - Stability Projection (SP)
Spec v2: Section 5.3, 5.5, 7.7.5, 7.7.6
Guaranteed spectral norm operator bound: ||T_t||_2 <= 1 via runtime projection.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class StabilityProjection(nn.Module):
    """
    Stability Projection (SP).
    
    Spec Section 5.3 & 5.5:
    Enforces the proven operator norm condition:
        ||T_t||_2 <= max_i(alpha_{t,i}) + beta_t * ||k_tilde_t|| * ||k_hat_t|| <= 1
    By projecting beta_t on each step:
        alpha_max_t = max_i(alpha_{t,i})
        budget_t = clip(margin - alpha_max_t, min=0.0)
        denom_t = ||k_tilde_t|| * ||k_hat_t|| + eps
        beta_max_t = budget_t / denom_t
        beta_t^{SP} = min(beta_t^{ASN}, beta_max_t)
        
    Gradient Behavior (Spec 7.7.5):
        Differentiable min clipping ensures gradient flows to beta when not clipped,
        and flows to alpha, k_hat, k_tilde through beta_max when clipped.
    """
    def __init__(self, margin: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.margin = margin
        self.eps = eps

    def forward(
        self,
        alpha_t: torch.Tensor,
        beta_asn_t: torch.Tensor,
        k_hat_t: torch.Tensor,
        k_tilde_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Projects beta_t for a single step or batch/sequence of steps.
        
        Args:
            alpha_t: Forget gate vector in (0, 1), shape (..., d_k)
            beta_asn_t: Step size from ASN, shape (...)
            k_hat_t: Write key, shape (..., d_k)
            k_tilde_t: Erase key, shape (..., d_k)
            
        Returns:
            beta_sp: Projected step size, shape (...)
            beta_max: Theoretical maximum allowable beta for ||T_t||_2 <= 1, shape (...)
            alpha_max: Maximum forget factor across channels, shape (...)
        """
        # Compute in float32 for maximum numerical precision (Spec 7.7.6)
        alpha_f = alpha_t.float()
        k_hat_f = k_hat_t.float()
        k_tilde_f = k_tilde_t.float()
        beta_asn_f = beta_asn_t.float()
        
        # alpha_max = max_i(alpha_{t,i})
        alpha_max = torch.amax(alpha_f, dim=-1)  # shape (...)
        
        # budget = max(margin - alpha_max, 0.0)
        budget = torch.clamp(self.margin - alpha_max, min=0.0)
        
        # Euclidean norms: ||k_hat|| and ||k_tilde||
        norm_k_hat = torch.linalg.vector_norm(k_hat_f, ord=2, dim=-1)
        norm_k_tilde = torch.linalg.vector_norm(k_tilde_f, ord=2, dim=-1)
        
        # denom = ||k_tilde_t|| * ||k_hat_t|| + eps
        denom = norm_k_hat * norm_k_tilde + self.eps
        
        # beta_max = budget / denom
        beta_max = budget / denom
        
        # beta_sp = min(beta_asn, beta_max)
        beta_sp = torch.minimum(beta_asn_f, beta_max)
        
        return (
            beta_sp.to(dtype=beta_asn_t.dtype),
            beta_max.to(dtype=beta_asn_t.dtype),
            alpha_max.to(dtype=beta_asn_t.dtype)
        )

    @staticmethod
    def project_raw(
        alpha: torch.Tensor,
        beta: torch.Tensor,
        k_hat: torch.Tensor,
        k_tilde: torch.Tensor,
        margin: float = 1.0,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Direct functional projection matching pseudocode in Section 5.5:
        function StabilityProjection(alpha_t, beta_t, k_hat_t, k_tilde_t, margin=1.0, eps=1e-6):
            alpha_max = max(alpha_t)
            budget    = max(margin - alpha_max, 0.0)
            denom     = norm(k_hat_t) * norm(k_tilde_t) + eps
            beta_max  = budget / denom
            return min(beta_t, beta_max)
        """
        alpha_max = torch.amax(alpha, dim=-1)
        budget = torch.clamp(margin - alpha_max, min=0.0)
        norm_k_hat = torch.linalg.vector_norm(k_hat, ord=2, dim=-1)
        norm_k_tilde = torch.linalg.vector_norm(k_tilde, ord=2, dim=-1)
        denom = norm_k_hat * norm_k_tilde + eps
        beta_max = budget / denom
        return torch.minimum(beta, beta_max)
