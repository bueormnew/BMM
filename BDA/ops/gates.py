"""
BUEORM Delta Attention (BDA) - Gates (LRFG & DEM)
Spec v2: Section 4.3 (LRFG) & Section 4.4 (DEM)
Low-Rank Forget Gate and Decoupled Erase Mask.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional


class LRFG(nn.Module):
    """
    Low-Rank Forget Gate (LRFG).
    
    Spec Section 4.3:
    Instead of an expensive full projection R^{d_model} -> R^{d_k} per head (cost O(d_model * d_k * H)),
    LRFG uses a shared low-rank bottleneck z_t = V @ x_t (V in R^{r x d_model}, r << d_k),
    and per-head projections:
        alpha_t^h = sigma(U_alpha^h @ z_t + b_alpha^h)  in (0, 1)^{d_k}
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: int,
        rank_r: int,
        bias: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.rank_r = rank_r
        
        # Shared bottleneck V: R^{d_model} -> R^r
        self.V = nn.Linear(d_model, rank_r, bias=False)
        
        # Per-head projections U_alpha: R^r -> R^{n_heads * d_k}
        self.U_alpha = nn.Linear(rank_r, n_heads * d_k, bias=bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize bottleneck and per-head weights
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.U_alpha.weight)
        if self.U_alpha.bias is not None:
            # Initialize forget gate bias slightly positive for gentle retention initially
            nn.init.constant_(self.U_alpha.bias, 1.0)

    def forward(self, x: torch.Tensor, z_shared: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (..., d_model)
            z_shared: Optional precomputed shared bottleneck tensor of shape (..., rank_r)
            
        Returns:
            alpha: Forget gate tensor of shape (..., n_heads, d_k) in (0, 1)
            z: Shared bottleneck tensor of shape (..., rank_r)
        """
        if z_shared is None:
            z = self.V(x)  # (..., rank_r)
        else:
            z = z_shared
            
        # Project from bottleneck to per-head forget vectors
        raw_alpha = self.U_alpha(z)  # (..., n_heads * d_k)
        
        # Reshape to (..., n_heads, d_k)
        shape = raw_alpha.shape[:-1] + (self.n_heads, self.d_k)
        alpha = torch.sigmoid(raw_alpha.view(shape))
        
        return alpha, z


class DEM(nn.Module):
    """
    Decoupled Erase Mask (DEM).
    
    Spec Section 4.4:
    Decouples the erase/reading key k_tilde_t from the write key k_hat_t to prevent interference.
    Reuses the shared bottleneck z_t from LRFG for maximum computational efficiency:
        e_t^h = sigma(U_e^h @ z_t + b_e^h)  in (0, 1)^{d_k}
        k_tilde_t^h = k_hat_t^h (elem-mult) e_t^h
    """
    def __init__(
        self,
        n_heads: int,
        d_k: int,
        rank_r: int,
        bias: bool = True
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.rank_r = rank_r
        
        # Per-head erase projection U_e: R^r -> R^{n_heads * d_k}
        self.U_e = nn.Linear(rank_r, n_heads * d_k, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.U_e.weight)
        if self.U_e.bias is not None:
            # Initialize with small positive bias for balanced initial erase mask
            nn.init.constant_(self.U_e.bias, 0.0)

    def forward(self, k_hat: torch.Tensor, z_shared: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            k_hat: Write key tensor of shape (..., n_heads, d_k)
            z_shared: Shared bottleneck tensor of shape (..., rank_r)
            
        Returns:
            k_tilde: Decoupled erase key of shape (..., n_heads, d_k)
            e_mask: Erase mask of shape (..., n_heads, d_k) in (0, 1)
        """
        raw_e = self.U_e(z_shared)  # (..., n_heads * d_k)
        shape = raw_e.shape[:-1] + (self.n_heads, self.d_k)
        e_mask = torch.sigmoid(raw_e.view(shape))
        
        k_tilde = k_hat * e_mask
        return k_tilde, e_mask
