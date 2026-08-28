"""
BUEORM Delta Attention (BDA) - Memory State (MS)
Spec v2: Section 4.1, 4.2, 4.4, 7.2
Memory State operations as linear associative memory S in R^{d_v x d_k}.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class MemoryState:
    """
    Primitives for manipulating the BDA Memory State S in R^{d_v x d_k}.
    
    In BDA, each head maintains S initialized to zeros as a recurrent buffer (not trainable parameter).
    The associative reading operation is o = S @ q.
    The update rule follows error-correction:
        S_decayed = S * alpha_t^T (column scaling by alpha_t)
        error_t = S_decayed @ k_tilde_t - v_t
        S_t = S_decayed - beta_t * (error_t (x) k_hat_t)
    """

    @staticmethod
    def init_state(
        batch_size: int,
        n_heads: int,
        d_v: int,
        d_k: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Initializes memory state S to zeros.
        Shape: (batch_size, n_heads, d_v, d_k)
        """
        return torch.zeros((batch_size, n_heads, d_v, d_k), device=device, dtype=dtype)

    @staticmethod
    def apply_decay(S: torch.Tensor, alpha_t: torch.Tensor) -> torch.Tensor:
        """
        Applies per-channel decay alpha_t to memory state columns.
        
        Args:
            S: Memory state of shape (..., d_v, d_k)
            alpha_t: Forget vector in (0, 1) of shape (..., d_k)
            
        Returns:
            S_decayed of shape (..., d_v, d_k)
        """
        # alpha_t is expanded along d_v: alpha_t[..., None, :]
        return S * alpha_t.unsqueeze(-2)

    @staticmethod
    def compute_error(
        S_decayed: torch.Tensor,
        k_tilde_t: torch.Tensor,
        v_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the prediction error on the erase/read key k_tilde_t:
            error_t = S_decayed @ k_tilde_t - v_t
            
        Args:
            S_decayed: Decayed memory state (..., d_v, d_k)
            k_tilde_t: Decoupled erase/read key (..., d_k)
            v_t: Target value to write (..., d_v)
            
        Returns:
            error_t: Prediction error vector (..., d_v)
        """
        # S_decayed @ k_tilde_t -> (..., d_v)
        prediction = torch.matmul(S_decayed, k_tilde_t.unsqueeze(-1)).squeeze(-1)
        return prediction - v_t

    @staticmethod
    def update_state(
        S_decayed: torch.Tensor,
        error_t: torch.Tensor,
        k_hat_t: torch.Tensor,
        beta_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies rank-1 error correction update:
            S_t = S_decayed - beta_t * (error_t (x) k_hat_t)
            
        Args:
            S_decayed: Decayed state (..., d_v, d_k)
            error_t: Prediction error (..., d_v)
            k_hat_t: Write key (..., d_k)
            beta_t: Step size scalar (..., 1) or broadcastable
            
        Returns:
            S_t: Updated memory state (..., d_v, d_k)
        """
        # outer product: error_t (..., d_v, 1) @ k_hat_t (..., 1, d_k)
        outer_prod = torch.matmul(error_t.unsqueeze(-1), k_hat_t.unsqueeze(-2))
        
        if beta_t.ndim < outer_prod.ndim:
            beta_t = beta_t.unsqueeze(-1)
            if beta_t.ndim < outer_prod.ndim:
                beta_t = beta_t.unsqueeze(-1)
                
        return S_decayed - beta_t * outer_prod

    @staticmethod
    def read_state(S_t: torch.Tensor, q_t: torch.Tensor) -> torch.Tensor:
        """
        Linear associative retrieval from memory state:
            o_t = S_t @ q_t
            
        Args:
            S_t: Memory state (..., d_v, d_k)
            q_t: Query vector (..., d_k)
            
        Returns:
            o_t: Output retrieval vector (..., d_v)
        """
        return torch.matmul(S_t, q_t.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def spectral_norm(S: torch.Tensor) -> torch.Tensor:
        """
        Computes the operator norm (spectral norm / max singular value) ||S||_2.
        Args:
            S: Matrix of shape (..., d_v, d_k)
        Returns:
            Spectral norm of shape (...)
        """
        # torch.linalg.norm(..., ord=2) or svdvals max
        return torch.linalg.matrix_norm(S, ord=2)
