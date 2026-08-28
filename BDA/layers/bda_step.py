"""
BUEORM Delta Attention (BDA) - Single Step Recurrent Computation
Spec v2: Section 7.2, 7.3
Single-step recurrence for streaming inference, token generation, and reference verification.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

from BDA.ops.state import MemoryState
from BDA.ops.gates import LRFG, DEM
from BDA.ops.normalization import ASN
from BDA.ops.stability import StabilityProjection


def bda_head_step(
    S_prev: torch.Tensor,
    m_prev: torch.Tensor,
    q_t: torch.Tensor,
    k_hat_t: torch.Tensor,
    v_t: torch.Tensor,
    alpha_t: torch.Tensor,
    e_t: torch.Tensor,
    raw_beta_t: torch.Tensor,
    ema_lambda: float = 0.01,
    eps: float = 1e-6,
    stability_margin: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes a single token update for one or multiple heads given projected inputs:
    
    Args:
        S_prev: Previous memory state of shape (..., d_v, d_k)
        m_prev: Previous EMA norm scalar of shape (...)
        q_t: Query vector of shape (..., d_k)
        k_hat_t: Write key vector of shape (..., d_k)
        v_t: Target value vector of shape (..., d_v)
        alpha_t: Forget gate vector of shape (..., d_k) in (0, 1)
        e_t: Erase mask vector of shape (..., d_k) in (0, 1)
        raw_beta_t: Raw sigmoid step size of shape (...) in (0, 1)
        ema_lambda: ASN EMA decay parameter
        eps: Numerical epsilon
        stability_margin: SP target margin (default 1.0)
        
    Returns:
        o_t: Output retrieval vector (..., d_v)
        S_t: Updated memory state (..., d_v, d_k)
        m_t: Updated EMA scalar (...)
        beta_t: Projected step size (...)
    """
    # 1. DEM: Compute decoupled erase key
    k_tilde_t = k_hat_t * e_t
    
    # 2. ASN: Adaptive Step Normalization (Spec 4.5 & 7.7.6 in float32)
    k_norm_sq = torch.sum(k_hat_t.float() ** 2, dim=-1)
    m_t = (1.0 - ema_lambda) * m_prev.float() + ema_lambda * k_norm_sq
    scale = torch.sqrt(torch.clamp(m_t, min=0.0)) + eps
    beta_asn = (raw_beta_t.float() / scale).to(dtype=raw_beta_t.dtype)
    
    # 3. SP: Stability Projection (Spec 5.3, 5.5)
    alpha_max = torch.amax(alpha_t.float(), dim=-1)
    budget = torch.clamp(stability_margin - alpha_max, min=0.0)
    norm_k_hat = torch.linalg.vector_norm(k_hat_t.float(), ord=2, dim=-1)
    norm_k_tilde = torch.linalg.vector_norm(k_tilde_t.float(), ord=2, dim=-1)
    denom = norm_k_hat * norm_k_tilde + eps
    beta_max = budget / denom
    beta_t = torch.minimum(beta_asn.float(), beta_max).to(dtype=raw_beta_t.dtype)
    
    # 4. Memory State Update (Spec 7.2)
    # Apply channel-wise decay: S_decayed = S_prev * alpha_t[..., None, :]
    S_decayed = MemoryState.apply_decay(S_prev, alpha_t)
    
    # Compute error: error_t = S_decayed @ k_tilde_t - v_t
    error_t = MemoryState.compute_error(S_decayed, k_tilde_t, v_t)
    
    # Update state: S_t = S_decayed - beta_t * (error_t (x) k_hat_t)
    S_t = MemoryState.update_state(S_decayed, error_t, k_hat_t, beta_t)
    
    # 5. Read output: o_t = S_t @ q_t
    o_t = MemoryState.read_state(S_t, q_t)
    
    return o_t, S_t, m_t.to(dtype=m_prev.dtype), beta_t
