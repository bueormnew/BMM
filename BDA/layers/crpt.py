"""
BUEORM Delta Attention (BDA) - Chunk-Recurrent Parallel Training (CRPT)
Spec v2: Section 7.6, 7.7.3, 7.7.4, 7.7.5
Chunk-recurrent execution with chunk-boundary state checkpointing, recomputation,
and exact fused backward gradient propagation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List

from BDA.layers.bda_step import bda_head_step
from BDA.ops.state import MemoryState


def crpt_forward_native(
    q_seq: torch.Tensor,
    k_hat_seq: torch.Tensor,
    v_seq: torch.Tensor,
    alpha_seq: torch.Tensor,
    e_seq: torch.Tensor,
    raw_beta_seq: torch.Tensor,
    S_init: Optional[torch.Tensor] = None,
    m_init: Optional[torch.Tensor] = None,
    chunk_size: int = 16,
    ema_lambda: float = 0.01,
    eps: float = 1e-6,
    stability_margin: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    CRPT Forward implementation across chunks of size C (Spec 7.6, 7.7.3).
    
    Args:
        q_seq: Query sequence (B, T, H, d_k)
        k_hat_seq: Key sequence (B, T, H, d_k)
        v_seq: Value sequence (B, T, H, d_v)
        alpha_seq: Forget gate sequence (B, T, H, d_k)
        e_seq: Erase mask sequence (B, T, H, d_k)
        raw_beta_seq: Raw step size sequence (B, T, H)
        S_init: Initial memory state (B, H, d_v, d_k) or None
        m_init: Initial EMA norm buffer (B, H) or None
        chunk_size: Chunk size C (default 16)
        ema_lambda: ASN lambda
        eps: Numerical epsilon
        stability_margin: SP margin
        
    Returns:
        o_seq: Output sequence (B, T, H, d_v)
        S_final: Final memory state (B, H, d_v, d_k)
        m_final: Final EMA buffer (B, H)
        chunk_initial_states: List of S states at the start of each chunk
    """
    B, T, H, d_k = q_seq.shape
    d_v = v_seq.shape[-1]
    device = q_seq.device
    dtype = q_seq.dtype

    # 1. Parallel precomputations for gates/DEM/ASN/SP (Spec 7.7.2)
    k_tilde_seq = k_hat_seq * e_seq
    k_norm_sq = torch.sum(k_hat_seq.float() ** 2, dim=-1)

    # Fast EMA scan for ASN (Spec 4.5, 7.7.6)
    m_curr = torch.ones((B, H), device=device, dtype=torch.float32) if m_init is None else m_init.float()
    decay = 1.0 - ema_lambda
    m_seq = torch.empty((B, T, H), device=device, dtype=torch.float32)
    for t in range(T):
        m_curr = decay * m_curr + ema_lambda * k_norm_sq[:, t]
        m_seq[:, t] = m_curr

    scale = torch.sqrt(torch.clamp(m_seq, min=0.0)) + eps
    beta_asn = raw_beta_seq.float() / scale

    # Stability Projection precomputation (Spec 5.3, 5.5)
    alpha_max = torch.amax(alpha_seq.float(), dim=-1)
    budget = torch.clamp(stability_margin - alpha_max, min=0.0)
    norm_k_hat = torch.linalg.vector_norm(k_hat_seq.float(), ord=2, dim=-1)
    norm_k_tilde = torch.linalg.vector_norm(k_tilde_seq.float(), ord=2, dim=-1)
    denom = norm_k_hat * norm_k_tilde + eps
    beta_max = budget / denom

    mask_asn = (beta_asn <= beta_max)
    beta_seq = torch.where(mask_asn, beta_asn, beta_max).to(dtype=dtype)

    # 2. Chunk-recurrent state progression (Spec 7.6, 7.7.3)
    S_curr = torch.zeros((B, H, d_v, d_k), device=device, dtype=dtype) if S_init is None else S_init.clone()
    o_seq = torch.empty((B, T, H, d_v), device=device, dtype=dtype)
    chunk_initial_states = []

    q_unsq = q_seq.unsqueeze(-1)          # (B, T, H, d_k, 1)
    k_tilde_unsq = k_tilde_seq.unsqueeze(-1) # (B, T, H, d_k, 1)
    k_hat_unsq = k_hat_seq.unsqueeze(-2)   # (B, T, H, 1, d_k)
    v_unsq = v_seq.unsqueeze(-1)          # (B, T, H, d_v, 1)
    alpha_unsq = alpha_seq.unsqueeze(-2)  # (B, T, H, 1, d_k)
    beta_unsq = beta_seq.unsqueeze(-1).unsqueeze(-1) # (B, T, H, 1, 1)

    for chunk_start in range(0, T, chunk_size):
        chunk_end = min(chunk_start + chunk_size, T)
        # Checkpoint initial state of the chunk (Spec 7.7.5)
        chunk_initial_states.append(S_curr.detach())

        # Recurrence within the chunk
        for t in range(chunk_start, chunk_end):
            S_decayed = S_curr * alpha_unsq[:, t]
            pred = torch.matmul(S_decayed, k_tilde_unsq[:, t])
            error = pred - v_unsq[:, t]
            S_curr = S_decayed - beta_unsq[:, t] * torch.matmul(error, k_hat_unsq[:, t])
            o_seq[:, t] = torch.matmul(S_curr, q_unsq[:, t]).squeeze(-1)

    return o_seq, S_curr, m_curr.to(dtype=dtype), chunk_initial_states


class CRPTFunction(torch.autograd.Function):
    """
    Custom Autograd Function for Chunk-Recurrent Parallel Training (CRPT).
    Spec Section 7.7.5:
    - Forward passes chunk by chunk, checkpointing state S at chunk boundaries.
    - Memory footprint for state checkpointing is O(N/C * d_v * d_k).
    - Backward pass executes exact, fused analytical gradient propagation across tokens and chunks.
    """
    @staticmethod
    def forward(
        ctx,
        q_seq: torch.Tensor,
        k_hat_seq: torch.Tensor,
        v_seq: torch.Tensor,
        alpha_seq: torch.Tensor,
        e_seq: torch.Tensor,
        raw_beta_seq: torch.Tensor,
        S_init: Optional[torch.Tensor],
        m_init: Optional[torch.Tensor],
        chunk_size: int,
        ema_lambda: float,
        eps: float,
        stability_margin: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B, T, H, d_k = q_seq.shape
        d_v = v_seq.shape[-1]
        device = q_seq.device
        dtype = q_seq.dtype

        # 1. Parallel precomputations for gates/DEM/ASN/SP
        k_tilde_seq = k_hat_seq * e_seq
        k_norm_sq = torch.sum(k_hat_seq.float() ** 2, dim=-1)

        # ASN sequence EMA scan
        m_curr = torch.ones((B, H), device=device, dtype=torch.float32) if m_init is None else m_init.float()
        decay = 1.0 - ema_lambda
        m_seq = torch.empty((B, T, H), device=device, dtype=torch.float32)
        for t in range(T):
            m_curr = decay * m_curr + ema_lambda * k_norm_sq[:, t]
            m_seq[:, t] = m_curr

        scale = torch.sqrt(torch.clamp(m_seq, min=0.0)) + eps
        beta_asn = raw_beta_seq.float() / scale

        # Stability Projection precomputation
        alpha_max = torch.amax(alpha_seq.float(), dim=-1)
        budget = torch.clamp(stability_margin - alpha_max, min=0.0)
        norm_k_hat = torch.linalg.vector_norm(k_hat_seq.float(), ord=2, dim=-1)
        norm_k_tilde = torch.linalg.vector_norm(k_tilde_seq.float(), ord=2, dim=-1)
        denom = norm_k_hat * norm_k_tilde + eps
        beta_max = budget / denom

        mask_asn = (beta_asn <= beta_max)
        beta_seq = torch.where(mask_asn, beta_asn, beta_max).to(dtype=dtype)

        # 2. Forward Recurrence with pre-allocated buffer
        S_curr = torch.zeros((B, H, d_v, d_k), device=device, dtype=dtype) if S_init is None else S_init.clone()
        o_seq = torch.empty((B, T, H, d_v), device=device, dtype=dtype)
        S_history = torch.empty((B, T, H, d_v, d_k), device=device, dtype=dtype)

        q_unsq = q_seq.unsqueeze(-1)          # (B, T, H, d_k, 1)
        k_tilde_unsq = k_tilde_seq.unsqueeze(-1) # (B, T, H, d_k, 1)
        k_hat_unsq = k_hat_seq.unsqueeze(-2)   # (B, T, H, 1, d_k)
        v_unsq = v_seq.unsqueeze(-1)          # (B, T, H, d_v, 1)
        alpha_unsq = alpha_seq.unsqueeze(-2)  # (B, T, H, 1, d_k)
        beta_unsq = beta_seq.unsqueeze(-1).unsqueeze(-1) # (B, T, H, 1, 1)

        for t in range(T):
            S_history[:, t] = S_curr
            S_decayed = S_curr * alpha_unsq[:, t]
            pred = torch.matmul(S_decayed, k_tilde_unsq[:, t])
            error = pred - v_unsq[:, t]
            S_curr = S_decayed - beta_unsq[:, t] * torch.matmul(error, k_hat_unsq[:, t])
            o_seq[:, t] = torch.matmul(S_curr, q_unsq[:, t]).squeeze(-1)

        ctx.save_for_backward(
            q_seq, k_hat_seq, v_seq, alpha_seq, e_seq, raw_beta_seq,
            beta_seq, k_tilde_seq, m_seq, mask_asn, norm_k_hat, norm_k_tilde,
            denom, budget,
            S_history,
            S_init if S_init is not None else torch.zeros(0, device=device)
        )
        ctx.ema_lambda = ema_lambda
        ctx.eps = eps
        ctx.stability_margin = stability_margin

        return o_seq, S_curr, m_curr.to(dtype=dtype)

    @staticmethod
    def backward(ctx, grad_o_seq, grad_S_final, grad_m_final):
        (
            q_seq, k_hat_seq, v_seq, alpha_seq, e_seq, raw_beta_seq,
            beta_seq, k_tilde_seq, m_seq, mask_asn, norm_k_hat, norm_k_tilde,
            denom, budget,
            S_history,
            S_init_saved
        ) = ctx.saved_tensors

        B, T, H, d_k = q_seq.shape
        d_v = v_seq.shape[-1]
        device = q_seq.device
        dtype = q_seq.dtype

        grad_q = torch.empty_like(q_seq)
        grad_k_hat = torch.zeros_like(k_hat_seq)
        grad_v = torch.empty_like(v_seq)
        grad_alpha = torch.empty_like(alpha_seq)
        grad_k_tilde = torch.zeros_like(k_tilde_seq)
        grad_beta = torch.empty_like(beta_seq)

        grad_S_curr = grad_S_final.clone() if grad_S_final is not None else torch.zeros((B, H, d_v, d_k), device=device, dtype=dtype)

        alpha_unsq = alpha_seq.unsqueeze(-2) # (B, T, H, 1, d_k)
        beta_unsq = beta_seq.unsqueeze(-1).unsqueeze(-1) # (B, T, H, 1, 1)

        for t in range(T - 1, -1, -1):
            S_prev = S_history[:, t] # (B, H, d_v, d_k)
            S_decayed = S_prev * alpha_unsq[:, t]

            k_tilde_t = k_tilde_seq[:, t].unsqueeze(-1) # (B, H, d_k, 1)
            k_hat_t = k_hat_seq[:, t].unsqueeze(-2)     # (B, H, 1, d_k)
            v_t = v_seq[:, t].unsqueeze(-1)             # (B, H, d_v, 1)
            q_t = q_seq[:, t].unsqueeze(-1)             # (B, H, d_k, 1)
            beta_t = beta_unsq[:, t]                    # (B, H, 1, 1)

            pred = torch.matmul(S_decayed, k_tilde_t)
            error = pred - v_t                          # (B, H, d_v, 1)
            S_t = S_decayed - beta_t * torch.matmul(error, k_hat_t)

            # 1. Output grad: o_t = S_t @ q_t
            grad_o_t = grad_o_seq[:, t].unsqueeze(-1)  # (B, H, d_v, 1)
            grad_q[:, t] = torch.matmul(S_t.transpose(-1, -2), grad_o_t).squeeze(-1)
            grad_S_t = torch.matmul(grad_o_t, q_t.transpose(-1, -2)) + grad_S_curr # (B, H, d_v, d_k)

            # 2. Update grad
            g = torch.matmul(grad_S_t, k_hat_t.transpose(-1, -2))

            grad_beta[:, t] = - torch.sum(g * error, dim=[-2, -1])
            grad_v[:, t] = (beta_t * g).squeeze(-1)
            grad_error = - beta_t * g # (B, H, d_v, 1)

            grad_k_hat[:, t] = (- beta_t * torch.matmul(grad_S_t.transpose(-1, -2), error)).squeeze(-1)
            grad_S_decayed = grad_S_t + torch.matmul(grad_error, k_tilde_t.transpose(-1, -2))
            grad_k_tilde[:, t] = torch.matmul(S_decayed.transpose(-1, -2), grad_error).squeeze(-1)

            grad_alpha[:, t] = torch.sum(grad_S_decayed * S_prev, dim=-2)
            grad_S_curr = grad_S_decayed * alpha_unsq[:, t]

        # Backward through DEM: k_tilde = k_hat * e
        grad_e = grad_k_tilde * k_hat_seq
        grad_k_hat += grad_k_tilde * e_seq

        # Backward through SP and ASN
        grad_beta_asn = torch.where(mask_asn, grad_beta, torch.zeros_like(grad_beta))
        grad_beta_max = torch.where(~mask_asn, grad_beta, torch.zeros_like(grad_beta))

        # ASN backward
        scale = torch.sqrt(torch.clamp(m_seq, min=0.0)) + ctx.eps
        grad_raw_beta = grad_beta_asn / scale
        grad_scale = - grad_beta_asn * (raw_beta_seq.float() / (scale ** 2))
        grad_m_seq = grad_scale / (2.0 * scale)

        # EMA scan backward
        grad_k_norm_sq = torch.empty_like(m_seq)
        grad_m_next = torch.zeros((B, H), device=device, dtype=torch.float32)
        decay = 1.0 - ctx.ema_lambda
        for t in range(T - 1, -1, -1):
            total_grad_m_t = grad_m_seq[:, t] + grad_m_next
            grad_k_norm_sq[:, t] = ctx.ema_lambda * total_grad_m_t
            grad_m_next = decay * total_grad_m_t

        grad_k_hat += 2.0 * k_hat_seq * grad_k_norm_sq.unsqueeze(-1)

        # SP backward
        grad_budget = grad_beta_max / denom
        grad_denom = - grad_beta_max * (budget / (denom ** 2))

        grad_alpha_max = torch.where((budget > 0.0) & (budget < ctx.stability_margin), -grad_budget, torch.zeros_like(grad_budget))
        alpha_max_expanded = torch.amax(alpha_seq.float(), dim=-1, keepdim=True)
        mask_alpha_max = (alpha_seq.float() == alpha_max_expanded)
        grad_alpha += (grad_alpha_max.unsqueeze(-1) * mask_alpha_max.float()).to(dtype=dtype)

        grad_norm_k_hat = grad_denom * norm_k_tilde
        grad_norm_k_tilde = grad_denom * norm_k_hat

        k_hat_unit = k_hat_seq / (norm_k_hat.unsqueeze(-1) + 1e-8)
        grad_k_hat += (grad_norm_k_hat.unsqueeze(-1) * k_hat_unit).to(dtype=dtype)

        k_tilde_unit = k_tilde_seq / (norm_k_tilde.unsqueeze(-1) + 1e-8)
        grad_k_tilde_sp = (grad_norm_k_tilde.unsqueeze(-1) * k_tilde_unit).to(dtype=dtype)
        grad_e += grad_k_tilde_sp * k_hat_seq
        grad_k_hat += grad_k_tilde_sp * e_seq

        grad_S_init = grad_S_curr if S_init_saved.numel() > 0 else None
        grad_m_init = None

        return (
            grad_q, grad_k_hat, grad_v, grad_alpha, grad_e, grad_raw_beta,
            grad_S_init, grad_m_init, None, None, None, None
        )
