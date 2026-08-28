"""
BUEORM Delta Attention (BDA) - Adaptive Step Normalization (ASN)
Spec v2: Section 4.5
Adaptive Step Normalization with per-head EMA buffer tracking key norms.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class ASN(nn.Module):
    """
    Adaptive Step Normalization (ASN).
    
    Spec Section 4.5:
    Normalizes the correction rate beta_t against key magnitude variations using an exponential moving average (EMA)
    of the squared norm of the write key ||k_hat_t||^2:
        m_t = (1 - lambda) * m_{t-1} + lambda * ||k_hat_t||^2   (scalar EMA per head)
        beta_t^{ASN} = sigma(w_beta^T x_t) / (eps + sqrt(m_t))
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ema_lambda: float = 0.01,
        eps: float = 1e-6,
        bias: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.ema_lambda = ema_lambda
        self.eps = eps
        
        # Linear projection for raw beta: R^{d_model} -> R^{n_heads}
        self.w_beta = nn.Linear(d_model, n_heads, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_beta.weight)
        if self.w_beta.bias is not None:
            # Initialize with small positive bias for moderate initial learning rate
            nn.init.constant_(self.w_beta.bias, 0.0)

    @staticmethod
    def init_ema_buffer(
        batch_size: int,
        n_heads: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Initializes EMA buffer m to ones or zeros.
        Shape: (batch_size, n_heads)
        """
        return torch.ones((batch_size, n_heads), device=device, dtype=dtype)

    def step(
        self,
        x_t: torch.Tensor,
        k_hat_t: torch.Tensor,
        m_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step ASN calculation for token x_t.
        
        Args:
            x_t: Input token representation of shape (batch_size, d_model)
            k_hat_t: Write key tensor of shape (batch_size, n_heads, d_k)
            m_prev: Previous EMA scalar buffer of shape (batch_size, n_heads)
            
        Returns:
            beta_asn: Normalized beta of shape (batch_size, n_heads)
            m_t: Updated EMA scalar buffer of shape (batch_size, n_heads)
            raw_beta: Raw unnormalized sigmoid output of shape (batch_size, n_heads)
        """
        # raw_beta = sigma(w_beta @ x_t) in (0, 1)
        raw_beta = torch.sigmoid(self.w_beta(x_t))  # (batch_size, n_heads)
        
        # ||k_hat_t||^2 computed in float32 for numerical stability
        k_norm_sq = torch.sum(k_hat_t.float() ** 2, dim=-1)  # (batch_size, n_heads)
        
        # Update EMA buffer
        m_t = (1.0 - self.ema_lambda) * m_prev.float() + self.ema_lambda * k_norm_sq
        m_t = m_t.to(dtype=x_t.dtype)
        
        # beta_t^{ASN} = raw_beta / (eps + sqrt(m_t))
        scale = torch.sqrt(torch.clamp(m_t.float(), min=0.0)) + self.eps
        beta_asn = (raw_beta.float() / scale).to(dtype=x_t.dtype)
        
        return beta_asn, m_t, raw_beta

    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        k_hat_seq: torch.Tensor,
        m_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sequence ASN calculation across sequence length T.
        
        Args:
            x_seq: Input sequence of shape (batch_size, seq_len, d_model)
            k_hat_seq: Write keys of shape (batch_size, seq_len, n_heads, d_k)
            m_init: Optional initial EMA buffer (batch_size, n_heads)
            
        Returns:
            beta_asn_seq: Shape (batch_size, seq_len, n_heads)
            m_final: Shape (batch_size, n_heads)
            raw_beta_seq: Shape (batch_size, seq_len, n_heads)
        """
        batch_size, seq_len, _ = x_seq.shape
        raw_beta_seq = torch.sigmoid(self.w_beta(x_seq))  # (batch_size, seq_len, n_heads)
        
        if m_init is None:
            m_curr = self.init_ema_buffer(batch_size, self.n_heads, device=x_seq.device, dtype=torch.float32)
        else:
            m_curr = m_init.float().clone()
            
        beta_asn_list = []
        k_norm_sq_seq = torch.sum(k_hat_seq.float() ** 2, dim=-1)  # (batch_size, seq_len, n_heads)
        
        for t in range(seq_len):
            k_norm_sq_t = k_norm_sq_seq[:, t, :]
            m_curr = (1.0 - self.ema_lambda) * m_curr + self.ema_lambda * k_norm_sq_t
            scale_t = torch.sqrt(torch.clamp(m_curr, min=0.0)) + self.eps
            beta_asn_t = (raw_beta_seq[:, t, :].float() / scale_t).to(dtype=x_seq.dtype)
            beta_asn_list.append(beta_asn_t)
            
        beta_asn_seq = torch.stack(beta_asn_list, dim=1)
        return beta_asn_seq, m_curr.to(dtype=x_seq.dtype), raw_beta_seq
