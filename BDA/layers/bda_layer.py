"""
BUEORM Delta Attention (BDA) - Multi-Head Layer (BDALayer)
Spec v2: Section 7.1, 7.2, 7.3, 7.6
Full multi-head BDA layer with input projections, shared bottleneck,
LRFG, DEM, ASN, SP, CRPT training, and streaming autoregressive step inference.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Union

from BDA.config import BDAConfig
from BDA.ops.gates import LRFG, DEM
from BDA.ops.normalization import ASN
from BDA.ops.stability import StabilityProjection
from BDA.ops.state import MemoryState
from BDA.layers.bda_step import bda_head_step
from BDA.layers.crpt import CRPTFunction, crpt_forward_native


class BDALayer(nn.Module):
    """
    BUEORM Delta Attention Multi-Head Layer.
    
    Implements the 6 named components:
        1. Memory State (MS) per head
        2. Low-Rank Forget Gate (LRFG) with shared bottleneck V
        3. Decoupled Erase Mask (DEM) generating decoupled k_tilde
        4. Adaptive Step Normalization (ASN) with EMA buffer
        5. Stability Projection (SP) guaranteeing ||T_t||_2 <= 1
        6. Chunk-Recurrent Parallel Training (CRPT)
    """
    def __init__(self, config: Optional[BDAConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = BDAConfig(**kwargs)
        self.config = config
        
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.d_v = config.d_v
        self.rank_r = config.rank_r
        self.chunk_size = config.chunk_size
        self.ema_lambda = config.ema_lambda
        self.eps = config.eps
        self.stability_margin = config.stability_margin
        
        # Parallel input projections for Q, K, V
        self.W_q = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=config.bias)
        self.W_k = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=config.bias)
        self.W_v = nn.Linear(self.d_model, self.n_heads * self.d_v, bias=config.bias)
        
        # LRFG with shared bottleneck V and per-head U_alpha
        self.lrfg = LRFG(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_k=self.d_k,
            rank_r=self.rank_r,
            bias=config.bias
        )
        
        # DEM with per-head U_e reusing shared bottleneck
        self.dem = DEM(
            n_heads=self.n_heads,
            d_k=self.d_k,
            rank_r=self.rank_r,
            bias=config.bias
        )
        
        # ASN with linear projection w_beta and EMA tracking
        self.asn = ASN(
            d_model=self.d_model,
            n_heads=self.n_heads,
            ema_lambda=self.ema_lambda,
            eps=self.eps,
            bias=config.bias
        )
        
        # Stability Projection
        self.sp = StabilityProjection(
            margin=self.stability_margin,
            eps=self.eps
        )
        
        # Output projection W_out
        self.W_out = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=config.bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
        if self.W_q.bias is not None:
            nn.init.zeros_(self.W_q.bias)
            nn.init.zeros_(self.W_k.bias)
            nn.init.zeros_(self.W_v.bias)
            nn.init.zeros_(self.W_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_crpt_autograd: bool = True
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sequence forward pass.
        
        Args:
            x: Input sequence of shape (batch_size, seq_len, d_model)
            state: Optional tuple (S_prev, m_prev)
                   S_prev shape: (batch_size, n_heads, d_v, d_k)
                   m_prev shape: (batch_size, n_heads)
            use_crpt_autograd: Whether to use checkpointed CRPT autograd in training
            
        Returns:
            output: Shape (batch_size, seq_len, d_model)
            (S_final, m_final): Final states
        """
        B, T, _ = x.shape
        
        # 1. Parallel GEMM projections across entire sequence (Spec 7.7.2)
        q_all = self.W_q(x).view(B, T, self.n_heads, self.d_k)
        k_hat_all = self.W_k(x).view(B, T, self.n_heads, self.d_k)
        v_all = self.W_v(x).view(B, T, self.n_heads, self.d_v)
        
        # 2. Shared bottleneck z and gate projections (LRFG, DEM, ASN)
        alpha_all, z_all = self.lrfg(x)       # alpha: (B, T, H, d_k), z: (B, T, r)
        _, e_all = self.dem(k_hat_all, z_all) # e: (B, T, H, d_k)
        raw_beta_all = torch.sigmoid(self.asn.w_beta(x)) # (B, T, H)
        
        S_init = state[0] if state is not None else None
        m_init = state[1] if state is not None else None
        
        # 3. Recurrent chunk processing
        if self.training and use_crpt_autograd and x.requires_grad:
            o_seq, S_final, m_final = CRPTFunction.apply(
                q_all, k_hat_all, v_all, alpha_all, e_all, raw_beta_all,
                S_init, m_init, self.chunk_size, self.ema_lambda, self.eps, self.stability_margin
            )
        else:
            o_seq, S_final, m_final, _ = crpt_forward_native(
                q_seq=q_all,
                k_hat_seq=k_hat_all,
                v_seq=v_all,
                alpha_seq=alpha_all,
                e_seq=e_all,
                raw_beta_seq=raw_beta_all,
                S_init=S_init,
                m_init=m_init,
                chunk_size=self.chunk_size,
                ema_lambda=self.ema_lambda,
                eps=self.eps,
                stability_margin=self.stability_margin
            )
            
        # 4. Multi-head output concatenation & projection: W_out @ concat(outputs)
        o_concat = o_seq.reshape(B, T, self.n_heads * self.d_v)
        output = self.W_out(o_concat)
        
        return output, (S_final, m_final)

    def step(
        self,
        x_t: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single-step streaming inference for autoregressive generation (Spec 7.2, 7.3).
        
        Args:
            x_t: Input token representation of shape (batch_size, d_model)
            state: Tuple (S_prev, m_prev)
            
        Returns:
            out_t: Output token representation (batch_size, d_model)
            (S_t, m_t): Updated state tuple
        """
        B = x_t.shape[0]
        if state is None:
            S_prev = MemoryState.init_state(B, self.n_heads, self.d_v, self.d_k, device=x_t.device, dtype=x_t.dtype)
            m_prev = self.asn.init_ema_buffer(B, self.n_heads, device=x_t.device, dtype=x_t.dtype)
        else:
            S_prev, m_prev = state
            
        # Step projections
        q_t = self.W_q(x_t).view(B, self.n_heads, self.d_k)
        k_hat_t = self.W_k(x_t).view(B, self.n_heads, self.d_k)
        v_t = self.W_v(x_t).view(B, self.n_heads, self.d_v)
        
        alpha_t, z_t = self.lrfg(x_t)
        _, e_t = self.dem(k_hat_t, z_t)
        raw_beta_t = torch.sigmoid(self.asn.w_beta(x_t))
        
        o_t, S_t, m_t, _ = bda_head_step(
            S_prev=S_prev,
            m_prev=m_prev,
            q_t=q_t,
            k_hat_t=k_hat_t,
            v_t=v_t,
            alpha_t=alpha_t,
            e_t=e_t,
            raw_beta_t=raw_beta_t,
            ema_lambda=self.ema_lambda,
            eps=self.eps,
            stability_margin=self.stability_margin
        )
        
        o_concat = o_t.reshape(B, self.n_heads * self.d_v)
        out_t = self.W_out(o_concat)
        
        return out_t, (S_t, m_t)

    def get_state_norm(self, state: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Returns the spectral norm ||S_t||_2 for each head for health monitoring (Spec 8.0, item 7).
        """
        S = state[0]
        return MemoryState.spectral_norm(S)
