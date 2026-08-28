"""
BUEORM Delta Attention (BDA) - Validation Test 6
Spec v2: Section 5.5, Section 10 (Test 6)
Long-Horizon Stability: T=200 to T=8000 steps with and without Stability Projection.
"""

import torch
from typing import Dict, Any, List

from BDA.config import BDAConfig
from BDA.layers.bda_step import bda_head_step
from BDA.ops.state import MemoryState


def run_long_horizon_single(
    T: int,
    use_sp: bool = True,
    d_model: int = 64,
    d_k: int = 16,
    d_v: int = 16,
    n_heads: int = 4,
    seed: int = 42
) -> Dict[str, float]:
    """Runs a long-horizon recurrence with or without Stability Projection."""
    torch.manual_seed(seed)
    
    # Random projection weights (fixed across comparison)
    W_q = torch.randn(n_heads * d_k, d_model) / (d_model ** 0.5)
    W_k = torch.randn(n_heads * d_k, d_model) / (d_model ** 0.5)
    W_v = torch.randn(n_heads * d_v, d_model) / (d_model ** 0.5)
    V_gate = torch.randn(4, d_model) / (d_model ** 0.5)
    U_alpha = torch.randn(n_heads * d_k, 4)
    U_erase = torch.randn(n_heads * d_k, 4)
    w_beta = torch.randn(n_heads, d_model) / (d_model ** 0.5)
    
    S_curr = MemoryState.init_state(1, n_heads, d_v, d_k)
    m_curr = torch.ones(1, n_heads)
    
    max_out_mag = 0.0
    max_state_norm = 0.0
    
    for t in range(T):
        x_t = torch.randn(1, d_model)
        
        q_t = (x_t @ W_q.T).view(1, n_heads, d_k)
        k_hat_t = (x_t @ W_k.T).view(1, n_heads, d_k)
        v_t = (x_t @ W_v.T).view(1, n_heads, d_v)
        
        z_t = x_t @ V_gate.T
        alpha_t = torch.sigmoid(z_t @ U_alpha.T).view(1, n_heads, d_k)
        e_t = torch.sigmoid(z_t @ U_erase.T).view(1, n_heads, d_k)
        raw_beta_t = torch.sigmoid(x_t @ w_beta.T)  # (1, n_heads)
        
        k_tilde_t = k_hat_t * e_t
        
        # ASN
        k_norm_sq = torch.sum(k_hat_t ** 2, dim=-1)
        m_curr = 0.99 * m_curr + 0.01 * k_norm_sq
        beta_asn = raw_beta_t / (1e-6 + torch.sqrt(torch.clamp(m_curr, min=0.0)))
        
        if use_sp:
            alpha_max = torch.amax(alpha_t, dim=-1)
            budget = torch.clamp(1.0 - alpha_max, min=0.0)
            denom = torch.linalg.vector_norm(k_hat_t, dim=-1) * torch.linalg.vector_norm(k_tilde_t, dim=-1) + 1e-6
            beta_max = budget / denom
            beta_t = torch.minimum(beta_asn, beta_max)
        else:
            # Without SP, using ASN beta directly without spectral margin constraint
            beta_t = beta_asn
            
        S_decayed = MemoryState.apply_decay(S_curr, alpha_t)
        error_t = MemoryState.compute_error(S_decayed, k_tilde_t, v_t)
        S_curr = MemoryState.update_state(S_decayed, error_t, k_hat_t, beta_t)
        o_t = MemoryState.read_state(S_curr, q_t)
        
        out_mag = torch.max(torch.abs(o_t)).item()
        state_norm = torch.max(MemoryState.spectral_norm(S_curr)).item()
        
        if out_mag > max_out_mag:
            max_out_mag = out_mag
        if state_norm > max_state_norm:
            max_state_norm = state_norm
            
    return {
        "max_output_magnitude": max_out_mag,
        "max_state_norm": max_state_norm
    }


def run_test_6_long_horizon_stability(
    horizons: List[int] = [200, 500, 1000, 2000, 4000, 8000],
    seed: int = 42
) -> Dict[str, Any]:
    """
    Spec Section 10 Test 6:
    Runs comparison across horizons T in [200, ..., 8000] with and without Stability Projection.
    """
    results_with_sp = {}
    results_without_sp = {}
    
    for T in horizons:
        res_sp = run_long_horizon_single(T=T, use_sp=True, seed=seed)
        res_no_sp = run_long_horizon_single(T=T, use_sp=False, seed=seed)
        results_with_sp[T] = res_sp
        results_without_sp[T] = res_no_sp
        
    sp_is_bounded = results_with_sp[8000]["max_output_magnitude"] < results_without_sp[8000]["max_output_magnitude"]
    
    results = {
        "test_name": "Test 6: Long-Horizon Stability (T=200 to T=8000)",
        "horizons": horizons,
        "with_sp": results_with_sp,
        "without_sp": results_without_sp,
        "sp_bounded_verified": sp_is_bounded,
        "status": "PASSED_STABILITY_BOUNDED" if sp_is_bounded else "FAILED"
    }
    
    return results


if __name__ == "__main__":
    res = run_test_6_long_horizon_stability()
    print("=" * 70)
    print(res["test_name"])
    print("=" * 70)
    print(f"{'Horizon (T)':<12} | {'With SP (Max Out)':<20} | {'Without SP (Max Out)':<20}")
    print("-" * 60)
    for T in res["horizons"]:
        out_sp = res["with_sp"][T]["max_output_magnitude"]
        out_no_sp = res["without_sp"][T]["max_output_magnitude"]
        print(f"{T:<12} | {out_sp:<20.4f} | {out_no_sp:<20.4f}")
    print("=" * 70)
    print(f"Result Status: {res['status']}")
    print("=" * 70)
