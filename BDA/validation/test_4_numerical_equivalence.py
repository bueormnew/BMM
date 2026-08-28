"""
BUEORM Delta Attention (BDA) - Validation Test 4
Spec v2: Section 7.6, Section 10 (Test 4)
Numerical Equivalence: Full Sequential Recurrence vs. Chunk-Recurrent (CRPT).
"""

import torch
from typing import Dict, Any

from BDA.config import BDAConfig
from BDA.layers.bda_layer import BDALayer


def run_test_4_numerical_equivalence(
    B: int = 2,
    T: int = 97,
    H: int = 4,
    d_k: int = 32,
    d_v: int = 32,
    d_model: int = 128,
    chunk_size: int = 16,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Spec Section 10 Test 4:
    - B=2, T=97, H=4, d_k=d_v=32, chunk_size=16.
    - Compares full token-by-token sequential step vs Chunk-Recurrent Parallel Training (CRPT).
    - Checks that max absolute difference in outputs is 0.0 (or within float precision).
    """
    torch.manual_seed(seed)
    
    config = BDAConfig(
        d_model=d_model,
        n_heads=H,
        d_k=d_k,
        d_v=d_v,
        rank_r=8,
        chunk_size=chunk_size,
        ema_lambda=0.01,
        eps=1e-6,
        stability_margin=1.0
    )
    
    layer = BDALayer(config)
    layer.eval()
    
    x = torch.randn(B, T, d_model)
    
    # 1. Run Chunk-Recurrent forward
    with torch.no_grad():
        out_crpt, (S_crpt, m_crpt) = layer(x, use_crpt_autograd=False)
        
    # 2. Run strictly sequential step-by-step recurrence
    with torch.no_grad():
        seq_outs = []
        state_curr = None
        for t in range(T):
            x_t = x[:, t, :]
            out_t, state_curr = layer.step(x_t, state=state_curr)
            seq_outs.append(out_t)
            
        out_seq = torch.stack(seq_outs, dim=1)
        S_seq, m_seq = state_curr
        
    # Compute differences
    max_diff_out = torch.max(torch.abs(out_crpt - out_seq)).item()
    max_diff_S = torch.max(torch.abs(S_crpt - S_seq)).item()
    max_diff_m = torch.max(torch.abs(m_crpt - m_seq)).item()
    
    is_exact = max_diff_out < 1e-6
    
    results = {
        "test_name": "Test 4: Numerical Equivalence (Sequential vs CRPT)",
        "config": f"B={B}, T={T}, H={H}, d_k={d_k}, d_v={d_v}, chunk_size={chunk_size}",
        "max_abs_diff_outputs": max_diff_out,
        "max_abs_diff_S": max_diff_S,
        "max_abs_diff_m": max_diff_m,
        "exact_match": is_exact,
        "status": "PASSED_EXACT_EQUIVALENCE" if is_exact else "FAILED"
    }
    
    return results


if __name__ == "__main__":
    res = run_test_4_numerical_equivalence()
    print("=" * 70)
    print(res["test_name"])
    print("=" * 70)
    print(f"Configuration:           {res['config']}")
    print(f"Max abs diff outputs:    {res['max_abs_diff_outputs']:.10f}")
    print(f"Max abs diff final S:    {res['max_abs_diff_S']:.10f}")
    print(f"Max abs diff final m:    {res['max_abs_diff_m']:.10f}")
    print(f"Result Status:           {res['status']}")
    print("=" * 70)
