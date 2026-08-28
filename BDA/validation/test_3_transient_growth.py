"""
BUEORM Delta Attention (BDA) - Validation Test 3
Spec v2: Section 5.4, Section 10 (Test 3)
Verification of Transient Growth from Non-Commutativity when rho(T_t) <= 1 but ||T_t||_2 > 1.
"""

import torch
from typing import Dict, Any


def run_test_3_transient_growth(
    d_k: int = 16,
    n_trials: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Spec Section 5.4 / Section 10 Test 3:
    - Finds non-commuting transition matrices T_1, T_2 where rho(T_1) <= 1 and rho(T_2) <= 1,
      but operator norms ||T_1||_2 > 1 and ||T_2||_2 > 1.
    - Measures ||T_2 @ T_1||_2 and observes transient amplification above 1.0 (e.g. ~1.285).
    """
    torch.manual_seed(seed)
    
    alpha1 = torch.empty(n_trials, d_k, dtype=torch.float64).uniform_(0.1, 0.9)
    k_hat1 = torch.randn(n_trials, d_k, dtype=torch.float64)
    k_tilde1 = torch.randn(n_trials, d_k, dtype=torch.float64)
    beta1 = 1.0 / (torch.linalg.vector_norm(k_hat1, dim=-1) * torch.linalg.vector_norm(k_tilde1, dim=-1))
    T1 = torch.diag_embed(alpha1) - beta1.unsqueeze(-1).unsqueeze(-1) * (k_tilde1.unsqueeze(-1) * k_hat1.unsqueeze(-2))
    
    alpha2 = torch.empty(n_trials, d_k, dtype=torch.float64).uniform_(0.1, 0.9)
    k_hat2 = torch.randn(n_trials, d_k, dtype=torch.float64)
    k_tilde2 = torch.randn(n_trials, d_k, dtype=torch.float64)
    beta2 = 1.0 / (torch.linalg.vector_norm(k_hat2, dim=-1) * torch.linalg.vector_norm(k_tilde2, dim=-1))
    T2 = torch.diag_embed(alpha2) - beta2.unsqueeze(-1).unsqueeze(-1) * (k_tilde2.unsqueeze(-1) * k_hat2.unsqueeze(-2))
    
    rho1 = torch.amax(torch.abs(torch.linalg.eigvals(T1)), dim=-1)
    rho2 = torch.amax(torch.abs(torch.linalg.eigvals(T2)), dim=-1)
    
    valid_mask = (rho1 <= 1.0001) & (rho2 <= 1.0001)
    
    T_comp = torch.matmul(T2, T1)
    norm_comp = torch.linalg.matrix_norm(T_comp, ord=2)
    
    valid_norm_comp = norm_comp[valid_mask]
    max_composite_norm = float(torch.max(valid_norm_comp).item()) if len(valid_norm_comp) > 0 else 0.0
    
    best_idx = torch.argmax(norm_comp * valid_mask.float()).item()
    best_example = {
        "rho_1": float(rho1[best_idx].item()),
        "rho_2": float(rho2[best_idx].item()),
        "norm_1": float(torch.linalg.matrix_norm(T1[best_idx], ord=2).item()),
        "norm_2": float(torch.linalg.matrix_norm(T2[best_idx], ord=2).item()),
        "norm_comp": float(norm_comp[best_idx].item()),
    }
    
    results = {
        "test_name": "Test 3: Transient Growth with Step Spectral Radius <= 1",
        "d_k": d_k,
        "n_trials": n_trials,
        "max_composite_norm_observed": max_composite_norm,
        "best_example": best_example,
        "amplification_observed": max_composite_norm > 1.0,
        "status": "CONFIRMED_TRANSIENT_GROWTH" if max_composite_norm > 1.0 else "UNEXPECTED"
    }
    
    return results


if __name__ == "__main__":
    res = run_test_3_transient_growth()
    print("=" * 70)
    print(res["test_name"])
    print("=" * 70)
    print(f"Max ||T2 @ T1||_2 observed: {res['max_composite_norm_observed']:.4f}")
    if res["best_example"]:
        ex = res["best_example"]
        print(f"Example - rho(T1): {ex['rho_1']:.4f}, rho(T2): {ex['rho_2']:.4f}")
        print(f"Example - ||T1||:  {ex['norm_1']:.4f}, ||T2||:  {ex['norm_2']:.4f}")
        print(f"Example - ||T2 T1||: {ex['norm_comp']:.4f} (> 1.0 confirms non-commutative peak)")
    print(f"Result Status:              {res['status']}")
    print("=" * 70)
