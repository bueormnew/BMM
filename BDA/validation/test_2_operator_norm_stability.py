"""
BUEORM Delta Attention (BDA) - Validation Test 2
Spec v2: Section 5.3, Section 10 (Test 2)
Verification of Proven Operator Norm Stability Condition: max(alpha) + beta * ||k_tilde|| * ||k_hat|| <= 1.
"""

import torch
from typing import Dict, Any


def run_test_2_operator_norm_stability(
    n_samples: int = 50000,
    d_k: int = 16,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Spec Section 5.3 / Section 10 Test 2:
    - Generates n_samples random samples of:
        alpha_t ~ Uniform(0.05, 0.99)^{d_k}
        k_hat_t ~ Normal(0, 1)^{d_k}
        k_tilde_t ~ Normal(0, 1)^{d_k}
    - Fixes beta_t = (1.0 - max(alpha_t)) / (||k_tilde_t|| * ||k_hat_t||) exactly.
    - Constructs T_t = diag(alpha_t) - beta_t * (k_tilde_t (x) k_hat_t^T).
    - Computes eigenvalues of T_t and verifies that spectral radius |lambda|_max <= 1.0 always.
    """
    torch.manual_seed(seed)
    
    alpha = torch.empty(n_samples, d_k, dtype=torch.float64).uniform_(0.05, 0.99)
    k_hat = torch.randn(n_samples, d_k, dtype=torch.float64)
    k_tilde = torch.randn(n_samples, d_k, dtype=torch.float64)
    
    norm_k_hat = torch.linalg.vector_norm(k_hat, dim=-1)
    norm_k_tilde = torch.linalg.vector_norm(k_tilde, dim=-1)
    alpha_max = torch.amax(alpha, dim=-1)
    
    # Exact boundary of proven operator norm condition
    beta = (1.0 - alpha_max) / (norm_k_hat * norm_k_tilde)
    
    outer = k_tilde.unsqueeze(-1) * k_hat.unsqueeze(-2)
    T = torch.diag_embed(alpha) - beta.unsqueeze(-1).unsqueeze(-1) * outer
    
    eigenvalues = torch.linalg.eigvals(T)
    max_abs_eig = torch.amax(torch.abs(eigenvalues), dim=-1)
    op_norms = torch.linalg.matrix_norm(T, ord=2)
    
    violations_mask = max_abs_eig > (1.0 + 1e-7)
    violations = int(violations_mask.sum().item())
    max_lambda_observed = float(torch.max(max_abs_eig).item())
    max_operator_norm_observed = float(torch.max(op_norms).item())
    
    results = {
        "test_name": "Test 2: Operator Norm Condition (max(alpha) + beta * ||k_tilde|| * ||k_hat|| <= 1)",
        "n_samples": n_samples,
        "d_k": d_k,
        "violations": violations,
        "violation_rate_pct": (violations / n_samples) * 100.0,
        "max_lambda_observed": max_lambda_observed,
        "max_operator_norm_observed": max_operator_norm_observed,
        "status": "PASSED_ZERO_VIOLATIONS" if violations == 0 else "FAILED"
    }
    
    return results


if __name__ == "__main__":
    res = run_test_2_operator_norm_stability()
    print("=" * 70)
    print(res["test_name"])
    print("=" * 70)
    print(f"Total samples:       {res['n_samples']}")
    print(f"Violations (|λ| > 1): {res['violations']} ({res['violation_rate_pct']:.4f}%)")
    print(f"Max |λ| observed:    {res['max_lambda_observed']:.6f}")
    print(f"Max ||T||_2 observed: {res['max_operator_norm_observed']:.6f}")
    print(f"Result Status:       {res['status']}")
    print("=" * 70)
