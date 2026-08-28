"""
BUEORM Delta Attention (BDA) - Validation Test 1
Spec v2: Section 5.2, Section 10 (Test 1)
Verification that the naive stability condition beta * ||k_tilde|| * ||k_hat|| <= 1 FAILS.
"""

import torch
from typing import Dict, Any


def run_test_1_naive_stability(
    n_samples: int = 20000,
    d_k: int = 16,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Spec Section 5.2 / Section 10 Test 1:
    - Generates n_samples random samples of:
        alpha_t ~ Uniform(0.05, 0.99)^{d_k}
        k_hat_t ~ Normal(0, 1)^{d_k}
        k_tilde_t ~ Normal(0, 1)^{d_k} (independent, non-collinear)
    - Fixes beta_t = 1.0 / (||k_tilde_t|| * ||k_hat_t||) exactly.
    - Constructs T_t = diag(alpha_t) - beta_t * (k_tilde_t (x) k_hat_t^T).
    - Computes eigenvalues of T_t and counts spectral radius violations (|lambda|_max > 1.0).
    """
    torch.manual_seed(seed)
    
    alpha = torch.empty(n_samples, d_k, dtype=torch.float64).uniform_(0.05, 0.99)
    k_hat = torch.randn(n_samples, d_k, dtype=torch.float64)
    k_tilde = torch.randn(n_samples, d_k, dtype=torch.float64)
    
    norm_k_hat = torch.linalg.vector_norm(k_hat, dim=-1)
    norm_k_tilde = torch.linalg.vector_norm(k_tilde, dim=-1)
    
    beta = 1.0 / (norm_k_hat * norm_k_tilde)
    
    # Outer product (n_samples, d_k, d_k)
    outer = k_tilde.unsqueeze(-1) * k_hat.unsqueeze(-2)
    T = torch.diag_embed(alpha) - beta.unsqueeze(-1).unsqueeze(-1) * outer
    
    # Eigenvalues across all samples simultaneously
    eigenvalues = torch.linalg.eigvals(T)
    max_abs_eig = torch.amax(torch.abs(eigenvalues), dim=-1)
    
    violations_mask = max_abs_eig > (1.0 + 1e-7)
    violations = int(violations_mask.sum().item())
    max_lambda_observed = float(torch.max(max_abs_eig).item())
    violation_rate = (violations / n_samples) * 100.0
    
    results = {
        "test_name": "Test 1: Naive Stability Condition (beta * ||k_tilde|| * ||k_hat|| <= 1)",
        "n_samples": n_samples,
        "d_k": d_k,
        "violations": violations,
        "violation_rate_pct": violation_rate,
        "max_lambda_observed": max_lambda_observed,
        "status": "CONFIRMED_FAILED_AS_PREDICTED" if violations > 0 else "UNEXPECTED"
    }
    
    return results


if __name__ == "__main__":
    res = run_test_1_naive_stability()
    print("=" * 70)
    print(res["test_name"])
    print("=" * 70)
    print(f"Total samples:       {res['n_samples']}")
    print(f"Violations (|λ| > 1): {res['violations']} ({res['violation_rate_pct']:.2f}%)")
    print(f"Max |λ| observed:    {res['max_lambda_observed']:.4f}")
    print(f"Result Status:       {res['status']}")
    print("=" * 70)
