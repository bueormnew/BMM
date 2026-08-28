"""
BUEORM Delta Attention (BDA) - Validation Test 5
Spec v2: Section 7.7.5, Section 10 (Test 5)
Gradient Flow Verification: Finite gradients, no NaN/Inf across all parameters and gates.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from BDA.config import BDAConfig
from BDA.layers.bda_layer import BDALayer


def run_test_5_gradient_flow(
    B: int = 2,
    T: int = 32,
    d_model: int = 64,
    n_heads: int = 4,
    d_k: int = 16,
    d_v: int = 16,
    chunk_size: int = 8,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Spec Section 10 Test 5:
    - Verifies backward pass through complete BDA implementation.
    - Checks that all parameter gradients and input gradients are finite (no NaN or Inf).
    """
    torch.manual_seed(seed)
    
    config = BDAConfig(
        d_model=d_model,
        n_heads=n_heads,
        d_k=d_k,
        d_v=d_v,
        rank_r=4,
        chunk_size=chunk_size
    )
    
    layer = BDALayer(config)
    layer.train()
    
    x = torch.randn(B, T, d_model, requires_grad=True)
    
    # Forward pass
    output, (S_final, m_final) = layer(x, use_crpt_autograd=True)
    
    # Loss: combination of output energy and final state
    loss = output.sum() + S_final.sum() + m_final.sum()
    loss.backward()
    
    param_checks = {}
    all_finite = True
    
    for name, param in layer.named_parameters():
        if param.grad is None:
            param_checks[name] = "NO_GRAD"
            all_finite = False
        elif torch.isnan(param.grad).any():
            param_checks[name] = "CONTAINS_NAN"
            all_finite = False
        elif torch.isinf(param.grad).any():
            param_checks[name] = "CONTAINS_INF"
            all_finite = False
        else:
            grad_norm = param.grad.norm().item()
            param_checks[name] = f"FINITE (norm: {grad_norm:.4e})"
            
    input_grad_finite = (
        x.grad is not None and
        not torch.isnan(x.grad).any() and
        not torch.isinf(x.grad).any()
    )
    
    if not input_grad_finite:
        all_finite = False
        
    results = {
        "test_name": "Test 5: Gradient Flow (LRFG, DEM, ASN, SP, Projections)",
        "all_gradients_finite": all_finite,
        "input_grad_finite": input_grad_finite,
        "parameters_checked": len(param_checks),
        "param_details": param_checks,
        "status": "PASSED_FINITE_GRADIENTS" if all_finite else "FAILED"
    }
    
    return results


if __name__ == "__main__":
    res = run_test_5_gradient_flow()
    print("=" * 70)
    print(res["test_name"])
    print("=" * 70)
    print(f"All parameters finite: {res['all_gradients_finite']}")
    print(f"Input gradient finite: {res['input_grad_finite']}")
    print("Parameter gradient norms:")
    for name, status in res["param_details"].items():
        print(f"  - {name:30s}: {status}")
    print(f"Result Status:         {res['status']}")
    print("=" * 70)
