"""
BUEORM Delta Attention (BDA) - Validation Package
"""

from BDA.validation.test_1_naive_stability import run_test_1_naive_stability
from BDA.validation.test_2_operator_norm_stability import run_test_2_operator_norm_stability
from BDA.validation.test_3_transient_growth import run_test_3_transient_growth
from BDA.validation.test_4_numerical_equivalence import run_test_4_numerical_equivalence
from BDA.validation.test_5_gradient_flow import run_test_5_gradient_flow
from BDA.validation.test_6_long_horizon_stability import run_test_6_long_horizon_stability

__all__ = [
    "run_test_1_naive_stability",
    "run_test_2_operator_norm_stability",
    "run_test_3_transient_growth",
    "run_test_4_numerical_equivalence",
    "run_test_5_gradient_flow",
    "run_test_6_long_horizon_stability",
]
