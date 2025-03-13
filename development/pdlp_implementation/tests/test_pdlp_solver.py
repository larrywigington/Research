import cupy as cp
import pytest
from Research.development.pdlp_implementation.src.pdlp.pdlp_solver import pdlp_gpu
from Research.development.test_data.utils import load_test_data


def test_solver_with_data():
    """Test PDLP solver with pre-loaded test data."""
    # Load test data
    try:
        c, A_eq, b_eq, A_ub, b_ub = load_test_data()
    except FileNotFoundError as e:
        pytest.skip(f"Skipping test due to missing data: {e}")

    # Run solver
    x = pdlp_gpu(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                 max_itr=100, tolcheck=10, eps_pri=1e-4, eps_dual=1e-4, eps_gap=1e-4)

    # Basic checks
    assert x.shape == (c.shape[0],), f"Expected shape ({c.shape[0],}), got {x.shape}"

    # Placeholder for specific assertions (add if you have expected results)
    # x_np = x.get()
    # assert np.allclose(x_np, expected_x, atol=1e-3), f"Expected {expected_x}, got {x_np}"