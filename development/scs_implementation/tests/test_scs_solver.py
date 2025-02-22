import cupy as cp
import pytest
from scs_implementation.src.scs_solver import SCSSolver

def test_solver():
    """Simple test case for the SCS solver."""
    c = cp.array([1, 2], dtype=cp.float32)
    A_eq = cp.array([[1, 1]], dtype=cp.float32)
    b_eq = cp.array([1], dtype=cp.float32)
    solver = SCSSolver(c, A_eq=A_eq, b_eq=b_eq)
    x = solver.solve(max_itr=100)
    assert x.shape == (2,)
