import cupy as cp
import pytest
from pdlp_implementation.src.pdlp_solver import PDLPSolver

def test_solver():
    """Simple test case for the PDLP solver."""
    c = cp.array([1, 2], dtype=cp.float32)
    A_eq = cp.array([[1, 1]], dtype=cp.float32)
    b_eq = cp.array([1], dtype=cp.float32)
    solver = PDLPSolver(c, A_eq=A_eq, b_eq=b_eq)
    x, y = solver.solve(max_iter=100)
    assert x.shape == (2,)
