import cupy as cp
import numpy as np
import scipy.sparse
import cupyx.scipy.sparse
import time


def prepare_gpu_data(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, dtype=cp.float32):
    """Prepare and transfer problem data to GPU."""
    start = time.time()

    if A_ub is not None and A_eq is not None:
        if isinstance(A_ub, np.ndarray) and isinstance(A_eq, np.ndarray):
            A_ub_gpu = cp.asarray(A_ub, dtype=dtype)
            A_eq_gpu = cp.asarray(A_eq, dtype=dtype)
        elif scipy.sparse.issparse(A_ub) and scipy.sparse.issparse(A_eq):
            A_ub_gpu = cupyx.scipy.sparse.csr_matrix(A_ub, dtype=dtype)
            A_eq_gpu = cupyx.scipy.sparse.csr_matrix(A_eq, dtype=dtype)
        else:
            raise ValueError("Matrix format not recognized")
    else:
        raise ValueError("A_ub and A_eq must be provided")

    q = np.concatenate((b_eq, -b_ub))
    q_gpu = cp.asarray(q, dtype=dtype)
    c_gpu = cp.asarray(c, dtype=dtype)

    print(f"Took {time.time() - start} seconds to move problem data to GPU")
    return c_gpu, A_ub_gpu, A_eq_gpu, q_gpu


def initialize_parameters(c, q, K_linop, dtype=cp.float32):
    """Initialize algorithm parameters including eta estimation."""
    c_norm = cp.linalg.norm(c)
    q_norm = cp.linalg.norm(q)

    cp.random.seed(0)  # For reproducibility
    eta = 0.9 / estimate_spectral_norm(K_linop, dtype=dtype)
    omega = 1.0
    tau = eta / omega
    sigma = omega * eta

    return eta, tau, sigma, c_norm, q_norm


def initialize_variables(c_size, q_size, dtype=cp.float32):
    """Initialize primal and dual variables."""
    x = cp.zeros(c_size, dtype=dtype)
    y = cp.zeros(q_size, dtype=dtype)
    new_x = cp.zeros(c_size, dtype=dtype)
    return x, y, new_x


def check_convergence(x, y, c, q, apply_K, apply_Kt, c_norm, q_norm,
                      eps_pri=1e-6, eps_dual=1e-4, eps_gap=1e-4, b_eq_size=None):
    """Check convergence conditions."""
    p_feas = q - apply_K(x)
    p_feas[b_eq_size:] = cp.maximum(p_feas[b_eq_size:], 0.)
    p_feas_gap = cp.linalg.norm(p_feas) / (1 + q_norm)

    d_feas_gap = cp.linalg.norm(c - apply_Kt(y)) / (1 + c_norm)
    dual_gap = cp.abs(q.T @ y - c.T @ x) / (1 + cp.abs(c.T @ x) + cp.abs(q.T @ y))

    return p_feas_gap, d_feas_gap, dual_gap


def estimate_spectral_norm(A, its=20, dtype=cp.float32):
    """Estimate the spectral norm of a linear operator."""
    v = cp.empty(A.shape[1], dtype=dtype)
    u = cp.empty(A.shape[0], dtype=dtype)
    v[:] = cp.random.uniform(low=-1., high=1., size=A.shape[1])
    v /= cp.linalg.norm(v)
    for j in range(its):
        u = A.matvec(v)
        v = A.rmatvec(u)
        snorm = cp.linalg.norm(v)
        if snorm > 0:
            v /= snorm
        snorm = cp.sqrt(snorm)
    return snorm