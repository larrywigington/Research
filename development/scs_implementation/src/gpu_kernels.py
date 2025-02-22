import cupy as cp

def apply_A_kernel(x, A_eq, A_ub, out):
    """CUDA-accelerated function to compute A @ x."""
    out[:A_eq.shape[0]] = A_eq.dot(x)
    out[A_eq.shape[0]:] = A_ub.dot(x)
    return out

def apply_At_kernel(y, A_eq, A_ub, out):
    """CUDA-accelerated function to compute A^T @ y."""
    out[:] = A_eq.T.dot(y[:A_eq.shape[0]]) + A_ub.T.dot(y[A_eq.shape[0]:])
    return out
