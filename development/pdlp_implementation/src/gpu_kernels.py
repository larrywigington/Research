import cupy as cp

def apply_K_kernel(x, K):
    """CUDA-accelerated function to compute K @ x."""
    return K @ x

def apply_Kt_kernel(y, K):
    """CUDA-accelerated function to compute K^T @ y."""
    return K.T @ y
