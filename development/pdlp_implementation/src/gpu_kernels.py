import cupy as cp
import cupyx.scipy.sparse
from cupyx.scipy.sparse.linalg import LinearOperator


def create_linear_operators(A_eq, A_ub, c_size, q_size, dtype=cp.float32):
    """Create K and K^T operators with pre-allocated output arrays."""
    apply_K_out = cp.empty(q_size, dtype=dtype)
    apply_Kt_out = cp.empty(c_size, dtype=dtype)

    def apply_K(x, out=apply_K_out):
        """Apply K operator: [A_eq; -A_ub] * x."""
        out[:A_eq.shape[0]] = A_eq.dot(x)
        out[A_eq.shape[0]:] = -A_ub.dot(x)
        return out

    def apply_Kt(y, out=apply_Kt_out):
        """Apply K^T operator: A_eq^T * y_eq - A_ub^T * y_ub."""
        out[:] = A_eq.T.dot(y[:A_eq.shape[0]])
        out[:] -= A_ub.T.dot(y[A_eq.shape[0]:])
        return out

    K_linop = LinearOperator((q_size, c_size), matvec=apply_K, rmatvec=apply_Kt)
    return apply_K, apply_Kt, K_linop