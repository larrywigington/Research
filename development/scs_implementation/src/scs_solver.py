import cupy as cp
import cupyx.scipy.sparse
import time
from scs_implementation.src.gpu_kernels import apply_A_kernel, apply_At_kernel

class SCSSolver:
    def __init__(self, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, dtype=cp.float32):
        """Initialize the SCS solver with problem data."""
        start = time.time()

        # Combine constraints
        self.b = cp.concatenate((b_eq, b_ub))
        self.h = cp.concatenate((c, self.b))
        self.c, self.b = self.h[:c.shape[0]], self.h[c.shape[0]:]

        # Move data to GPU
        self.A_eq, self.A_ub = self._convert_matrices(A_eq, A_ub, dtype)
        print(f"Took {time.time() - start:.4f} seconds to move problem data to GPU")

        # Prepare GPU buffers
        self.apply_A_out = cp.empty(self.A_eq.shape[0] + self.A_ub.shape[0], dtype=dtype)
        self.apply_At_out = cp.empty(self.c.shape[0], dtype=dtype)

    def _convert_matrices(self, A_eq, A_ub, dtype):
        """Convert input matrices to appropriate GPU format."""
        if isinstance(A_eq, cp.ndarray) and isinstance(A_ub, cp.ndarray):
            return cp.asarray(A_eq, dtype=dtype), cp.asarray(A_ub, dtype=dtype)
        elif cupyx.scipy.sparse.issparse(A_eq) and cupyx.scipy.sparse.issparse(A_ub):
            return cupyx.scipy.sparse.csr_matrix(A_eq, dtype=dtype), cupyx.scipy.sparse.csr_matrix(A_ub, dtype=dtype)
        else:
            raise ValueError("Matrix format not recognized")

    def apply_A(self, x):
        """Apply A to x using CUDA kernel."""
        return apply_A_kernel(x, self.A_eq, self.A_ub, self.apply_A_out)

    def apply_At(self, y):
        """Apply A^T to y using CUDA kernel."""
        return apply_At_kernel(y, self.A_eq, self.A_ub, self.apply_At_out)

    def solve(self, max_itr=100000, tolcheck=10, eps_pri=1e-6, eps_dual=1e-4, eps_gap=1e-4):
        """Solve the linear program using the SCS method."""
        u, v = cp.zeros(self.h.shape[0]+1, dtype=self.h.dtype), cp.zeros(self.h.shape[0]+1, dtype=self.h.dtype)
        v[-1] = 1.0
        itr = 0

        while itr < max_itr:
            itr += 1
            # Apply GPU-accelerated linear algebra operations
            utilde = self._compute_utilde(u, v)
            u, v = self._update_primal_dual(utilde, u, v)

            # Check for convergence
            if itr % tolcheck == 0 and self._check_termination(u, v, eps_pri, eps_dual, eps_gap):
                break

        return u[:self.c.shape[0]] / u[-1]

    def _compute_utilde(self, u, v):
        """Compute intermediate update step."""
        w = u + v
        rhs = w[:-1] - self.h * w[-1]
        utilde = cp.empty_like(w)
        utilde[:self.c.shape[0]], _ = cupyx.scipy.sparse.linalg.cg(self.apply_At, rhs[:self.c.shape[0]])
        utilde[self.c.shape[0]:-1] = rhs[self.c.shape[0]:] + self.apply_A(utilde[:self.c.shape[0]])
        return utilde

    def _update_primal_dual(self, utilde, u, v):
        """Update primal and dual variables."""
        u[:-1] = cp.maximum(0, utilde[:-1] - v[:-1])
        v += u - utilde
        return u, v

    def _check_termination(self, u, v, eps_pri, eps_dual, eps_gap):
        """Check stopping conditions for optimization."""
        x = u[:self.c.shape[0]] / u[-1]
        y = u[self.c.shape[0]:self.c.shape[0]+self.b.shape[0]] / u[-1]
        p_feas = cp.linalg.norm(self.apply_A(x) - self.b) / (1 + cp.linalg.norm(self.b))
        d_feas = cp.linalg.norm(self.apply_At(y) + self.c) / (1 + cp.linalg.norm(self.c))
        dual_gap = cp.abs(self.c.T @ x + self.b.T @ y) / (1 + cp.abs(self.c.T @ x) + cp.abs(self.b.T @ y))

        print(f"Iter {itr}: Primal Feasibility {p_feas:.6e}, Dual Feasibility {d_feas:.6e}, Dual Gap {dual_gap:.6e}")

        return p_feas < eps_pri and d_feas < eps_dual and dual_gap < eps_gap
