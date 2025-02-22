import cupy as cp
import cupyx.scipy.sparse
import time
from pdlp_implementation.src.gpu_kernels import apply_K_kernel, apply_Kt_kernel
from pdlp_implementation.src.utils import get_xp, proj_x, proj_y

class PDLPSolver:
    def __init__(self, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, dtype=cp.float32):
        """Initialize the PDLP solver with problem data."""
        start = time.time()

        # Move data to GPU
        self.A_eq, self.A_ub, self.b_eq, self.b_ub, self.c = self._convert_data(A_eq, A_ub, b_eq, b_ub, c, dtype)
        print(f"Took {time.time() - start:.4f} seconds to move problem data to GPU")

        # Define problem dimensions
        self.n = self.c.shape[0]
        self.K = cupyx.scipy.sparse.vstack([self.A_ub, self.A_eq])
        self.q = cp.concatenate((self.b_ub, self.b_eq))
        self.K_norm_squared = float(self.K.multiply(self.K).sum())

        # Compute optimal primal weights
        self.omega, self.nu = self._compute_optimal_weights()
        self.tau = self.nu / self.omega
        self.sigma = self.omega * self.nu

    def _convert_data(self, A_eq, A_ub, b_eq, b_ub, c, dtype):
        """Convert input matrices to appropriate GPU format."""
        A_eq = cupyx.scipy.sparse.csr_matrix(A_eq, dtype=dtype)
        A_ub = cupyx.scipy.sparse.csr_matrix(A_ub, dtype=dtype)
        return A_eq, A_ub, cp.asarray(b_eq, dtype=dtype), cp.asarray(b_ub, dtype=dtype), cp.asarray(c, dtype=dtype)

    def _compute_optimal_weights(self):
        """Find optimal omega and nu based on K_norm_squared."""
        K_norm = cp.sqrt(self.K_norm_squared)
        nu = 0.99 / K_norm  # Slightly less than 1 / ||K||_2
        omega = 1.0
        return omega, nu

    def solve(self, tol=1e-6, max_iter=1000, verbose=False):
        """Solve the linear program using PDLP."""
        x = cp.zeros(self.n, dtype=self.c.dtype)
        y = cp.zeros(self.K.shape[0], dtype=self.c.dtype)

        for k in range(max_iter):
            x_new, y_new = self._update_primal_dual(x, y)

            # Check convergence
            if k % 100 == 0 and verbose:
                self._print_diagnostics(k, x_new, y_new)

            tol_x = tol * cp.linalg.norm(x_new)
            tol_y = tol * cp.linalg.norm(y_new)
            if cp.linalg.norm(x_new - x) < tol_x and cp.linalg.norm(y_new - y) < tol_y:
                return x_new, y_new

            x, y = x_new, y_new

        print(f"Maximum number of iterations reached: {max_iter}")
        return x, y

    def _update_primal_dual(self, x, y):
        """Compute updates for x and y."""
        delta_x = self.c - apply_Kt_kernel(y, self.K)
        x_new = proj_x(x - self.tau * delta_x)

        x_bar = 2 * x_new - x
        delta_y = self.q - apply_K_kernel(x_bar, self.K)
        y_new = proj_y(y + self.sigma * delta_y, self.A_ub.shape[0])
        return x_new, y_new

    def _print_diagnostics(self, k, x, y):
        """Print diagnostics at iteration k."""
        primal_residual = cp.linalg.norm(x)
        dual_residual = cp.linalg.norm(y)
        saddle_obj = cp.dot(self.c, x) - cp.dot(y, self.K @ x) + cp.dot(self.q, y)
        print(f"Iteration {k}: Primal Residual = {primal_residual:.4e}, Dual Residual = {dual_residual:.4e}, Saddle-Point Value = {saddle_obj:.4e}")
