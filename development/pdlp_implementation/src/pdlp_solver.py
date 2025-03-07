import cupy as cp
from .gpu_kernels import create_linear_operators
from .utils import (prepare_gpu_data, initialize_parameters,
                    initialize_variables, check_convergence)


def pdlp_gpu(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
             tolcheck=10, eps_pri=1e-6, eps_dual=1e-4, eps_gap=1e-4,
             eps_ubdd=1e-4, eps_infeas=1e-4, max_itr=100000, dtype=cp.float32):
    """Primal Dual Hybrid Gradient for Linear Programs on GPU."""
    # Prepare data
    c, A_ub, A_eq, q = prepare_gpu_data(c, A_ub, b_ub, A_eq, b_eq, dtype)

    # Setup linear operators
    apply_K, apply_Kt, K_linop = create_linear_operators(A_eq, A_ub, c.shape[0],
                                                         q.shape[0], dtype)

    # Initialize parameters
    eta, tau, sigma, c_norm, q_norm = initialize_parameters(c, q, K_linop, dtype)
    print("eta estimate is:", eta)

    # Initialize variables
    x, y, new_x = initialize_variables(c.shape[0], q.shape[0], dtype)

    # Main iteration
    for itr in range(max_itr):
        # Update steps
        new_x[:] = x - tau * (c - apply_Kt(y))
        y += sigma * (q - apply_K(2 * new_x - x))
        y[b_eq.shape[0]:] = cp.maximum(0, y[b_eq.shape[0]:])  # Projection
        x[:] = new_x

        # Check convergence
        if itr % tolcheck == 0:
            p_feas_gap, d_feas_gap, dual_gap = check_convergence(
                x, y, c, q, apply_K, apply_Kt, c_norm, q_norm,
                eps_pri, eps_dual, eps_gap, b_eq.shape[0]
            )
            print("| itr | primal_feas |  dual_feas  | primal/dual gap |")
            print(f"{itr:5d} {p_feas_gap:.2e} {d_feas_gap:.2e} {dual_gap:.2e}")

            if p_feas_gap < eps_pri and d_feas_gap < eps_dual and dual_gap < eps_gap:
                print("We're optimal. Terminating...")
                break

    if itr == max_itr - 1:
        print("Iteration limit hit")

    return x