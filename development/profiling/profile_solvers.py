import numpy as np
import cupy as cp
import time
import scipy.io
import tracemalloc
from development.scs_implementation.src.scs_solver import SCSSolver
from development.pdlp_implementation.src.pdlp_solver import PDLPSolver
from development.scs_implementation.tests.baseline_solver import linprog_baseline


# Load problem data
def load_problem_data(data_dir="development/test_data/data"):
    """Load test problem data from disk."""
    c = np.load(f"{data_dir}/c.npy")
    b_eq = np.load(f"{data_dir}/b_eq.npy")
    b_ub = np.load(f"{data_dir}/b_ub.npy")
    A_eq = scipy.io.mmread(f"{data_dir}/A_eq.mtx")
    A_ub = scipy.io.mmread(f"{data_dir}/A_ub.mtx")
    return c, A_eq, b_eq, A_ub, b_ub


# Profiling function
def profile_solver(solver_func, solver_name, *args, **kwargs):
    """Profiles execution time and memory usage of a solver."""
    tracemalloc.start()
    start_time = time.time()

    result = solver_func(*args, **kwargs)

    elapsed_time = time.time() - start_time
    memory_used = tracemalloc.get_traced_memory()[1] / (1024 ** 2)  # MB
    tracemalloc.stop()

    print(f"\n--- Profiling {solver_name} ---")
    print(f"Execution Time: {elapsed_time:.6f} sec")
    print(f"Memory Usage: {memory_used:.2f} MB")

    return result


# Run profiling for all solvers
def run_profiling():
    """Profile all solvers and compare performance."""
    c, A_eq, b_eq, A_ub, b_ub = load_problem_data()

    # Baseline: SciPy's linprog
    x_baseline = profile_solver(linprog_baseline, "SciPy linprog",
                                c, A_ub, b_ub, A_eq, b_eq)

    # SCS Solver
    scs_solver = SCSSolver(cp.asarray(c), A_ub=cp.asarray(A_ub), b_ub=cp.asarray(b_ub),
                           A_eq=cp.asarray(A_eq), b_eq=cp.asarray(b_eq))
    x_scs = profile_solver(scs_solver.solve, "SCS Solver", max_itr=1000)

    # PDLP Solver
    pdlp_solver = PDLPSolver(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
    x_pdlp, y_pdlp = profile_solver(pdlp_solver.solve, "PDLP Solver", max_iter=1000)


if __name__ == "__main__":
    run_profiling()
