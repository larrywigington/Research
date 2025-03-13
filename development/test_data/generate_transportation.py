import numpy as np
import scipy.sparse
import scipy.io
import os
from utils import save_matrix


def generate_transportation_problem(m, n, write=False, data_dir="data"):
    """Generate a feasible transportation problem with random data.

    Args:
        m (int): Number of supply nodes.
        n (int): Number of demand nodes.
        write (bool): If True, saves generated data.
        data_dir (str): Directory to store the test data files.

    Returns:
        dict: A dictionary containing c, A_ub, b_ub, A_eq, b_eq.
    """
    np.random.seed(0)

    # Cost vector
    c = np.random.uniform(size=m * n)

    # Supply and demand constraints
    s = np.random.exponential(size=m)
    d = np.random.exponential(size=n)
    d *= (s.sum() - 1e-1) / d.sum()  # Normalize supply and demand

    # Constraint matrices
    supply_cons = scipy.sparse.dok_matrix((m, m * n))
    for i in range(m):
        for j in range(n):
            supply_cons[i, j + i * m] = 1.0

    demand_cons = scipy.sparse.dok_matrix((n, m * n))
    for i in range(m):
        for j in range(n):
            demand_cons[i, j * n + i] = -1.0

    nneg_cons = -scipy.sparse.eye(m * n, format='csr')
    A_ub = scipy.sparse.vstack((supply_cons.tocsr(), demand_cons.tocsr(), nneg_cons))
    b_ub = np.concatenate((s, -d, np.zeros(m * n)))

    # Ensure well-defined system by fixing one variable
    A_eq = scipy.sparse.dok_matrix((1, m * n))
    A_eq[0, 0] = 1.0
    A_eq = scipy.sparse.csr_matrix(A_eq)
    b_eq = np.array([0.0])

    # Save data if requested
    if write:
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, "c.npy"), c)
        np.save(os.path.join(data_dir, "b_eq.npy"), b_eq)
        np.save(os.path.join(data_dir, "b_ub.npy"), b_ub)
        save_matrix(os.path.join(data_dir, "A_eq.mtx"), A_eq)
        save_matrix(os.path.join(data_dir, "A_ub.mtx"), A_ub)

    return {"c": c, "A_ub": A_ub, "b_ub": b_ub, "A_eq": A_eq, "b_eq": b_eq}


if __name__ == "__main__":
    generate_transportation_problem(100, 100, write=True)
