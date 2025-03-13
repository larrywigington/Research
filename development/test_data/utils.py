import cupy as cp
import numpy as np
from scipy.io import mmwrite
import os

def save_matrix(filename, matrix):
    """Save a sparse matrix to a file in Matrix Market (.mtx) format."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    mmwrite(filename, matrix)


# Define the test data directory relative to this file
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_test_data(data_dir=None):
    """
    Load test data from the specified directory or default test_data/data/.

    Args:
        data_dir (str, optional): Path to the data directory. Defaults to TEST_DATA_DIR.

    Returns:
        tuple: (c, A_eq, b_eq, A_ub, b_ub) as CuPy arrays.

    Raises:
        FileNotFoundError: If any test data file is missing.
    """
    data_path = data_dir if data_dir is not None else TEST_DATA_DIR

    try:
        c = np.load(os.path.join(data_path, "c.npy"))
        A_eq = np.load(os.path.join(data_path, "A_eq.npy"))
        b_eq = np.load(os.path.join(data_path, "b_eq.npy"))
        A_ub = np.load(os.path.join(data_path, "A_ub.npy"))
        b_ub = np.load(os.path.join(data_path, "b_ub.npy"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Test data file not found: {e}")

    # Convert to CuPy arrays
    return (cp.array(c, dtype=cp.float32), cp.array(A_eq, dtype=cp.float32),
            cp.array(b_eq, dtype=cp.float32), cp.array(A_ub, dtype=cp.float32),
            cp.array(b_ub, dtype=cp.float32))