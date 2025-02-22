import scipy.io
import os

def save_matrix(filename, matrix):
    """Save a sparse matrix to a file in Matrix Market (.mtx) format."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    scipy.io.mmwrite(filename, matrix)
