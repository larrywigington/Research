import cupy as cp

def get_xp(*args):
    """Determine the array module (NumPy or CuPy) based on the input arrays."""
    return cp.get_array_module(*args)

def proj_x(x, x_min=-cp.inf, x_max=cp.inf):
    """Projection function for primal variable x."""
    return cp.clip(x, a_min=x_min, a_max=x_max)

def proj_y(y, m):
    """Projection function for dual variable y."""
    y[:m] = cp.maximum(y[:m], 0)
    return y
