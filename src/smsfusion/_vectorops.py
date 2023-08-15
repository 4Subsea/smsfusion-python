import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit
def _normalize(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    L2 normalization of a vector.
    """
    return q / np.sqrt((q * q).sum())


@njit
def _cross(a, b):
    """
    Cross product of two vectors.
    """
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )
