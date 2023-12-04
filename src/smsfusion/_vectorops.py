import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit  # type: ignore[misc]
def _normalize(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    L2 normalization of a vector.
    """
    return q / np.sqrt((q * q).sum())  # type: ignore[no-any-return]  # numpy funcs declare Any as return when given scalar-like


@njit  # type: ignore[misc]
def _cross(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
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


@njit  # type: ignore[misc]
def _quaternion_product(
    qa: NDArray[np.float64], qb: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Unit quaternion (Schur) product.
    """
    qa_w, qa_xyz = np.split(qa, [1])
    qb_w, qb_xyz = np.split(qb, [1])
    return np.concatenate(
        (
            qa_w * qb_w - qa_xyz.T @ qb_xyz,
            qa_w * qb_xyz + qb_w * qa_xyz + np.cross(qa_xyz, qb_xyz),
        ),
        axis=0,
    )
