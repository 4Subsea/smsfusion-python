import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit  # type: ignore[misc]
def _normalize(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    L2-normalize a vector.

    Parameters
    ----------
    q : numpy.ndarray
        Vector to be normalized

    Returns
    -------
    numpy.ndarray
        Normalized copy of `q`.
    """
    return q / np.sqrt((q * q).sum())  # type: ignore[no-any-return]  # numpy funcs declare Any as return when given scalar-like


@njit  # type: ignore[misc]
def _cross(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculate the cross product of two vectors.

    Parameters
    ----------
    a, b : numpy.ndarray, shape (3,)
        Vector to cross, such that ``a x b``.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Vector result of the cross product.
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
    Unit quaternion (Schur) product: ``qa * qb``.

    Parameters
    ----------
    qa, qb : numpy.ndarray, shape (4,)
        Unit quaternions.

    Returns
    -------
    numpy.ndarray, shape (4,)
        Unit quaternions result of the product.
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


@njit  # type: ignore[misc]
def _skew_symmetric(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the cross product equivalent skew symmetric matrix.

    Parameters
    ----------
    a : numpy.ndarray, shape (3,)
        Vector in which the skew symmetric matrix is based on, such that
        ``a x b = S(a) b``.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Skew symmetric matrix.
    """
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
