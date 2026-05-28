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


@njit  # type: ignore[misc]
def _adjugate_and_det_3_by_3(
    m: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """
    Calculates and returns the adjugate matrix and determinant of the matrix
    ```python
    m = [[a, b, c],
         [d, e, f],
         [g, h, i]]
    ```
    If the determinant is non-zero, one can calculate the inverse of m as
    ``inv(m) = adj(m) / det(m)``.

    Parameters
    ----------
    m : numpy.ndarray, shape (3, 3)
        The matrix for which to calculate the adjugate and determinant.

    Returns
    -------
    adj_m : numpy.ndarray, shape (3, 3)
        The adjugate of the input matrix m.
    det_m : float
        The determinant of the input matrix m.
    """
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    A = e * i - f * h
    B = -(d * i - f * g)
    C = d * h - e * g
    D = -(b * i - c * h)
    E = a * i - c * g
    F = -(a * h - b * g)
    G = b * f - c * e
    H = -(a * f - c * d)
    I_ = a * e - b * d
    # Equations fetched from https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_3_%C3%97_3_matrices
    adj_m = np.array([[A, D, G], [B, E, H], [C, F, I_]])
    det_m = a * A + b * B + c * C
    return adj_m, det_m
