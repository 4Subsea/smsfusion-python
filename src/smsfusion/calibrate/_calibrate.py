import numpy as np
from numpy.typing import ArrayLike, NDArray


def calibrate(
    xyz_ref: ArrayLike, xyz: ArrayLike, bias_alt: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate the calibration values for 3-axis sensors.

    Parameters
    ----------
    xyz_ref : array-like, shape (N, 3)
        Reference values for the 3-axis sensors.
    xyz : array-like, shape (N, 3)
        Measured values from the 3-axis sensors.
    bias_alt : bool, default False
        If set to ``True``, the bias definition of the alternative calibration model
        is returned. See Notes.

    Notes
    -----
    The calibration model is defined as::

        xyz_ref = W @ xyz + bias

    The alternative calibration model where biases are added first is defined as::

        xyz_ref = W @ (xyz + bias)

    The alternative model is enabled by setting ``bias_alt=True``.

    In total, 12 calibration parameters are needed. Accordingly, at least 4
    measurements (of 3 data points) are required to calibrate.

    Returns
    -------
    W : numpy.ndarray, shape (3, 3)
        Calibration matrix for the 3-axis sensors.
    bias : numpy.ndarray, shape (3,)
        Bias values for the 3-axis sensors.
    """

    xyz_ref = np.asarray_chkfinite(xyz_ref)
    xyz = np.asarray_chkfinite(xyz)

    if xyz_ref.shape != xyz.shape:
        raise ValueError("xyz_ref and xyz must have the same shape.")
    elif xyz_ref.shape[-1] != 3:
        raise ValueError("xyz_ref and xyz must have shape (N, 3).")
    elif len(xyz_ref) < 4:
        raise ValueError("Too few measurements. Reqires at least 4 measurement points.")

    A_mat = np.column_stack((xyz, np.ones(len(xyz_ref))))
    x, *_ = np.linalg.lstsq(A_mat, xyz_ref, rcond=None)

    W = x[:3, :].T
    bias = x[3, :]

    if bias_alt:
        bias = np.linalg.inv(W) @ bias
    return W, bias


def decompose(
    W: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Polar decompose the calibration matrix into pure rotation and scaling.

    The polar decomposition is defined as::

        W = R @ S

    where ``R`` is pure rotation and ``S`` is scaling.

    Parameters
    ----------
    W : array-like, shape (3, 3)
        Calibration matrix.

    Returns
    -------
    R : numpy.ndarray, shape (3, 3)
        Pure rotation associated with the calibration matrix.
    S : numpy.ndarray, shape (3, 3)
        Scaling associated with the calibration matrix.
    """

    W = np.asarray_chkfinite(W)

    if W.shape != (3, 3):
        raise ValueError("Invalid shape. W must have shape (3, 3).")

    # Copied from scipy.linalg.polar
    w, s, vh = np.linalg.svd(W, full_matrices=False)
    R = w.dot(vh)
    S = (vh.T.conj() * s).dot(vh)
    return R, S
