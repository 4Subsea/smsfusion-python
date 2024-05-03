import numpy as np
from numpy.typing import ArrayLike, NDArray


def calibrate(
    xyz_ref: ArrayLike, xyz: ArrayLike, bias_pre: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate the calibration values for 3-axis sensors.

    Parameters
    ----------
    xyz_ref : array-like, shape (N, 3)
        Reference values for the 3-axis sensors.
    xyz : array-like, shape (N, 3)
        Measured values from the 3-axis sensors.
    bias_pre : bool, default False
        If set to ``True``, the alternative calibration model where biases are added
        first is assumed. See Notes.

    Notes
    -----
    The calibration model is defined as::

        xyz_ref = W @ xyz + bias

    The alternative calibration model where biases are added first is defined as::

        xyz_ref = W @ (xyz + bias)

    The alternative model is enabled by setting ``bias_pre=True``.

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

    if bias_pre:
        bias = np.linalg.inv(W) @ bias
    return W, bias
