from typing import Self

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._ins import _dhda_head, _h_head, _signed_smallest_angle
from ._transforms import _angular_matrix_from_quaternion as T
from ._transforms import _euler_from_quaternion
from ._vectorops import _normalize, _skew_symmetric


def _state_transition(
    dt: float, f_b: NDArray[np.float64], w_b: NDArray[np.float64], R_nb: NDArray[np.float64], gbc: float
) -> NDArray[np.float64]:
    """
    State transition matrix.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    f_b : ndarray, shape (3,)
        Specific force measurement in body frame.
    w_b : ndarray, shape (3,)
        Angular rate measurement (bias corrected) in body frame.
    R_nb : ndarray, shape (3, 3)
        Rotation matrix from body to navigation frame.
    gbc : float
        Gyro bias correlation time in seconds.

    Returns
    -------
    ndarray, shape (9, 9)
        State transition matrix.
    """
    phi = np.eye(9)
    phi[0:3, 0:3] -= dt * _skew_symmetric(w_b)  # NB! update each time step
    phi[0:3, 3:6] -= dt * np.eye(3)
    phi[3:6, 3:6] -= dt * np.eye(3) / gbc
    phi[6:9, 0:3] -= dt * R_nb @ _skew_symmetric(f_b)  # NB! update each time step
    return phi


@njit  # type: ignore[misc]
def _update_state_transition(
    phi: NDArray[np.float64],
    dt: float,
    f_b: NDArray[np.float64],
    w_b: NDArray[np.float64],
    R_nb: NDArray[np.float64],
) -> None:
    """
    Update the state transition matrix in place.

    Parameters
    ----------
    phi : ndarray, shape (9, 9)
        State transition matrix to be updated in place.
    dt : float
        Time step.
    f_b : ndarray, shape (3,)
        Specific force measurement in body frame.
    w_b : ndarray, shape (3,)
        Angular rate measurement (bias corrected) in body frame.
    R_nb : ndarray, shape (3, 3)
        Rotation matrix from body to navigation frame.
    """
    wx, wy, wz = w_b
    fx, fy, fz = f_b

    r00, r01, r02 = R_nb[0]
    r10, r11, r12 = R_nb[1]
    r20, r21, r22 = R_nb[2]

    # phi[0:3, 0:3] = np.eye(3) - dt * S(w_b)
    phi[0, 1] = dt * wz
    phi[0, 2] = -dt * wy
    phi[1, 0] = -dt * wz
    phi[1, 2] = dt * wx
    phi[2, 0] = dt * wy
    phi[2, 1] = -dt * wx

    # phi[6:9, 0:3] = -dt * R_nb @ S(f_b)
    phi[6, 0] = -dt * (fz * r01 - fy * r02)
    phi[7, 0] = -dt * (fz * r11 - fy * r12)
    phi[8, 0] = -dt * (fz * r21 - fy * r22)
    phi[6, 1] = -dt * (-fz * r00 + fx * r02)
    phi[7, 1] = -dt * (-fz * r10 + fx * r12)
    phi[8, 1] = -dt * (-fz * r20 + fx * r22)
    phi[6, 2] = -dt * (fy * r00 - fx * r01)
    phi[7, 2] = -dt * (fy * r10 - fx * r11)
    phi[8, 2] = -dt * (fy * r20 - fx * r21)


def _process_noise_cov(
    dt: float, arw: float, gbs: float, gbc: float
) -> NDArray[np.float64]:
    """
    Process noise covariance matrix.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    arw : float
        Angular random walk (gyroscope noise density) in rad/√Hz.
    gbs : float
        Gyro bias stability (bias instability) in rad/s.
    gbc : float
        Gyro bias correlation time in seconds.

    Returns
    -------
    Q : ndarray, shape (6, 6)
        Process noise covariance matrix.
    """
    Q = np.zeros((6, 6))
    Q[0:3, 0:3] = dt * arw**2 * np.eye(3)
    Q[3:6, 3:6] = dt * (2.0 * gbs**2 / gbc) * np.eye(3)
    return Q


def _measurement_matrix(
    q_nb: NDArray[np.float64], vg_b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Measurement matrix.

    Parameters
    ----------
    q_nb : ndarray, shape (4,)
        Unit quaternion.
    vg_b : ndarray, shape (3,)
        Gravity reference unit vector expressed in the body frame.

    Returns
    -------
    ndarray, shape (4, 6)
        Linearized measurement matrix.
    """
    dhdx = np.zeros((4, 6))
    dhdx[0:3, 0:3] = _skew_symmetric(vg_b)  # gravity ref vector
    dhdx[3:4, 0:3] = _dhda_head(q_nb)  # heading
    return dhdx

