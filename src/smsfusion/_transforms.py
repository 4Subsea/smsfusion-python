from warnings import warn

import numpy as np
from numba import njit
from numpy.typing import NDArray


def _angular_matrix_from_euler(
    alpha_beta_gamma: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Estimate angular velocity transformation matrix from Euler angles.

    Parameters
    ----------
    alpha_beta_gamma : ndarray
        Euler angle about x-axis (alpha-roll), y-axis (beta-pitch), and z-axis
        (gamma-yaw) in radians.

    Return
    ------
    T : array (Nx3x3)
        Angular velocity tansformation matrix.

    Notes
    -----
    Transform is singular for beta = +-90 degrees (gimbal lock).

    """
    alpha, beta, gamma = alpha_beta_gamma.T

    n = np.ones_like(beta)
    z = np.zeros_like(beta)

    cos_beta = np.cos(beta)
    tan_beta = np.tan(beta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    if (np.abs(np.array([cos_beta])) < 1e-8).any():
        warn("Beta is close to +-90 degrees, angular matrix  may be undefined.")

    t_00 = n
    t_01 = sin_alpha * tan_beta
    t_02 = cos_alpha * tan_beta

    t_10 = z
    t_11 = cos_alpha
    t_12 = -sin_alpha

    t_20 = z
    t_21 = sin_alpha / cos_beta
    t_22 = cos_alpha / cos_beta

    t = np.array([[t_00, t_10, t_20], [t_01, t_11, t_21], [t_02, t_12, t_22]]).T
    return t.reshape(-1, 3, 3)


@njit  # type: ignore[misc]
def _rot_matrix_from_quaternion(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert quaternion to rotation matrix.
    """
    q0, q1, q2, q3 = q

    _2q1 = q1 + q1
    _2q2 = q2 + q2
    _2q3 = q3 + q3

    _2q1q1 = q1 * _2q1
    _2q1q2 = q1 * _2q2
    _2q1q3 = q1 * _2q3
    _2q2q2 = q2 * _2q2
    _2q2q3 = q2 * _2q3
    _2q3q3 = q3 * _2q3
    _2q0q1 = q0 * _2q1
    _2q0q2 = q0 * _2q2
    _2q0q3 = q0 * _2q3

    rot_00 = 1.0 - (_2q2q2 + _2q3q3)
    rot_01 = _2q1q2 + _2q0q3
    rot_02 = _2q1q3 - _2q0q2

    rot_10 = _2q1q2 - _2q0q3
    rot_11 = 1.0 - (_2q1q1 + _2q3q3)
    rot_12 = _2q2q3 + _2q0q1

    rot_20 = _2q1q3 + _2q0q2
    rot_21 = _2q2q3 - _2q0q1
    rot_22 = 1.0 - (_2q1q1 + _2q2q2)

    rot = np.array(
        [
            [rot_00, rot_01, rot_02],
            [rot_10, rot_11, rot_12],
            [rot_20, rot_21, rot_22],
        ]
    )
    return rot


@njit  # type: ignore[misc]
def _euler_from_quaternion(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert quaternion to Euler angles (ZYX convention).
    """
    q0, q1, q2, q3 = q

    _2q1 = q1 + q1
    _2q2 = q2 + q2
    _2q3 = q3 + q3

    _2q1q1 = q1 * _2q1
    _2q1q2 = q1 * _2q2
    _2q1q3 = q1 * _2q3
    _2q2q2 = q2 * _2q2
    _2q2q3 = q2 * _2q3
    _2q3q3 = q3 * _2q3
    _2q0q1 = q0 * _2q1
    _2q0q2 = q0 * _2q2
    _2q0q3 = q0 * _2q3

    rot_00 = 1.0 - (_2q2q2 + _2q3q3)
    rot_01 = _2q1q2 + _2q0q3
    rot_02 = _2q1q3 - _2q0q2

    rot_12 = _2q2q3 + _2q0q1

    rot_22 = 1.0 - (_2q1q1 + _2q2q2)

    gamma = np.arctan2(rot_01, rot_00)
    beta = -np.arcsin(rot_02)
    alpha = np.arctan2(rot_12, rot_22)

    return np.array([alpha, beta, gamma])


@njit  # type: ignore[misc]
def _gamma_from_quaternion(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Get yaw from quaternion (ZYX convention).
    """
    q0, q1, q2, q3 = q

    _2q2 = q2 + q2
    _2q3 = q3 + q3

    _2q1q2 = q1 * _2q2
    _2q2q2 = q2 * _2q2
    _2q3q3 = q3 * _2q3
    _2q0q3 = q0 * _2q3

    rot_00 = 1.0 - (_2q2q2 + _2q3q3)
    rot_01 = _2q1q2 + _2q0q3

    yaw = np.arctan2(rot_01, rot_00)
    return yaw  # type: ignore[no-any-return]  # numpy funcs declare Any as return when given scalar-like


@njit  # type: ignore[misc]
def _angular_matrix_from_quaternion(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Angular transformation matrix, such that dq/dt = T(q) * omega.
    """
    return 0.5 * np.array(
        [
            [-q[1], -q[2], -q[3]],
            [q[0], -q[3], q[2]],
            [q[3], q[0], -q[1]],
            [-q[2], q[1], q[0]],
        ]
    )


@njit  # type: ignore[misc]
def _rot_matrix_from_euler(euler: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Rotation matrix defined from Euler angles (ZYX convention). Note that the rotation
    matrix describes the rigid body rotation from-origin-to-body, according to ZYX convention.


    Parameters
    ----------
    euler : 1D array (3,)
        Euler angle in radians given as (roll, pitch, yaw) but rotaions are applied
        according to the ZYX convention. That is, **yaw -> pitch -> roll**.

    Return
    ------
    rot : ndarray (3, 3)
        Rotation matrix.

    """
    alpha, beta, gamma = euler
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    rot_01 = cos_beta * sin_gamma
    rot_00 = cos_gamma * cos_beta
    rot_02 = -sin_beta

    rot_10 = cos_gamma * sin_beta * sin_alpha - cos_alpha * sin_gamma
    rot_11 = cos_gamma * cos_alpha + sin_gamma * sin_beta * sin_alpha
    rot_12 = cos_beta * sin_alpha

    rot_20 = cos_gamma * cos_alpha * sin_beta + sin_gamma * sin_alpha
    rot_21 = cos_alpha * sin_gamma * sin_beta - cos_gamma * sin_alpha
    rot_22 = cos_beta * cos_alpha

    rot = np.array(
        [[rot_00, rot_01, rot_02], [rot_10, rot_11, rot_12], [rot_20, rot_21, rot_22]]
    )
    return rot
