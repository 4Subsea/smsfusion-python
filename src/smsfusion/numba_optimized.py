"""
Collection of elemantary functions that are Numba compiled for performance.

General rule of thumb, it that every function in this module should have a public
counterpart in ``sensor_4s``. These functions are private performance optimized
and inteded to be used where NumPy vectorization is not possible.
"""

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


@njit
def _rot_matrix_from_quaternion(q):
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


@njit
def _euler_from_quaternion(q):
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


@njit
def _gamma_from_quaternion(q):
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
    return yaw


@njit
def _angular_matrix_from_quaternion(q):
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


@njit
def _rot_matrix_from_euler(euler):
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
