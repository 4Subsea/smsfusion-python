"""
Collection of elemantary functions that are Numba compiled for performance.

General rule of thumb, it that every function in this module should have a public
counterpart in ``sensor_4s``. These functions are private performance optimized
and inteded to be used where NumPy vectorization is not possible.
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def _normalize(q):
    """
    L2 normalization of a vector.
    """
    return q / np.sqrt(np.dot(q, q))


@jit(nopython=True)
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


@jit(nopython=True)
def _rot_matrix_from_quaternion(q):
    """
    Convert quaternion to rotation matrix. From origin-to-body (ZYX convention).
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


@jit(nopython=True)
def _euler_from_quaternion(q):
    """
    Convert quaternion to rotation matrix. From origin-to-body (ZYX convention).
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

    return alpha, beta, gamma


@jit(nopython=False)
def _gamma_from_quaternion(q):
    """
    Calculate yaw from quaternion. From origin-to-body (ZYX convention).
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


@jit(nopython=True)
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


@jit(nopython=True)
def _quaternion_from_rot_matrix(rot_matrix):
    """
    Quaternion from rotation matrix.

    Note: Rotation matrix is origin-to-body (ZYX convention).

    Parameters
    ----------
    rot_matrix : 3x3 array
        Rotation matrix.

    Return
    ------
    q : 1D array
        Quaternion given as array.
    """
    rot_00 = rot_matrix[0, 0]
    rot_01 = rot_matrix[0, 1]
    rot_02 = rot_matrix[0, 2]
    rot_10 = rot_matrix[1, 0]
    rot_11 = rot_matrix[1, 1]
    rot_12 = rot_matrix[1, 2]
    rot_20 = rot_matrix[2, 0]
    rot_21 = rot_matrix[2, 1]
    rot_22 = rot_matrix[2, 2]

    trace = rot_00 + rot_11 + rot_22

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)

        q0 = 0.25 / s
        q1 = (rot_12 - rot_21) * s
        q2 = (rot_20 - rot_02) * s
        q3 = (rot_01 - rot_10) * s

    elif rot_00 > rot_11 and rot_00 > rot_22:
        s = 2.0 * np.sqrt(1.0 + rot_00 - rot_11 - rot_22)

        q0 = (rot_12 - rot_21) / s
        q1 = 0.25 * s
        q2 = (rot_10 + rot_01) / s
        q3 = (rot_20 + rot_02) / s

    elif rot_11 > rot_22:
        s = 2.0 * np.sqrt(1.0 + rot_11 - rot_00 - rot_22)

        q0 = (rot_20 - rot_02) / s
        q1 = (rot_10 + rot_01) / s
        q2 = 0.25 * s
        q3 = (rot_21 + rot_12) / s

    else:
        s = 2.0 * np.sqrt(1.0 + rot_22 - rot_00 - rot_11)

        q0 = (rot_01 - rot_10) / s
        q1 = (rot_20 + rot_02) / s
        q2 = (rot_21 + rot_12) / s
        q3 = 0.25 * s
    return np.array([q0, q1, q2, q3])


def _rot_matrix_from_euler(alpha, beta, gamma):
    """
    Rotation matrix defined from Euler angles. The rotation matrix describes
    rigid body rotation from-origin-to-body, according to ZYX convention. That
    is, first rotate about the z-axis (gamma - yaw), then about the y-axis
    (beta - pitch), and lastly about the x-axis (alpha - roll).

    Parameters
    ----------
    alpha : float
        Euler angle about x-axis (alpha-roll) in radians.
    beta : float
        Euler angle about y-axis (beta-pitch) in radians.
    gamma : float
        Euler angle about z-axis (gamma-yaw) in radians.

    Return
    ------
    rot : array (3x3)
        3D rotation matrix.

    """
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
