from warnings import warn

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from smsfusion._vectorops import _normalize


def _angular_matrix_from_euler(
    euler: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute the angular velocity transformation matrix, **T**, from Euler angles.


    Parameters
    ----------
    euler : numpy.ndarray, shape (3,)
        Vector of Euler angles in radians (ZYX convention). Contains the following
        three Euler angles in order:
            - Roll (alpha): Rotation about the x-axis.
            - Pitch (beta): Rotation about the y-axis.
            - Yaw (gamma): Rotation about the z-axis.

    Return
    ------
    numpy.ndarray, shape (3, 3)
        Angular velocity tansformation matrix.

    Notes
    -----
    Transform is singular for beta = +-90 degrees (gimbal lock).
    """
    alpha, beta, _ = euler

    cos_beta = np.cos(beta)
    tan_beta = np.tan(beta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    if np.abs(cos_beta) < 1e-8:
        warn("Beta is close to +-90 degrees, angular matrix may be undefined.")

    t_00 = 1.0
    t_01 = sin_alpha * tan_beta
    t_02 = cos_alpha * tan_beta

    t_10 = 0.0
    t_11 = cos_alpha
    t_12 = -sin_alpha

    t_20 = 0.0
    t_21 = sin_alpha / cos_beta
    t_22 = cos_alpha / cos_beta

    T = np.array([[t_00, t_01, t_02], [t_10, t_11, t_12], [t_20, t_21, t_22]])
    return T


def _inv_angular_matrix_from_euler(euler: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the inverse angular velocity transformation matrix, **T**^-1, from Euler angles.

    Parameters
    ----------
    euler : numpy.ndarray, shape (3,)
        Vector of Euler angles in radians (ZYX convention). Contains the following
        three Euler angles in order:
            - Roll (alpha): Rotation about the x-axis.
            - Pitch (beta): Rotation about the y-axis.
            - Yaw (gamma): Rotation about the z-axis.

    Return
    ------
    numpy.ndarray, shape (3, 3)
        Inverse angular velocity tansformation matrix.
    """
    alpha, beta, _ = euler

    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    t_00 = 1.0
    t_01 = 0.0
    t_02 = -sin_beta

    t_10 = 0.0
    t_11 = cos_alpha
    t_12 = cos_beta * sin_alpha

    t_20 = 0.0
    t_21 = -sin_alpha
    t_22 = cos_beta * cos_alpha

    T_inv = np.array([[t_00, t_01, t_02], [t_10, t_11, t_12], [t_20, t_21, t_22]])
    return T_inv


@njit  # type: ignore[misc]
def _rot_matrix_from_quaternion(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the rotation matrix from a unit quaternion.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Unit quaternion.

    Returns
    -------
    rot : numpy.ndarray, shape (3, 3)
        Rotation matrix.
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
    rot_01 = _2q1q2 - _2q0q3
    rot_02 = _2q1q3 + _2q0q2

    rot_10 = _2q1q2 + _2q0q3
    rot_11 = 1.0 - (_2q1q1 + _2q3q3)
    rot_12 = _2q2q3 - _2q0q1

    rot_20 = _2q1q3 - _2q0q2
    rot_21 = _2q2q3 + _2q0q1
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
    Compute the Euler angles (ZYX convention) from a unit quaternion.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Unit quaternion (representing transformation from-body-to-origin).

    Returns
    -------
    numpy.ndarray, shape (3,)
        Vector of Euler angles in radians (ZYX convention). Contains the following
        three Euler angles in order:
            - Roll (alpha): Rotation about the x-axis.
            - Pitch (beta): Rotation about the y-axis.
            - Yaw (gamma): Rotation about the z-axis.
    """
    q_w, q_x, q_y, q_z = q

    alpha = np.arctan2(2.0 * (q_y * q_z + q_x * q_w), 1.0 - 2.0 * (q_x**2 + q_y**2))
    beta = -np.arcsin(2.0 * (q_x * q_z - q_y * q_w))
    gamma = np.arctan2(2.0 * (q_x * q_y + q_z * q_w), 1.0 - 2.0 * (q_y**2 + q_z**2))

    return np.array([alpha, beta, gamma])


@njit  # type: ignore[misc]
def _gamma_from_quaternion(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the yaw Euler angle (ZYX convention) from a unit quaternion.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Unit quaternion (representing transformation from-body-to-origin).

    Returns
    -------
    float
        Yaw (gamma) Euler angle (ZYX convention).
    """
    q_w, q_x, q_y, q_z = q

    gamma = np.arctan2(2.0 * (q_x * q_y + q_z * q_w), 1.0 - 2.0 * (q_y**2 + q_z**2))

    return gamma  # type: ignore[no-any-return]  # numpy funcs declare Any as return when given scalar-like


@njit  # type: ignore[misc]
def _angular_matrix_from_quaternion(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute angular transformation matrix, **T**, from a unit quaternion.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Unit quaternion.

    Returns
    -------
    numpy.ndarray, shape (4, 3)
        Angular transformation matrix.
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
    Compute the rotation matrix (from-body-to-origin) from Euler angles.


    Parameters
    ----------
    euler : numpy.ndarray, shape (3,)
        Vector of Euler angles in radians (ZYX convention). Contains the following
        three Euler angles in order:
            - Roll (alpha): Rotation about the x-axis.
            - Pitch (beta): Rotation about the y-axis.
            - Yaw (gamma): Rotation about the z-axis.

    Notes
    -----
    The Euler angles describe how to transition from the 'origin' frame to the 'body'
    frame through three consecutive (passive, intrinsic) rotations in the ZYX order.
    However, the returned rotation matrix represents the transformation of a vector
    from the 'body' frame to the 'origin' frame.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Rotation matrix.
    """
    alpha, beta, gamma = euler
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    rot_00 = cos_gamma * cos_beta
    rot_01 = -sin_gamma * cos_alpha + cos_gamma * sin_beta * sin_alpha
    rot_02 = sin_gamma * sin_alpha + cos_gamma * sin_beta * cos_alpha

    rot_10 = sin_gamma * cos_beta
    rot_11 = cos_gamma * cos_alpha + sin_gamma * sin_beta * sin_alpha
    rot_12 = -cos_gamma * sin_alpha + sin_gamma * sin_beta * cos_alpha

    rot_20 = -sin_beta
    rot_21 = cos_beta * sin_alpha
    rot_22 = cos_beta * cos_alpha

    rot = np.array(
        [[rot_00, rot_01, rot_02], [rot_10, rot_11, rot_12], [rot_20, rot_21, rot_22]]
    )
    return rot


@njit  # type: ignore[misc]
def _quaternion_from_euler(euler: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the unit quaternion (representing transformation from-body-to-origin)
    from Euler angles using the ZYX convention.

    Parameters
    ----------
    euler : numpy.ndarray, shape (3,)
        Vector of Euler angles in radians (ZYX convention). Contains the following
        three Euler angles in order:

            - Roll (alpha): Rotation about the x-axis.
            - Pitch (beta): Rotation about the y-axis.
            - Yaw (gamma): Rotation about the z-axis.
    degrees : bool, default False
        Whether provided Euler angles are in degrees or radians.

    Return
    ------
    numpy.ndarray, shape (4,)
        Unit quaternion.

    Notes
    -----
    The Euler angles describe how to transition from the 'origin' frame to the 'body'
    frame through three consecutive (passive, intrinsic) rotations in the ZYX order.
    However, the returned unit quaternion represents the transformation from the
    'body' frame to the 'origin' frame.

    """
    alpha2, beta2, gamma2 = euler / 2.0  # half angles
    cos_alpha2 = np.cos(alpha2)
    sin_alpha2 = np.sin(alpha2)
    cos_beta2 = np.cos(beta2)
    sin_beta2 = np.sin(beta2)
    cos_gamma2 = np.cos(gamma2)
    sin_gamma2 = np.sin(gamma2)

    # Quaternion
    q_w = cos_gamma2 * cos_beta2 * cos_alpha2 + sin_gamma2 * sin_beta2 * sin_alpha2
    q_x = cos_gamma2 * cos_beta2 * sin_alpha2 - sin_gamma2 * sin_beta2 * cos_alpha2
    q_y = cos_gamma2 * sin_beta2 * cos_alpha2 + sin_gamma2 * cos_beta2 * sin_alpha2
    q_z = sin_gamma2 * cos_beta2 * cos_alpha2 - cos_gamma2 * sin_beta2 * sin_alpha2

    return _normalize(np.array([q_w, q_x, q_y, q_z]))  # type: ignore[no-any-return]  # see _normalize


def quaternion_from_euler(euler: ArrayLike, degrees=False):
    """
    Compute the unit quaternion (representing transformation
    from-body-to-navigation-frame) from Euler angles using the ZYX convention,
    see Notes.

    Parameters
    ----------
    euler : numpy.ndarray, shape (3,)
        Vector of Euler angles (ZYX convention). Contains the following three Euler
        angles in order:

            - Roll (alpha): Rotation about the x-axis.
            - Pitch (beta): Rotation about the y-axis.
            - Yaw (gamma): Rotation about the z-axis.
    degrees : bool, default False
        Whether the provided Euler angles are in degrees or radians (default).

    Return
    ------
    numpy.ndarray, shape (4,)
        Unit quaternion.

    Notes
    -----
    The returned unit quaternion represents the transformation from the
    'body' frame to the 'navigation' frame.

    However, the Euler angles describe how to transition from the 'navigation' frame
    ('NED' or 'ENU) to the 'body' frame through three consecutive intrinsic
    and passive rotations in the ZYX order:

    #. A rotation by an angle gamma (often called yaw) about the z-axis.
    #. A subsequent rotation by an angle beta (often called pitch) about the y-axis.
    #. A final rotation by an angle alpha (often called roll) about the x-axis.

    This sequence of rotations is used to describe the orientation of the 'body' frame
    relative to the 'navigation' frame ('NED' or 'ENU) in 3D space.

    Intrinsic rotations mean that the rotations are with respect to the changing
    coordinate system; as one rotation is applied, the next is about the axis of
    the newly rotated system.

    Passive rotations mean that the frame itself is rotating, not the object
    within the frame.
    """

    euler_ = np.asarray_chkfinite(euler)  # , dtype=np.float64)

    if degrees:
        euler_ = (np.pi / 180.0) * euler_
    return _quaternion_from_euler(euler_)
