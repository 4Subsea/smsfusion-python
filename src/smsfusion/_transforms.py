from warnings import warn

import numpy as np


def _angular_matrix_from_euler(alpha_beta_gamma):
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


def _rot_matrix_from_euler(alpha_beta_gamma):
    """
    Rotation matrix defined from Euler angles. The rotation matrix describes
    rigid body rotation from-body-to-origin, according to xyz convention. That
    is, first rotate about the x-axis (alpha - roll), then about the y-axis
    (beta - pitch), and lastly about the z-axis (gamma - yaw).

    Rotating from-origin-to-body according to zyx convention, that is first
    rotate about the z-axis (gamma - yaw), then about the y-axis
    (beta - pitch), and lastly about the x-axis (alpha - roll) is achieved by
    transposing (inverting) the rotation matrix defined here.

    Parameters
    ----------
    alpha_beta_gamma : ndarray
        Euler angle about x-axis (alpha-roll), y-axis (beta-pitch), and z-axis
        (gamma-yaw) in radians.

    Return
    ------
    rot : array (Nx3x3)
        3D rotation matrix.

    Notes
    -----
    The shape of 'alpha_beta_gamma' determines the casting rule for the output.
    If 'alpha_beta_gamma' is ndarray (Nx3), then the output is (Nx3x3).

    """
    alpha, beta, gamma = alpha_beta_gamma.T

    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    rot_00 = cos_gamma * cos_beta
    rot_01 = cos_gamma * sin_beta * sin_alpha - cos_alpha * sin_gamma
    rot_02 = cos_gamma * cos_alpha * sin_beta + sin_gamma * sin_alpha

    rot_10 = cos_beta * sin_gamma
    rot_11 = cos_gamma * cos_alpha + sin_gamma * sin_beta * sin_alpha
    rot_12 = cos_alpha * sin_gamma * sin_beta - cos_gamma * sin_alpha

    rot_20 = -sin_beta
    rot_21 = cos_beta * sin_alpha
    rot_22 = cos_beta * cos_alpha

    rot = np.array(
        [[rot_00, rot_10, rot_20], [rot_01, rot_11, rot_21], [rot_02, rot_12, rot_22]]
    ).T
    return rot.reshape(-1, 3, 3)