import numpy as np
from numba import njit
from numpy.typing import NDArray

from smsfusion._vectorops import _normalize


@njit  # type: ignore[misc]
def _dhda_head(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute yaw angle gradient wrt to the unit quaternion.

    Defined in terms of scaled Gibbs vector in ref [1]_, but implemented in terms of
    unit quaternion here to avoid singularities.

    Parameters
    ----------
    q : numpy.ndarray, shape (3,)
        Unit quaternion.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Yaw angle gradient vector.

    References
    ----------
    .. [1] Fossen, T.I., "Handbook of Marine Craft Hydrodynamics and Motion Control",
    2nd Edition, equation 14.254, John Wiley & Sons, 2021.
    """
    q_w, q_x, q_y, q_z = q
    u_y = 2.0 * (q_x * q_y + q_z * q_w)
    u_x = 1.0 - 2.0 * (q_y**2 + q_z**2)
    u = u_y / u_x

    duda_scale = 1.0 / u_x**2
    duda_x = -(q_w * q_y) * (1.0 - 2.0 * q_w**2) - (2.0 * q_w**2 * q_x * q_z)
    duda_y = (q_w * q_x) * (1.0 - 2.0 * q_z**2) + (2.0 * q_w**2 * q_y * q_z)
    duda_z = q_w**2 * (1.0 - 2.0 * q_y**2) + (2.0 * q_w * q_x * q_y * q_z)
    duda = duda_scale * np.array([duda_x, duda_y, duda_z])

    dhda = 1.0 / (1.0 + u**2) * duda

    return dhda  # type: ignore[no-any-return]


@njit  # type: ignore[misc]
def _h_head(q: NDArray[np.float64]) -> float:
    """
    Compute yaw angle from unit quaternion.

    Defined in terms of scaled Gibbs vector in ref [1]_, but implemented in terms of
    unit quaternion here to avoid singularities.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Unit quaternion.

    Returns
    -------
    float
        Yaw angle in the NED reference frame.

    References
    ----------
    .. [1] Fossen, T.I., "Handbook of Marine Craft Hydrodynamics and Motion Control",
    2nd Edition, equation 14.251, John Wiley & Sons, 2021.
    """
    q_w, q_x, q_y, q_z = q
    u_y = 2.0 * (q_x * q_y + q_z * q_w)
    u_x = 1.0 - 2.0 * (q_y**2 + q_z**2)
    return np.arctan2(u_y, u_x)  # type: ignore[no-any-return]


@njit  # type: ignore[misc]
def _signed_smallest_angle(angle: float, degrees: bool = True) -> float:
    """
    Convert the given angle to the smallest angle between [-180., 180) degrees.

    Parameters
    ----------
    angle : float
        Value of angle.
    degrees : bool, default True
        Specify whether ``angle`` is given degrees or radians.

    Returns
    -------
    float
        The smallest angle between [-180., 180) degrees (or  [-pi, pi] radians).
    """
    base = 180.0 if degrees else np.pi
    return (angle + base) % (2.0 * base) - base  # type: ignore[no-any-return]


@njit  # type: ignore[misc]
def _update_quaternion_with_rotvec(
    q: NDArray[np.float64], dtheta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Update a unit quaternion, q, with a small attitude increment, dtheta, parameterized
    as a rotation vector.

    Parameters
    ----------
    q : ndarray, shape (4,)
        Unit quaternion (qw, qx, qy, qz) to be updated (in place).
    dtheta : ndarray, shape (3,)
        Attitude increment (rotation vector).

    References
    ----------
    .. [1] https://www.vectornav.com/resources/inertial-navigation-primer/math-fundamentals/math-coning (Eq. 2.5.1)
    """

    qw, qx, qy, qz = q
    rx, ry, rz = dtheta

    gamma = 0.5 * np.sqrt(rx**2 + ry**2 + rz**2)
    cos_gamma = np.cos(gamma)

    if gamma >= 1e-5:
        scale = np.sin(gamma) / (2.0 * gamma)
    else:
        scale = 0.5

    # Psi
    px = scale * rx
    py = scale * ry
    pz = scale * rz

    q[0] = cos_gamma * qw - px * qx - py * qy - pz * qz
    q[1] = px * qw + cos_gamma * qx + pz * qy - py * qz
    q[2] = py * qw - pz * qx + cos_gamma * qy + px * qz
    q[3] = pz * qw + py * qx - px * qy + cos_gamma * qz
    q[:] = _normalize(q)
    return q


@njit  # type: ignore[misc]
def _update_quaternion_with_gibbs2(
    q: NDArray[np.float64], da: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Update/correct a unit quaternion, q, with a small attitude error, da, parameterized
    as a scaled (2x) Gibbs vector.

    As described in ref [1]_, this correction can be simplified by doing it in two
    steps: first a correction, followed by renormalization. The scaling factor becomes
    obsolete due to the renormalization step.

    Parameters
    ----------
    q : ndarray, shape (4,)
        Unit quaternion (qw, qx, qy, qz) to be updated (in place).
    da : ndarray, shape (3,)
        Attitude error correction parameterized as a scaled (2x) Gibbs vector.

    References
    ----------
    .. [1] Markley & Crassidis (2014), Fundamentals of Spacecraft Attitude Determination
    and Control, Eq. (6.27)-(6.28).
    """

    qw, qx, qy, qz = q
    dax, day, daz = da

    q[0] = qw - 0.5 * (qx * dax + qy * day + qz * daz)
    q[1] = qx + 0.5 * (qw * dax + qy * daz - qz * day)
    q[2] = qy + 0.5 * (qw * day - qx * daz + qz * dax)
    q[3] = qz + 0.5 * (qw * daz + qx * day - qy * dax)
    q[:] = _normalize(q)
    return q


@njit  # type: ignore[misc]
def _kalman_gain(
    P: NDArray[np.float64], h: NDArray[np.float64], r: float
) -> NDArray[np.float64]:
    """
    Compute the Kalman gain for a scalar measurement.

    Parameters
    ----------
    P : ndarray, shape (n, n)
        State error covariance matrix.
    h : ndarray, shape (n,)
        Measurement matrix (row vector).
    r : float
        Scalar measurement noise variance.

    Returns
    -------
    ndarray, shape (n,)
        Kalman gain vector.
    """

    Ph = np.dot(P, h)

    # Innovation covariance
    s = np.dot(h, Ph) + r

    # Kalman gain
    k = Ph / s

    return k


@njit  # type: ignore[misc]
def _covariance_update(
    P: NDArray[np.float64],
    k: NDArray[np.float64],
    h: NDArray[np.float64],
    r: float,
) -> NDArray[np.float64]:
    """
    Compute the updated error covariance matrix estimate (Joseph form).

    Parameters
    ----------
    P : ndarray, shape (n, n)
        Error covariance matrix to be updated in place.
    k : ndarray, shape (n,)
        Kalman gain vector.
    h : ndarray, shape (n,)
        Measurement matrix (row vector).
    r : float
        Scalar measurement noise variance.

    Returns
    -------
    ndarray, shape (n, n)
        Updated state error covariance matrix.
    """
    A = np.eye(k.size) - np.outer(k, h)
    P = A @ P @ A.T + r * np.outer(k, k)
    return P


@njit  # type: ignore[misc]
def _kalman_update_scalar(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: float,
    r: float,
    h: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Scalar Kalman filter measurement update.

    Parameters
    ----------
    x : ndarray, shape (n,)
        State estimate to be updated in place.
    P : ndarray, shape (n, n)
        Error covariance matrix to be updated in place.
    z : float
        Scalar measurement.
    r : float
        Scalar measurement noise variance.
    h : ndarray, shape (n,)
        Measurement matrix (row vector).
    """

    # Kalman gain
    k = _kalman_gain(P, h, r)

    # Updated (a posteriori) state estimate
    x[:] += k * (z - np.dot(h, x))

    # Updated (a posteriori) covariance estimate (Joseph form)
    P[:, :] = _covariance_update(P, k, h, r)
    return x, P


@njit  # type: ignore[misc]
def _kalman_update_sequential(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Sequential (one-at-a-time) Kalman filter measurement update.

    Parameters
    ----------
    x : ndarray, shape (n,)
        State estimate to be updated in place.
    P : ndarray, shape (n, n)
        Error covariance matrix to be updated in place.
    z : ndarray, shape (m,)
        Measurement vector.
    var : ndarray, shape (m,)
        Measurement noise variances corresponding to each scalar measurement.
    H : ndarray, shape (m, n)
        Measurement matrix where each row corresponds to a scalar measurement model.
    """
    m = z.shape[0]
    for i in range(m):
        x, P = _kalman_update_scalar(x, P, z[i], var[i], H[i])
    return x, P


@njit  # type: ignore[misc]
def _project_covariance_ahead(
    P: NDArray[np.float64], phi: NDArray[np.float64], Q: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Project the error covariance matrix estimate ahead.

    Parameters
    ----------
    P : ndarray, shape (n, n)
        Error covariance matrix to be projected ahead (in place).
    phi : ndarray, shape (n, n)
        State transition matrix.
    Q : ndarray, shape (n, n)
        Process noise covariance matrix.
    """
    P[:, :] = phi @ P @ phi.T + Q
    return P


@njit  # type: ignore[misc]
def _nz2vg(nav_frame: str) -> float:
    """
    Gravity direction along the navigation frame's z-axis. Transforms the z-axis
    of the navigation frame to a gravity reference vector (unit vector).

    Parameters
    ----------
    nav_frame : {'NED', 'ENU'}
        Navigation frame.

    Returns
    -------
    float
        Gravity direction along the navigation frame's z-axis. +1.0 for 'NED' and
        -1.0 for 'ENU'.
    """
    if nav_frame.lower() == "ned":
        return 1.0
    elif nav_frame.lower() == "enu":
        return -1.0
    else:
        raise ValueError(f"Unknown navigation frame: {nav_frame}.")


@njit  # type: ignore[misc]
def _gref_b_from_quat(
    q_nb: NDArray[np.float64], nav_frame_factor: float = 1.0
) -> NDArray[np.float64]:
    """
    Compute the gravity reference vector (unit vector) expressed in the body frame
    from a unit quaternion.

    Parameters
    ----------
    q_nb : numpy.ndarray, shape (4,)
        Unit quaternion which transforms a vector from frame {b} to frame {n}.
    nav_frame_factor: float, default 1.0
        Gravity direction along the navigation frame's z-axis. +1.0 for 'NED' and
        -1.0 for 'ENU'.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Gravity reference vector expressed in the body frame (unit vector).
    """

    x = 2.0 * (q_nb[1] * q_nb[3] - q_nb[0] * q_nb[2])
    y = 2.0 * (q_nb[2] * q_nb[3] + q_nb[0] * q_nb[1])
    z = 1.0 - 2.0 * (q_nb[1] ** 2 + q_nb[2] ** 2)

    return nav_frame_factor * np.array([x, y, z])
