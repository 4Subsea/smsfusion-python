import numpy as np
from numba import njit
from numpy.typing import NDArray

from .._vectorops import _normalize, _quaternion_product


def _nz2vg(nav_frame: str) -> float:
    """
    Gravity direction along the navigation frame's z-axis.
    """
    if nav_frame == "ned":
        return 1.0
    elif nav_frame == "enu":
        return -1.0
    else:
        raise ValueError("Invalid navigation frame. Must be 'NED' or 'ENU'.")


@njit  # type: ignore[misc]
def _vg_b(q_nb: NDArray[np.float64], nz2vg: float) -> NDArray[np.float64]:
    """
    Gravity reference vector expressed in the body frame, computed from a unit quaternion.

    Parameters
    ----------
    q_nb : numpy.ndarray, shape (4,)
        Unit quaternion.
    nz2vg : float
        Gravity direction along the navigation frame's z-axis. Should be +1 for
        NED and -1 for ENU.
    """
    qw, qx, qy, qz = q_nb

    x = 2.0 * (qx * qz - qw * qy)
    y = 2.0 * (qy * qz + qw * qx)
    z = 1.0 - 2.0 * (qx**2 + qy**2)

    return nz2vg * np.array([x, y, z])


@njit  # type: ignore[misc]
def _correct_quat_with_gibbs2(q: NDArray[np.float64], da: NDArray[np.float64]) -> None:
    """
    Corrects a unit quaternion, q, with a small attitude error, da, parameterized
    as a scaled (2x) Gibbs vector:

        q = q ⊗ dq(da)

    As described in ref [1]_, this correction can be simplified by doing it in two
    steps: first a correction, followed by renormalization. The scaling factor becomes
    obsolete due to the renormalization step.

    Parameters
    ----------
    q : ndarray, shape (4,)
        Unit quaternion [qw, qx, qy, qz] (modified in place).
    da : ndarray, shape (3,)
        Small attitude error parameterized as a scaled (2x) Gibbs vector.

    References
    ----------
    Markley & Crassidis (2014), Fundamentals of Spacecraft Attitude Determination
    and Control, Eq. (6.27)-(6.28).
    """

    qw, qx, qy, qz = q
    dax, day, daz = da

    q[0] -= 0.5 * (qx * dax + qy * day + qz * daz)
    q[1] += 0.5 * (qw * dax + qy * daz - qz * day)
    q[2] += 0.5 * (qw * day - qx * daz + qz * dax)
    q[3] += 0.5 * (qw * daz + qx * day - qy * dax)
    q[:] = _normalize(q)


@njit  # type: ignore[misc]
def _quat_from_rotvec(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the unit quaternion from a rotation vector.

    Parameters
    ----------
    r : numpy.ndarray, shape (3,)
        Rotation vector (rx, ry, rz).

    Returns
    -------
    numpy.ndarray, shape (4,)
        Unit quaternion (qw, qx, qy, qz).
    """
    # TODO: add reference

    rx, ry, rz = r

    angle2 = rx**2 + ry**2 + rz**2

    if angle2 < 1e-6:  # 2nd order approximation (avoids division by zero)
        a = 0.25 * angle2
        c = 1.0 - a / 2.0
        s = 0.5 * (1.0 - a / 6.0)
    else:
        angle = np.sqrt(angle2)
        half_angle = 0.5 * angle
        c = np.cos(half_angle)
        s = np.sin(half_angle) / angle

    q = np.array([c, s * rx, s * ry, s * rz])

    return _normalize(q)


@njit  # type: ignore[misc]
def _correct_quat_with_rotvec(
    q: NDArray[np.float64], dtheta: NDArray[np.float64]
) -> None:
    """
    Corrects a unit quaternion, q, with a small attitude change vector, dtheta,
    parameterized as a rotation vector:

        q = q ⊗ dq(dtheta)

    Parameters
    ----------
    q : ndarray, shape (4,)
        Unit quaternion (modified in place).
    dtheta : ndarray, shape (3,)
        Small attitude change parameterized as a rotation vector.
    """
    q[:] = _normalize(_quaternion_product(q, _quat_from_rotvec(dtheta)))


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

    # Innovation covariance (inverse)
    Ph = np.dot(P, h)
    s_inv = 1.0 / (np.dot(h, Ph) + r)

    # Kalman gain
    k = Ph * s_inv

    return k


@njit  # type: ignore[misc]
def _covariance_update(
    P: NDArray[np.float64],
    k: NDArray[np.float64],
    h: NDArray[np.float64],
    r: float,
    I_: NDArray[np.float64],
) -> None:
    """
    Compute the updated state error covariance matrix estimate (Joseph form).

    Parameters
    ----------
    P : ndarray, shape (n, n)
        State error covariance matrix to be updated in place.
    k : ndarray, shape (n,)
        Kalman gain vector.
    h : ndarray, shape (n,)
        Measurement matrix (row vector).
    r : float
        Scalar measurement noise variance.
    I_ : ndarray, shape (n, n)
        Identity matrix.
    """
    A = I_ - np.outer(k, h)
    P = A @ P @ A.T + r * np.outer(k, k)
    return P


@njit  # type: ignore[misc]
def _kalman_update_scalar(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: float,
    r: float,
    h: NDArray[np.float64],
    I_: NDArray[np.float64],
) -> None:
    """
    Scalar Kalman filter measurement update.

    Parameters
    ----------
    x : ndarray, shape (n,)
        State estimate to be updated in place.
    P : ndarray, shape (n, n)
        State error covariance matrix to be updated in place.
    z : float
        Scalar measurement.
    r : float
        Scalar measurement noise variance.
    h : ndarray, shape (n,)
        Measurement matrix (row vector).
    I_ : ndarray, shape (n, n)
        Identity matrix.
    """

    # Kalman gain
    k = _kalman_gain(P, h, r)

    # Updated (a posteriori) state estimate
    x[:] += k * (z - np.dot(h, x))

    # Updated (a posteriori) covariance estimate (Joseph form)
    P[:, :] = _covariance_update(P, k, h, r, I_)


@njit  # type: ignore[misc]
def _kalman_update_sequential(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    I_: NDArray[np.float64],
) -> None:
    """
    Sequential (one-at-a-time) Kalman filter measurement update.

    Parameters
    ----------
    x : ndarray, shape (n,)
        State estimate to be updated in place.
    P : ndarray, shape (n, n)
        State error covariance matrix to be updated in place.
    z : ndarray, shape (m,)
        Measurement vector.
    var : ndarray, shape (m,)
        Measurement noise variances corresponding to each scalar measurement.
    H : ndarray, shape (m, n)
        Measurement matrix where each row corresponds to a scalar measurement model.
    I_ : ndarray, shape (n, n)
        Identity matrix.
    """
    m = z.shape[0]
    for i in range(m):
        _kalman_update_scalar(x, P, z[i], var[i], H[i], I_)


@njit  # type: ignore[misc]
def _project_cov_ahead(
    P: NDArray[np.float64], phi: NDArray[np.float64], Q: NDArray[np.float64]
) -> None:
    """
    Project the error covariance matrix estimate ahead.

    Parameters
    ----------
    P : ndarray, shape (n, n)
        State error covariance matrix to be projected ahead.
    phi : ndarray, shape (n, n)
        State transition matrix.
    Q : ndarray, shape (n, n)
        Process noise covariance matrix.

    Returns
    -------
    ndarray, shape (n, n)
        Projected error covariance matrix estimate.
    """
    P = phi @ P @ phi.T + Q
    return P
