from typing import Self

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._transforms import _euler_from_quaternion, _rot_matrix_from_quaternion
from ._vectorops import _normalize, _skew_symmetric


P0 = (
    (1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0e-6, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0e-6, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 1.0e-6, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-6),
)



@njit  # type: ignore[misc]
def _update_quaternion_with_gibbs2(
    q: NDArray[np.float64], da: NDArray[np.float64]
) -> None:
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


@njit  # type: ignore[misc]
def _update_quaternion_with_rotvec(
    q: NDArray[np.float64], dtheta: NDArray[np.float64]
) -> None:
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
) -> None:
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


@njit  # type: ignore[misc]
def _kalman_update_sequential(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
) -> None:
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
        _kalman_update_scalar(x, P, z[i], var[i], H[i])


@njit  # type: ignore[misc]
def _project_covariance_ahead(
    P: NDArray[np.float64], phi: NDArray[np.float64], Q: NDArray[np.float64]
) -> None:
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


def _state_transition_matrix(
    dt: float,
    dtheta: NDArray[np.float64],
    gbc: float,
) -> NDArray[np.float64]:
    """
    State transition matrix.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    dvel : ndarray, shape (3,)
        Velocity increment measurement (sculling integral).
    dtheta : ndarray, shape (3,)
        Attitude increment measurement (coning integral).
    R_nb : ndarray, shape (3, 3)
        Rotation matrix from body to navigation frame.
    gbc : float
        Gyro bias correlation time in seconds.

    Returns
    -------
    ndarray, shape (6, 6)
        State transition matrix.
    """
    phi = np.eye(6)
    phi[0:3, 0:3] -= _skew_symmetric(dtheta)  # NB! update each time step
    phi[0:3, 3:6] -= dt * np.eye(3)
    phi[3:6, 3:6] -= dt * np.eye(3) / gbc
    return phi


@njit  # type: ignore[misc]
def _update_state_transition_matrix(
    phi: NDArray[np.float64],
    dtheta: NDArray[np.float64],
) -> None:
    """
    Update the state transition matrix in place.

    Parameters
    ----------
    phi : ndarray, shape (9, 9)
        State transition matrix to be updated in place.
    dvel : ndarray, shape (3,)
        Velocity increment measurement (sculling integral).
    dtheta : ndarray, shape (3,)
        Attitude increment measurement (coning integral).
    R_nb : ndarray, shape (3, 3)
        Rotation matrix from body to navigation frame.
    """
    dtx, dty, dtz = dtheta

    # phi[0:3, 0:3] = np.eye(3) - dt * S(w_b)
    phi[0, 1] = dtz
    phi[0, 2] = -dty
    phi[1, 0] = -dtz
    phi[1, 2] = dtx
    phi[2, 0] = dty
    phi[2, 1] = -dtx


def _process_noise_covariance_matrix(
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
    Q : ndarray, shape (9, 9)
        Process noise covariance matrix.
    """
    Q = np.zeros((6, 6))
    Q[0:3, 0:3] = dt * arw**2 * np.eye(3)
    Q[3:6, 3:6] = dt * (2.0 * gbs**2 / gbc) * np.eye(3)
    return Q


@njit  # type: ignore[misc]
def _reset(q_nb, bg_b, dx) -> None:
    """
    Reset state.

    Parameters
    ----------
    q_nb : ndarray, shape (4,)
        Attitude state estimate parameterized as a unit quaternion to be reset in place.
    bg_b : ndarray, shape (3,)
        Gyroscope bias state estimate to be reset in place.
    dx : ndarray, shape (9,)
        Error state vector containing the corrections to be applied to the state
        estimates. Will be reset to zero after applying the corrections.
    """
    _update_quaternion_with_gibbs2(q_nb, dx[0:3])
    bg_b[:] += dx[3:6]
    dx[:] = 0.0


@njit  # type: ignore[misc]
def _nz_b_from_quat(q_nb: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Unit vector describing the z-axis of frame {n} expressed in frame {b}, computed
    from a unit quaternion, q_nb.

    Note that this vector corresponds to the third row of the rotation matrix which
    transforms a vector from {b} to {n}.

    Parameters
    ----------
    q_nb : numpy.ndarray, shape (4,)
        Unit quaternion which transforms a vector from frame {b} to frame {n}.

    Returns
    -------
    numpy.ndarray, shape (3,)
        The z-axis (unit vector) of frame {n} expressed in frame {b}.
    """

    x = 2.0 * (q_nb[1] * q_nb[3] - q_nb[0] * q_nb[2])
    y = 2.0 * (q_nb[2] * q_nb[3] + q_nb[0] * q_nb[1])
    z = 1.0 - 2.0 * (q_nb[1] ** 2 + q_nb[2] ** 2)

    return np.array([x, y, z])


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


class VRUv2:
    """
    Vertical Reference Unit (VRU) using a multiplicative extended
    Kalman filter (MEKF). Uses only gravitational vector as aiding.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    q : Attitude or array_like, shape (4,), optional
        Initial attitude estimate as a unit quaternion (qw, qx, qy, qz). Defaults
        to the identity quaternion (1.0, 0.0, 0.0, 0.0) (i.e., no rotation).
    bg : array_like, shape (3,), optional
        Initial gyroscope bias estimate (bgx, bgy, bgz) in rad/s. Defaults to zero bias.
    P : array_like, shape (6, 6), optional
        Initial (a priori) estimate of the error covariance matrix. Defaults to
        a small diagonal matrix (1e-6 * np.eye(9)).
    acc_noise_density : float, optional
        Accelerometer noise density (velocity random walk) in (m/s)/√Hz. Defaults to
        0.0007 (m/s)/√Hz (SMS Motion 2 noise level).
    gyro_noise_density : float, optional
        Gyroscope noise density (angular random walk) in (rad/s)/√Hz. Defaults to
        0.00005 (rad/s)/√Hz (SMS Motion 2 noise level).
    gyro_bias_stability : float, optional
        Gyroscope bias stability in rad/s. Defaults to 0.00005 rad/s (SMS Motion 2
        noise level).
    gyro_bias_corr_time : float, optional
        Gyroscope bias correlation time in seconds. Defaults to 50.0 s.
    nav_frame : {'NED', 'ENU'}, optional
        Specifies the assumed inertial-like 'navigation' frame. Should be 'NED' (North-East-Down)
        (default) or 'ENU' (East-North-Up). The body's (or IMU sensor's) degrees of freedom
        will be expressed relative to this frame.

    """
    def __init__(
        self,
        fs: float,
        q: ArrayLike = (1.0, 0.0, 0.0, 0.0),
        bg: ArrayLike = (0.0, 0.0, 0.0),
        P: ArrayLike = P0,
        acc_noise_density: float = 0.0007,
        gyro_noise_density: float = 0.0001,
        gyro_bias_stability: float = 0.00005,
        gyro_bias_corr_time: float = 50.0,
        nav_frame: str = "NED",
    ) -> None:
        self._fs = fs
        self._dt = 1.0 / fs
        self._nav_frame = nav_frame.lower()
        self._nz2vg = _nz2vg(self._nav_frame)

        # IMU noise parameters
        self._vrw = acc_noise_density  # velocity random walk
        self._arw = gyro_noise_density  # angular random walk
        self._gbs = gyro_bias_stability  # gyro bias stability
        self._gbc = gyro_bias_corr_time  # gyro bias correlation time

        # State and covariance estimates
        self._q_nb = np.asarray_chkfinite(q).reshape(4).copy()
        self._bg_b = np.asarray_chkfinite(bg).reshape(3).copy()
        self._P = np.asarray_chkfinite(P).reshape(6, 6).copy()
        self._dx = np.zeros(6)

        # Discrete state-space model
        self._phi = _state_transition_matrix(
            self._dt,
            np.zeros(3),
            self._gbc,
        )
        self._Q = _process_noise_covariance_matrix(
            self._dt, self._arw, self._gbs, self._gbc
        )

        self._dhdx = np.zeros((3, 6))

    def quaternion(self) -> NDArray[np.float64]:
        """
        Attitude expressed as a unit quaternion.
        """
        return self._q_nb.copy()

    def euler(self, degrees: bool = False) -> NDArray[np.float64]:
        """
        Attitude expressed as Euler angles (roll, pitch, yaw).

        Parameters
        ----------
        degrees : bool, default False
            Whether to return the Euler angles in degrees or radians.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Euler angles (roll, pitch, yaw).
        """

        theta = _euler_from_quaternion(self._q_nb)

        if degrees:
            theta = (180.0 / np.pi) * theta

        return theta

    def bias_gyro(self, degrees=False) -> NDArray[np.float64]:
        """
        Gyroscope bias estimate (rad/s) expressed in the body frame.

        Parameters
        ----------
        degrees : bool, optional
            Whether to return the bias in deg/s or rad/s. Defaults to rad/s.
        """
        bg_b = self._bg_b.copy()
        if degrees:
            bg_b = (180.0 / np.pi) * bg_b
        return bg_b

    @property
    def P(self) -> NDArray[np.float64]:
        """
        Copy of the error covariance matrix estimate.
        """
        return self._P.copy()

    def _aiding_update_gref(
        self,
        dvel: NDArray[np.float64],
        gref_var: NDArray[np.float64] | None,
        q_nb: NDArray[np.float64],
    ) -> None:
        """
        Update state and covariance with gravity reference vector aiding measurement.
        """

        if gref_var is None:
            raise ValueError("gref_var is not provided; required for gref aiding.")

        vg_b = self._nz2vg * _nz_b_from_quat(q_nb)
        dz = -_normalize(dvel) - vg_b
        self._dhdx[0:3, 0:3] = _skew_symmetric(vg_b)

        _kalman_update_sequential(self._dx, self._P, dz, gref_var, self._dhdx[0:3])


    def update(
        self,
        dvel: ArrayLike,
        dtheta: ArrayLike,
        degrees: bool = False,
        gref: bool = True,
        gref_var: ArrayLike = (0.001, 0.001, 0.001),
    ) -> Self:
        """
        Update state estimates with IMU and aiding measurements.

        Parameters
        ----------
        dvel : array_like, shape (3,), optional
            Velocity increment (sculling integral) in m/s.
        dtheta : array_like, shape (3,), optional
            Attitude increment (coning integral) in radians.
        degrees : bool, optional
            Specifies whether the unit of the attitude increment, ``dtheta``, is
            degrees or radians. Defaults to radians.
        gref : bool, optional
            Specifies whether to use accelerometer measurements (dv) and the known
            direction of gravity as aiding. Defaults to ``True``.
        gref_var : array_like, shape (3,), optional
            Variance of gravity reference vector measurement noise (dimensionless).
            Required for gravity reference vector aiding. Defaults to (0.001, 0.001, 0.001).

        Returns
        -------
        AHRS
            A reference to the instance itself after the update.
        """

        dvel = np.asarray(dvel)
        dtheta = np.asarray(dtheta)

        if degrees:
            dtheta = np.radians(dtheta)

        dtheta = dtheta - self._dt * self._bg_b

        # Update state-space model
        _update_state_transition_matrix(self._phi, dtheta)

        # Project (a priori) state estimates ahead
        _update_quaternion_with_rotvec(self._q_nb, dtheta)

        # Project (a priori) error covariance matrix estimate ahead
        _project_covariance_ahead(self._P, self._phi, self._Q)

        # Update (a posteriori) state and covariance estimates with aiding measurements
        if gref is True:
            self._aiding_update_gref(
                dvel,
                gref_var,
                self._q_nb
            )

        # Reset state
        _reset(self._q_nb, self._bg_b, self._dx)

        return self
