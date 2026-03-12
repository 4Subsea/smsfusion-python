from numba import njit
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._ins import dhda_head
from ._vectorops import _normalize, _quaternion_product, _skew_symmetric


def _gravity_nav(g: float, nav_frame: str) -> NDArray[np.float64]:
    """
    Gravity vector expressed in the navigation frame ('NED' or 'ENU').

    Parameters
    ----------
    g : float
        Gravitational acceleration in m/s^2.
    nav_frame : {'NED', 'ENU'}
        Navigation frame in which the gravity vector is expressed.

    Returns
    -------
    ndarray, shape (3,)
        Gravity vector expressed in the navigation frame.
    """
    if nav_frame.lower() == "ned":
        g_n = np.array([0.0, 0.0, g])
    elif nav_frame.lower() == "enu":
        g_n = np.array([0.0, 0.0, -g])
    else:
        raise ValueError(f"Unknown navigation frame: {nav_frame}.")
    return g_n


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
    Gravity reference vector expressed in the body frame, computed from the attitude
    quaternion, q_nb.

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


def _state_transition(
    dt: float, w_b: NDArray[np.float64], gbc: float
) -> NDArray[np.float64]:
    """
    State transition matrix.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    w_b : ndarray, shape (3,)
        Angular rate measurement (bias corrected) in body frame.
    gbc : float
        Gyro bias correlation time in seconds.

    Returns
    -------
    ndarray, shape (6, 6)
        State transition matrix.
    """
    phi = np.eye(6)
    phi[0:3, 0:3] -= dt * _skew_symmetric(w_b)  # NB! update each time step
    phi[0:3, 3:6] -= dt * np.eye(3)
    phi[3:6, 3:6] -= dt * np.eye(3) / gbc
    return phi


@njit  # type: ignore[misc]
def _update_state_transition(
    phi: NDArray[np.float64],
    dt: float,
    w_b: NDArray[np.float64],
) -> None:
    """
    Update the state transition matrix in place.

    Parameters
    ----------
    phi : ndarray, shape (6, 6)
        State transition matrix to be updated in place.
    dt : float
        Time step.
    w_b : ndarray, shape (3,)
        Angular rate measurement (bias corrected) in body frame.
    """
    wx, wy, wz = w_b
    phi[0, 1] = dt * wz
    phi[0, 2] = -dt * wy
    phi[1, 0] = -dt * wz
    phi[1, 2] = dt * wx
    phi[2, 0] = dt * wy
    phi[2, 1] = -dt * wx


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
    dhdx[3:4, 0:3] = dhda_head(q_nb)  # heading
    return dhdx


@njit  # type: ignore[misc]
def _correct_quat_with_gibbs2(q: NDArray[np.float64], da: NDArray[np.float64]) -> None:
    """
    Corrects a unit quaternion, q, with a small attitude error, da, parameterized
    as a scaled (2x) Gibbs vector:

        q = q ⊗ dq(da)

    Parameters
    ----------
    q : ndarray, shape (4,)
        Unit quaternion [qw, qx, qy, qz] (modified in place).
    da : ndarray, shape (3,)
        Small attitude error parameterized as a scaled (2x) Gibbs vector.

    Notes
    -----
    As described in ref [1]_, this correction can be simplified by doing it in two
    steps: first a correction, followed by renormalization. The scaling factor becomes
    obsolete due to the renormalization step.

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


class VRU:
    """
    Vertical Reference Unit (VRU) using a multiplicative extended Kalman filter (MEKF).

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    q_nb : Attitude or array_like, shape (4,), optional
        Initial attitude estimate as a unit quaternion (qw, qx, qy, qz). Defaults
        to the identity quaternion (1.0, 0.0, 0.0, 0.0) (i.e., no rotation).
    bg_b : array_like, shape (3,), optional
        Initial gyroscope bias estimate (bgx, bgy, bgz) in rad/s. Defaults to zero bias.
    w_b : array_like, shape (3,), optional
        Initial angular rate estimate (wx, wy, wz) in rad/s expressed in the body frame.
        Defaults to zero angular rate (stationary).
    P : array_like, shape (6, 6), optional
        Initial (a priori) estimate of the error covariance matrix, **P**. If not
        given, a small diagonal matrix will be used.
    gyro_noise_density : float, optional
        Gyroscope noise density (angular random walk) in (rad/s)/√Hz. Defaults to
        0.00005 (SMS Motion 2 noise level).
    gyro_bias_stability : float, optional
        Gyroscope bias stability in rad/s. Defaults to 0.00005 (SMS Motion 2 noise level).
    gyro_bias_corr_time : float, optional
        Gyroscope bias correlation time in seconds. Defaults to 50.0 s.
    nav_frame : {'NED', 'ENU'}, default 'NED'
        Specifies the assumed inertial-like 'navigation' frame. Should be 'NED' (North-East-Down)
        (default) or 'ENU' (East-North-Up). The body's (or IMU sensor's) degrees of freedom
        will be expressed relative to this frame.

    """
    _I: NDArray[np.float64] = np.eye(6)

    def __init__(
        self,
        fs: float,
        q_nb: ArrayLike = (1.0, 0.0, 0.0, 0.0),
        bg_b: ArrayLike = (0.0, 0.0, 0.0),
        w_b: ArrayLike = (0.0, 0.0, 0.0),
        P: ArrayLike = 1e-6 * np.eye(6),
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
        self._arw = gyro_noise_density  # angular random walk
        self._gbs = gyro_bias_stability  # gyro bias stability
        self._gbc = gyro_bias_corr_time  # gyro bias correlation time

        # State and covariance estimates
        self._q_nb = np.asarray_chkfinite(q_nb).reshape(4).copy()
        self._bg_b = np.asarray_chkfinite(bg_b).reshape(3).copy()
        self._w_b = np.asarray_chkfinite(w_b).reshape(3).copy()
        self._P = np.asarray_chkfinite(P).reshape(6, 6).copy()
        self._dx = np.zeros(6)

        # Discrete state-space model
        self._phi = _state_transition(self._dt, self._w_b, self._gbc)
        self._Q = _process_noise_cov(self._dt, self._arw, self._gbs, self._gbc)
        self._dhdx = _measurement_matrix(self._q_nb, self._vg_b)

    @property
    def _vg_b(self):
        """Gravity reference vector (unit vector) expressed in the body frame."""
        return _vg_b(self._q_nb, self._nz2vg)

    def quaternion(self) -> NDArray[np.float64]:
        """
        Return a copy of the attitude quaternion.
        """
        return self._q_nb.copy()

    def bias_gyro(self) -> NDArray[np.float64]:
        """
        Return a copy of the gyroscope bias estimate (rad/s) expressed in the body frame.
        """
        return self._bg_b.copy()

    def angular_rate(self) -> NDArray[np.float64]:
        """
        Return a copy of the bias corrected angular rate measurement (rad/s).
        """
        return self._w_b.copy()

    @property
    def P(self) -> NDArray[np.float64]:
        """
        Copy of the error covariance matrix estimate.
        """
        return self._P.copy()

    def _dhdx_gref(self, vg_b: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Gravity reference vector part of the measurement matrix, shape (3, 6).
        """
        self._dhdx[0:3, 0:3] = _skew_symmetric(vg_b)
        return self._dhdx[0:3]

    def _dhdx_yaw(self, q_nb: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Heading (yaw angle) part of the measurement matrix, shape (6,).
        """
        self._dhdx[3:4, 0:3] = dhda_head(q_nb)
        return self._dhdx[3]

    def _reset(self) -> None:
        """
        Reset state.
        """

        if not self._dx.any():
            return

        _correct_quat_with_gibbs2(self._q_nb, self._dx[0:3])
        self._bg_b[:] += self._dx[3:6]
        self._dx[:] = 0.0
