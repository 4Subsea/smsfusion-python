from typing import Self

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._ins import _dhda_head, _h_head, _signed_smallest_angle
from ._transforms import _euler_from_quaternion, _rot_matrix_from_quaternion
from ._vectorops import _normalize, _skew_symmetric

VEL_IDX = slice(0, 3)
ATT_IDX = slice(3, 6)
BG_IDX = slice(6, 9)


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
    p1 = scale * rx
    p2 = scale * ry
    p3 = scale * rz

    q[0] = cos_gamma * qw - p1 * qx - p2 * qy - p3 * qz
    q[1] = p1 * qw + cos_gamma * qx + p3 * qy - p2 * qz
    q[2] = p2 * qw - p3 * qx + cos_gamma * qy + p1 * qz
    q[3] = p3 * qw + p2 * qx - p1 * qy + cos_gamma * qz
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


def _state_transition(
    dt: float,
    dvel: NDArray[np.float64],
    dtheta: NDArray[np.float64],
    R_nb: NDArray[np.float64],
    gbc: float,
) -> NDArray[np.float64]:
    """
    State transition matrix.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    dvel : ndarray, shape (3,)
        Velocity change vector (sculling integral).
    dtheta : ndarray, shape (3,)
        Attitude change vector (coning integral).
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
    phi[VEL_IDX, ATT_IDX] -= R_nb @ _skew_symmetric(dvel)  # NB! update each time step
    phi[ATT_IDX, ATT_IDX] -= _skew_symmetric(dtheta)  # NB! update each time step
    phi[ATT_IDX, BG_IDX] -= dt * np.eye(3)
    phi[BG_IDX, BG_IDX] -= dt * np.eye(3) / gbc
    return phi


@njit  # type: ignore[misc]
def _update_state_transition(
    phi: NDArray[np.float64],
    dvel: NDArray[np.float64],
    dtheta: NDArray[np.float64],
    R_nb: NDArray[np.float64],
) -> None:
    """
    Update the state transition matrix in place.

    Parameters
    ----------
    phi : ndarray, shape (9, 9)
        State transition matrix to be updated in place.
    dvel : ndarray, shape (3,)
        Velocity change vector (sculling integral).
    dtheta : ndarray, shape (3,)
        Attitude change vector (coning integral).
    R_nb : ndarray, shape (3, 3)
        Rotation matrix from body to navigation frame.
    """
    dtx, dty, dtz = dtheta
    dvx, dvy, dvz = dvel

    r00, r01, r02 = R_nb[0]
    r10, r11, r12 = R_nb[1]
    r20, r21, r22 = R_nb[2]

    # phi[3:6, 3:6] = np.eye(3) - dt * S(w_b)
    phi[3, 4] = dtz
    phi[3, 5] = -dty
    phi[4, 3] = -dtz
    phi[4, 5] = dtx
    phi[5, 3] = dty
    phi[5, 4] = -dtx

    # phi[0:3, 3:6] = -dt * R_nb @ S(f_b)
    phi[0, 3] = -(dvz * r01 - dvy * r02)
    phi[1, 3] = -(dvz * r11 - dvy * r12)
    phi[2, 3] = -(dvz * r21 - dvy * r22)
    phi[0, 4] = -(-dvz * r00 + dvx * r02)
    phi[1, 4] = -(-dvz * r10 + dvx * r12)
    phi[2, 4] = -(-dvz * r20 + dvx * r22)
    phi[0, 5] = -(dvy * r00 - dvx * r01)
    phi[1, 5] = -(dvy * r10 - dvx * r11)
    phi[2, 5] = -(dvy * r20 - dvx * r21)


def _process_noise_cov(
    dt: float, vrw: float, arw: float, gbs: float, gbc: float
) -> NDArray[np.float64]:
    """
    Process noise covariance matrix.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    vrw : float
        Velocity random walk (accelerometer noise density) in m/s/√Hz.
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
    Q = np.zeros((9, 9))
    Q[VEL_IDX, VEL_IDX] = dt * vrw**2 * np.eye(3)
    Q[ATT_IDX, ATT_IDX] = dt * arw**2 * np.eye(3)
    Q[BG_IDX, BG_IDX] = dt * (2.0 * gbs**2 / gbc) * np.eye(3)
    return Q


def _measurement_matrix(q_nb: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Measurement matrix.

    Parameters
    ----------
    q_nb : ndarray, shape (4,)
        Unit quaternion.

    Returns
    -------
    ndarray, shape (4, 6)
        Linearized measurement matrix.
    """
    dhdx = np.zeros((4, 9))
    dhdx[0:3, VEL_IDX] = np.eye(3)  # velocity
    dhdx[3:4, ATT_IDX] = _dhda_head(q_nb)  # heading
    return dhdx


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


class AHRSv2:
    """
    Attitude and Heading Reference System (AHRS) using a multiplicative extended
    Kalman filter (MEKF).

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    v : array_like, shape (3,), optional
        Initial velocity estimate in m/s.
    q : Attitude or array_like, shape (4,), optional
        Initial attitude estimate as a unit quaternion (qw, qx, qy, qz). Defaults
        to the identity quaternion (1.0, 0.0, 0.0, 0.0) (i.e., no rotation).
    bg : array_like, shape (3,), optional
        Initial gyroscope bias estimate (bgx, bgy, bgz) in rad/s. Defaults to zero bias.
    dvel : array_like, shape (3,), optional
        Initial velocity change vector measurement (sculling integral).
    dtheta : array_like, shape (3,), optional
        Initial attitude change vector measurement (coning integral).
    P : array_like, shape (6, 6), optional
        Initial (a priori) estimate of the error covariance matrix, **P**. If not
        given, a small diagonal matrix will be used.
    acc_noise_density : float, optional
        Accelerometer noise density (velocity random walk) in m/s/√Hz. Defaults to
        0.0007 (SMS Motion 2 noise level).
    gyro_noise_density : float, optional
        Gyroscope noise density (angular random walk) in (rad/s)/√Hz. Defaults to
        0.00005 (SMS Motion 2 noise level).
    gyro_bias_stability : float, optional
        Gyroscope bias stability in rad/s. Defaults to 0.00005 (SMS Motion 2 noise level).
    gyro_bias_corr_time : float, optional
        Gyroscope bias correlation time in seconds. Defaults to 50.0 s.
    g : float, default 9.80665
        The gravitational acceleration m/s^2. Default is 'standard gravity' of 9.80665
        m/s^2.
    nav_frame : {'NED', 'ENU'}, default 'NED'
        Specifies the assumed inertial-like 'navigation' frame. Should be 'NED' (North-East-Down)
        (default) or 'ENU' (East-North-Up). The body's (or IMU sensor's) degrees of freedom
        will be expressed relative to this frame.

    """

    _I: NDArray[np.float64] = np.eye(9)

    def __init__(
        self,
        fs: float,
        v: ArrayLike = (0.0, 0.0, 0.0),
        q: ArrayLike = (1.0, 0.0, 0.0, 0.0),
        bg: ArrayLike = (0.0, 0.0, 0.0),
        dvel: ArrayLike = (0.0, 0.0, 0.0),
        dtheta: ArrayLike = (0.0, 0.0, 0.0),
        P: ArrayLike = 1e-6 * np.eye(9),
        acc_noise_density: float = 0.0007,
        gyro_noise_density: float = 0.0001,
        gyro_bias_stability: float = 0.00005,
        gyro_bias_corr_time: float = 50.0,
        g: float = 9.80665,
        nav_frame: str = "NED",
    ) -> None:
        self._fs = fs
        self._dt = 1.0 / fs
        self._g = g
        self._nav_frame = nav_frame.lower()
        self._g_n = _gravity_nav(self._g, self._nav_frame)
        self._dvel_g_corr = self._dt * self._g_n

        # IMU noise parameters
        self._vrw = acc_noise_density  # velocity random walk
        self._arw = gyro_noise_density  # angular random walk
        self._gbs = gyro_bias_stability  # gyro bias stability
        self._gbc = gyro_bias_corr_time  # gyro bias correlation time

        # State and covariance estimates
        self._v_n = np.asarray_chkfinite(v).reshape(3).copy()
        self._q_nb = np.asarray_chkfinite(q).reshape(4).copy()
        self._R_nb = _rot_matrix_from_quaternion(self._q_nb)
        self._bg_b = np.asarray_chkfinite(bg).reshape(3).copy()
        self._dvel = np.asarray_chkfinite(dvel).reshape(3).copy()
        self._dtheta = np.asarray_chkfinite(dtheta).reshape(3).copy()
        self._P = np.asarray_chkfinite(P).reshape(9, 9).copy()
        self._dx = np.zeros(9)

        # Discrete state-space model
        self._phi = _state_transition(
            self._dt, self._dvel, self._dtheta, self._R_nb, self._gbc
        )
        self._Q = _process_noise_cov(
            self._dt, self._vrw, self._arw, self._gbs, self._gbc
        )
        self._dhdx = _measurement_matrix(self._q_nb)

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

    def dvel(self) -> NDArray[np.float64]:
        """
        Previous velocity change vector measurement (sculling integral).
        """
        return self._dvel.copy()

    def dtheta(self, degrees=False) -> NDArray[np.float64]:
        """
        Previous attitude change vector measurement (coning integral).

        Parameters
        ----------
        degrees : bool, optional
            Whether to return the coning integral in degrees or radians. Defaults
            to radians.
        """
        dtheta = self._dtheta.copy()
        if degrees:
            dtheta = (180.0 / np.pi) * dtheta
        return dtheta

    @property
    def P(self) -> NDArray[np.float64]:
        """
        Copy of the error covariance matrix estimate.
        """
        return self._P.copy()

    def _dhdx_vel(self) -> NDArray[np.float64]:
        """
        Velocity part of the measurement matrix, shape (3, 6).
        """
        return self._dhdx[0:3]

    def _dhdx_yaw(self, q_nb: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Heading (yaw angle) part of the measurement matrix, shape (6,).
        """
        self._dhdx[3:4, ATT_IDX] = _dhda_head(q_nb)
        return self._dhdx[3]

    def _reset(self) -> None:
        """
        Reset state.
        """

        if not self._dx.any():
            return

        _update_quaternion_with_gibbs2(self._q_nb, self._dx[ATT_IDX])
        self._v_n[:] += self._dx[VEL_IDX]
        self._bg_b[:] += self._dx[BG_IDX]
        self._dx[:] = 0.0

    def _aiding_update_vel(
        self, vel_meas: ArrayLike | None, vel_var: ArrayLike | None
    ) -> None:
        """
        Update with velocity aiding measurement.
        """

        if vel_meas is None:
            return None

        if vel_var is None:
            raise ValueError("'vg_var' not provided.")

        dz = vel_meas - self._v_n
        dhdx = self._dhdx_vel()
        _kalman_update_sequential(self._dx, self._P, dz, vel_var, dhdx, self._I)

    def _aiding_update_head(
        self, head_meas: float | None, head_var: float | None, head_degrees: bool
    ) -> None:
        """
        Update with heading aiding measurement.
        """

        if head_meas is None:
            return None

        if head_var is None:
            raise ValueError("'head_var' not provided.")

        if head_degrees:
            head_meas = (np.pi / 180.0) * head_meas
            head_var = (np.pi / 180.0) ** 2 * head_var

        dz = _signed_smallest_angle(head_meas - _h_head(self._q_nb))
        dhdx = self._dhdx_yaw(self._q_nb)
        _kalman_update_scalar(self._dx, self._P, dz, head_var, dhdx, self._I)

    def _project_ahead(self, dvel, dtheta) -> None:
        """
        Project state and covariance estimates ahead.
        """

        # Velocity (dead reckoning)
        self._v_n[:] += self._R_nb @ dvel + self._dvel_g_corr

        # Attitude (dead reckoning)
        _update_quaternion_with_rotvec(self._q_nb, dtheta)

        # Covariance
        self._P[:, :] = _project_cov_ahead(self._P, self._phi, self._Q)

    def update(
        self,
        dvel: ArrayLike,
        dtheta: ArrayLike,
        degrees: bool = False,
        head: float | None = None,
        head_var: float | None = None,
        head_degrees: bool = False,
        vel: ArrayLike | None = (0.0, 0.0, 0.0),
        vel_var: ArrayLike | None = (100.0, 100.0, 100.0),
    ) -> Self:
        """
        Update state estimates with IMU and aiding measurements.

        Parameters
        ----------
        dvel : array_like, shape (3,), optional
            Velocity change vector (sculling integral).
        dtheta : array_like, shape (3,), optional
            Attitude change vector (coning integral).
        degrees : bool, optional
            Specifies whether the unit of the attitude change vector, ``dtheta``,
            is degrees or radians. Defaults to radians.
        head : float, optional
            Heading measurement. I.e., the yaw angle of the 'body' frame relative to the
            assumed 'navigation' frame ('NED' or 'ENU') specified during initialization.
            If ``None``, compass aiding is not used. See ``head_degrees`` for units.
        head_var : float, optional
            Variance of heading measurement noise. Units must be compatible with ``head``.
             See ``head_degrees`` for units. Required for ``head``.
        head_degrees : bool, default False
            Specifies whether the unit of ``head`` and ``head_var`` are in degrees and degrees^2,
            or radians and radians^2. Default is in radians and radians^2.

        Returns
        -------
        AHRS
            A reference to the instance itself after the update.
        """

        self._dvel[:] = dvel
        self._dtheta[:] = np.degrees(dtheta) if degrees else dtheta
        self._dtheta -= self._dt * self._bg_b

        # Project (a priori) state and covariance estimates ahead
        self._project_ahead(self._dvel, self._dtheta)

        # Update (a posteriori) state and covariance estimates with aiding measurements
        self._aiding_update_vel(vel, vel_var)
        self._aiding_update_head(head, head_var, head_degrees)

        # Reset state
        self._reset()

        # Update model
        self._R_nb[:] = _rot_matrix_from_quaternion(self._q_nb)
        _update_state_transition(self._phi, self._dvel, self._dtheta, self._R_nb)

        return self
