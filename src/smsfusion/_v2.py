from typing import Self

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._ins import _dhda_head, _h_head, _signed_smallest_angle
from ._transforms import _angular_matrix_from_quaternion as T
from ._transforms import _euler_from_quaternion
from ._v2common import (
    _correct_quat_with_gibbs2,
    _kalman_update_scalar,
    _kalman_update_sequential,
    _nz2vg,
    _project_cov_ahead,
    _vg_b,
)
from ._vectorops import _normalize, _skew_symmetric


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
    dhdx[3:4, 0:3] = _dhda_head(q_nb)  # heading
    return dhdx


class AHRSv2a:
    """
    Attitude and Heading Reference System (AHRS) using a multiplicative extended
    Kalman filter (MEKF).

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
        self._dhdx = _measurement_matrix(self._q_nb, _vg_b(self._q_nb, self._nz2vg))

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

    def angular_rate(self, degrees=False) -> NDArray[np.float64]:
        """
        Bias corrected angular rate measurement expressed in the body frame.

        Parameters
        ----------
        degrees : bool, optional
            Whether to return the angular rate in deg/s or rad/s. Defaults to rad/s.
        """
        w_b = self._w_b.copy()
        if degrees:
            w_b = (180.0 / np.pi) * w_b
        return w_b

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
        self._dhdx[3:4, 0:3] = _dhda_head(q_nb)
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

    def _aiding_update_gref(
        self, vg_meas: ArrayLike | None, vg_var: ArrayLike | None
    ) -> None:
        """
        Update with gravity reference vector aiding measurement.
        """

        if vg_meas is None:
            return None

        if vg_var is None:
            raise ValueError("'vg_var' not provided.")

        vg_b = _vg_b(self._q_nb, self._nz2vg)
        dz = vg_meas - vg_b
        dhdx = self._dhdx_gref(vg_b)
        _kalman_update_sequential(self._dx, self._P, dz, vg_var, dhdx, self._I)

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

    def _project_ahead(self) -> None:
        """
        Project state and covariance estimates ahead.
        """

        # Attitude (dead reckoning)
        self._q_nb[:] += self._dt * T(self._q_nb) @ self._w_b
        self._q_nb[:] = _normalize(self._q_nb)

        # Covariance
        self._P[:, :] = _project_cov_ahead(self._P, self._phi, self._Q)

    def update(
        self,
        f: ArrayLike,
        w: ArrayLike,
        degrees: bool = False,
        head: float | None = None,
        head_var: float | None = None,
        head_degrees: bool = False,
        g_ref: bool = True,
        g_var: ArrayLike | None = (0.001, 0.001, 0.001),
    ) -> Self:
        """
        Update state estimates with IMU and aiding measurements.

        Parameters
        ----------
        f : array_like, shape (3,)
            Specific force (i.e., acceleration + gravity) measurement (fx, fy, fz)
            in m/s^2.
        w : array_like, shape (3,)
            Angular rate measurement (wx, wy, wz) in rad/s (default) or deg/s. See
            ``degrees`` parameter for units.
        degrees : bool, optional
            Specifies whether the unit of the rotation rate, ``w``, is deg/s or
            rad/s. Defaults to rad/s.
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
        g_ref : bool, optional
            Specifies whether the gravity reference vector is used as an aiding measurement.
        g_var : array-like, optional
            Variance of gravitational reference vector measurement noise. Required for
            ``g_ref``.

        Returns
        -------
        AHRS
            A reference to the instance itself after the update.
        """

        if degrees:
            w = np.radians(w)

        # Project (a priori) state and covariance estimates ahead
        self._project_ahead()

        # Update (a posteriori) state and covariance estimates with aiding measurements
        self._aiding_update_gref(-_normalize(f) if g_ref else None, g_var)
        self._aiding_update_head(head, head_var, head_degrees)

        # Reset state
        self._reset()

        # Update model
        self._w_b[:] = w - self._bg_b
        _update_state_transition(self._phi, self._dt, self._w_b)

        return self
