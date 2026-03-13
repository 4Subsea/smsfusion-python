from typing import Self

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._ins import _dhda_head, _h_head, _signed_smallest_angle
from ._transforms import _angular_matrix_from_quaternion as T
from ._transforms import _euler_from_quaternion, _rot_matrix_from_quaternion
from ._v2common import (
    _correct_quat_with_gibbs2,
    _kalman_update_scalar,
    _kalman_update_sequential,
    _project_cov_ahead,
)
from ._vectorops import _normalize, _skew_symmetric


VEL_IDX = slice(0, 3)
ATT_IDX = slice(3, 6)
BG_IDX = slice(6, 9)


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


class AHRSv2c:
    """
    Attitude and Heading Reference System (AHRS) using a multiplicative extended
    Kalman filter (MEKF).

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    v_n : array_like, shape (3,), optional
        Initial velocity estimate in m/s.
    q_nb : Attitude or array_like, shape (4,), optional
        Initial attitude estimate as a unit quaternion (qw, qx, qy, qz). Defaults
        to the identity quaternion (1.0, 0.0, 0.0, 0.0) (i.e., no rotation).
    bg_b : array_like, shape (3,), optional
        Initial gyroscope bias estimate (bgx, bgy, bgz) in rad/s. Defaults to zero bias.
    dvel : array_like, shape (3,), optional
        Initial velocity change vector (sculling integral).
    dtheta : array_like, shape (3,), optional
        Initial attitude change vector (coning integral).
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
        v_n: ArrayLike = (0.0, 0.0, 0.0),
        q_nb: ArrayLike = (1.0, 0.0, 0.0, 0.0),
        bg_b: ArrayLike = (0.0, 0.0, 0.0),
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

        if self._nav_frame == "ned":
            self._g_n = np.array([0.0, 0.0, g])
        elif self._nav_frame == "enu":
            self._g_n = np.array([0.0, 0.0, -g])
        else:
            raise ValueError("Invalid navigation frame. Must be 'NED' or 'ENU'.")

        # IMU noise parameters
        self._vrw = acc_noise_density  # velocity random walk
        self._arw = gyro_noise_density  # angular random walk
        self._gbs = gyro_bias_stability  # gyro bias stability
        self._gbc = gyro_bias_corr_time  # gyro bias correlation time

        # State and covariance estimates
        self._v_n = np.asarray_chkfinite(v_n).reshape(3).copy()
        self._q_nb = np.asarray_chkfinite(q_nb).reshape(4).copy()
        self._R_nb = _rot_matrix_from_quaternion(self._q_nb)
        self._bg_b = np.asarray_chkfinite(bg_b).reshape(3).copy()
        self._dvel = np.asarray_chkfinite(dvel).reshape(3).copy()
        self._dtheta = np.asarray_chkfinite(dtheta).reshape(3).copy()
        self._v_n = np.asarray_chkfinite(v_n).reshape(3).copy()
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

    # def angular_rate(self, degrees=False) -> NDArray[np.float64]:
    #     """
    #     Bias corrected angular rate measurement expressed in the body frame.

    #     Parameters
    #     ----------
    #     degrees : bool, optional
    #         Whether to return the angular rate in deg/s or rad/s. Defaults to rad/s.
    #     """
    #     w_b = self._w_b.copy()
    #     if degrees:
    #         w_b = (180.0 / np.pi) * w_b
    #     return w_b

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

        _correct_quat_with_gibbs2(self._q_nb, self._dx[ATT_IDX])
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

    def _project_ahead(self) -> None:
        """
        Project state and covariance estimates ahead.
        """

        # Velocity (dead reckoning)
        self._v_n[:] += self._dvel

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
        vel: ArrayLike | None = (0.0, 0.0, 0.0),
        vel_var: ArrayLike | None = (100.0, 100.0, 100.0),
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
        self._aiding_update_vel(vel, vel_var)
        self._aiding_update_head(head, head_var, head_degrees)

        # Reset state
        self._reset()

        # Update model
        self._f_b[:] = f
        self._w_b[:] = w - self._bg_b
        self._R_nb[:] = _rot_matrix_from_quaternion(self._q_nb)
        self._a_n[:] = self._R_nb @ self._f_b + self._g_n
        _update_state_transition(self._phi, self._dt, self._f_b, self._w_b, self._R_nb)

        return self
