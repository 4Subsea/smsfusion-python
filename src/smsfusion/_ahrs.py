from warnings import warn

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from smsfusion._transforms import (
    _angular_matrix_from_quaternion,
    _euler_from_quaternion,
    _gamma_from_quaternion,
    _rot_matrix_from_quaternion,
)
from smsfusion._vectorops import _cross, _normalize


class AHRS:
    """
    AHRS algorithm using accelerometer and gyroscope (IMU), and compass heading
    based on the non-linear observer filter by Mahony et. al.

    Parameters
    ----------
    fs : float
        Sampling rate (Hz).
    Kp : float
        Error gain factor.
    Ki : float
        Bias gain factor.
    q_init : 1D array
        Quaternion initial value. If None (default), q = [1., 0., 0., 0.] is used.
    bias_init : 1D array
        Bias initial value. If None (default), bias= [0., 0., 0.] is used.
    """

    def __init__(
        self,
        fs: float,
        Kp: float,
        Ki: float,
        q_init: ArrayLike | None = None,
        bias_init: ArrayLike | None = None,
    ) -> None:
        self._fs = fs
        self._dt = 1.0 / fs

        self._Kp = Kp
        self._Ki = Ki

        self._q = self._q_init(q_init)
        self._bias = self._bias_init(bias_init)
        self._error = np.array([0.0, 0.0, 0.0], dtype=float)

    @staticmethod
    def _q_init(q_init: ArrayLike | None) -> NDArray[np.float64]:
        """Initiate quaternion."""
        if q_init is not None:
            q_init = np.asarray_chkfinite(q_init, dtype=float)
            q_abs = np.sqrt(np.dot(q_init, q_init))
            if (0.99 > q_abs) or (q_abs > 1.01):
                warn("'q_init' is not a unit quaternion.")
            q_init /= q_abs
        else:
            q_init = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return q_init

    @staticmethod
    def _bias_init(bias_init: ArrayLike | None) -> NDArray[np.float64]:
        """Initiate bias."""
        if bias_init is not None:
            bias_init = np.asarray_chkfinite(bias_init, dtype=float)
        else:
            bias_init = np.array([0.0, 0.0, 0.0], dtype=float)
        return bias_init

    @staticmethod
    @njit  # type: ignore[misc]
    def _update(
        dt: float,
        q: NDArray[np.float64],
        bias: NDArray[np.float64],
        omega: NDArray[np.float64],
        omega_corr: NDArray[np.float64],
        Kp: float,
        Ki: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Attitude (quaternion) update.

        Implemented as static method so that Numba JIT compiler can be used.

        Parameters
        ----------
        dt : float
            Increment step size in time.
        q : 1D array
            Current quaternion estimate.
        bias : 1D array
            Current quaternion estimate.
        omega : 1D array
            Gyroscope based rotation rate measurements in radians. Measurements
            are assumed to be in the body frame of reference.
        omega_corr : 1D array
            "Corrective" rotation rate measured by other sensors
            (accelerometer, magnetometer, compass).
        Kp : float
            Error gain factor.
        Ki : float
            Bias gain factor.

        Returns
        -------
        q : 1D array
            Updated quaternion estimate.
        bias : 1D array
            Updated bias estimate.
        omega_corr : 1D array
            Error measurement. I.e., "corrective" rotation rate measured by other
            sensors (accelerometer, magnetometer, compass).

        """
        bias = bias - 0.5 * Ki * omega_corr * dt

        omega = omega - bias + Kp * omega_corr

        q = q + (_angular_matrix_from_quaternion(q) @ omega) * dt
        q = _normalize(q)
        return q, bias, omega_corr

    def update(
        self,
        f_imu: ArrayLike,
        w_imu: ArrayLike,
        head: float,
        degrees: bool = True,
        head_degrees: bool = True,
    ) -> None:
        """
        Update the attitude estimate with new measurements from the IMU and compass.

        Parameters
        ----------
        f_imu : array-like (3,)
            IMU specific force measurements (i.e., accelerations + gravity). Given as
            ``[f_x, f_y, f_z]^T`` where ``f_x``, ``f_y`` and ``f_z`` are
            acceleration measurements in x-, y-, and z-direction, respectively. See Notes.
        w_imu : array-like (3,)
            IMU rotation rate measurements. Given as ``[w_x, w_y, w_z]^T`` where
            ``w_x``, ``w_y`` and ``w_z`` are rotation rates about the x-, y-,
            and z-axis, respectively. Unit determined with ``degrees`` keyword argument.
            See Notes.
        head : float
            Compass heading measurement. Assumes right hand-rule about the NED
              z-axis. Thus, the commonly used clockwise compass heading.
        degrees : bool
            If ``True`` (default), the rotation rates are assumed to be in
            degrees/s. Otherwise in radians/s.
        head_degrees : bool
            If ``True`` (default), the heading is assumed to be in
            degrees. Otherwise in radians.
        """
        f_imu = np.asarray_chkfinite(f_imu, dtype=float)
        w_imu = np.asarray_chkfinite(w_imu, dtype=float)

        v01 = np.array([0.0, 0.0, 1.0], dtype=float)  # inertial direction of gravity
        v2_est_o = np.array([1.0, 0.0, 0.0], dtype=float)

        if degrees:
            w_imu = np.radians(w_imu)
        if head_degrees:
            head = np.radians(head)

        delta = head - _gamma_from_quaternion(self._q)

        v1_meas_i = -_normalize(f_imu)
        v1_est_i = _rot_matrix_from_quaternion(self._q) @ v01

        # postpone rotation to after cross product
        v2_meas_o_i = np.array([np.cos(delta), -np.sin(delta), 0.0])

        w_meas_i = _cross(v1_meas_i, v1_est_i) + _rot_matrix_from_quaternion(
            self._q
        ) @ _cross(v2_meas_o_i, v2_est_o)

        self._q, self._bias, self._error = self._update(
            self._dt, self._q, self._bias, w_imu, w_meas_i, self._Kp, self._Ki
        )
        return self

    def attitude(self, degrees=True):
        """
        Current attitude estimate as Euler angles in ZYX convention.

        Parameters
        ----------
        degrees : bool
            Whether to return the attitude in degrees (default) or radians.

        Returns
        -------
        attitude : 1D array
            Euler angles, i.e., roll, pitch and yaw (in that order). However, the angles
            are according to the ZYX convention.
        """
        attitude = _euler_from_quaternion(self._q)
        if degrees:
            attitude = np.degrees(attitude)
        return attitude

    @property
    def q(self) -> NDArray[np.float64]:
        """
        Get current attitude (quaternion) estimate.
        """
        return self._q.copy()

    @property
    def error(self) -> NDArray[np.float64]:
        """
        Get current error estimate.
        """
        return self._error.copy()

    @property
    def bias(self) -> NDArray[np.float64]:
        """
        Get current bias estimate.
        """
        return self._bias.copy()
