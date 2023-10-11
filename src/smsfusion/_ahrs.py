from __future__ import annotations

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
    q_init : array_like, optional
        Quaternion initial value. If ``None`` (default), ``q_init = [1., 0., 0., 0.]`` is used.
    bias_init : array_like, optional
        Bias initial value. If ``None`` (default), ``bias_init = [0., 0., 0.]`` is used.
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
        self._error = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def _q_init(q_init: ArrayLike | None) -> NDArray[np.float64]:
        """Initiate quaternion."""
        if q_init is not None:
            q_init = np.asarray_chkfinite(q_init, dtype=np.float64).reshape(4)
            q_abs = np.sqrt(np.dot(q_init, q_init))
            if (0.99 > q_abs) or (q_abs > 1.01):
                warn("'q_init' is not a unit quaternion.")
            q_init /= q_abs
        else:
            q_init = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return q_init

    @staticmethod
    def _bias_init(bias_init: ArrayLike | None) -> NDArray[np.float64]:
        """Initiate bias."""
        if bias_init is not None:
            bias_init = np.asarray_chkfinite(bias_init, dtype=np.float64).reshape(3)
        else:
            bias_init = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        return bias_init

    @staticmethod
    @njit  # type: ignore[misc]
    def _update(
        dt: np.float64,
        q: NDArray[np.float64],
        bias: NDArray[np.float64],
        w_imu: NDArray[np.float64],
        w_mes: NDArray[np.float64],
        Kp: np.float64,
        Ki: np.float64,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Attitude (quaternion) update.

        Implemented as static method so that Numba JIT compiler can be used.

        Parameters
        ----------
        dt : float
            Increment step size in time.
        q : ndarray
            Current quaternion estimate as 1D array.
        bias : ndarray
            Current quaternion estimate as 1D array.
        w_imu : ndarray
            Gyroscope based rotation rate measurements in radians as 1D array.
            Measurements are assumed to be in the body frame of reference.
        w_mes : ndarray
            "Corrective" rotation rate measured by other sensors
            (accelerometer, magnetometer, compass) as 1D array.
        Kp : float
            Error gain factor.
        Ki : float
            Bias gain factor.

        Returns
        -------
        q : ndarray
            Updated quaternion estimate as 1D array.
        bias : ndarray
            Updated bias estimate as 1D array.
        omega_corr : ndarray
            Error measurement. I.e., "corrective" rotation rate measured by other
            sensors (accelerometer, magnetometer, compass) as 1D array.

        """
        bias = bias - Ki * w_mes * dt

        w = w_imu - bias + Kp * w_mes

        q = q + (_angular_matrix_from_quaternion(q) @ w) * dt
        q = _normalize(q)
        return q, bias, w_mes

    def update(
        self,
        f_imu: ArrayLike,
        w_imu: ArrayLike,
        head: float,
        degrees: bool = True,
        head_degrees: bool = True,
    ) -> "AHRS":  # TODO: Replace with ``typing.Self`` when Python > 3.11
        """
        Update the attitude estimate with new measurements from the IMU and compass.

        Parameters
        ----------
        f_imu : array_like
            IMU specific force measurements (i.e., accelerations + gravity). Given as
            ``[f_x, f_y, f_z]^T`` where ``f_x``, ``f_y`` and ``f_z`` are
            acceleration measurements in x-, y-, and z-direction, respectively. See Notes.
        w_imu : array_like
            IMU rotation rate measurements. Given as ``[w_x, w_y, w_z]^T`` where
            ``w_x``, ``w_y`` and ``w_z`` are rotation rates about the x-, y-,
            and z-axis, respectively. Unit determined with ``degrees`` keyword argument.
            See Notes.
        head : float
            Compass heading measurement. Assumes right-hand rule about the NED z-axis.
            Thus, the commonly used clockwise compass heading.
        degrees : bool
            If ``True`` (default), the rotation rates are assumed to be in
            degrees/s. Otherwise in radians/s.
        head_degrees : bool
            If ``True`` (default), the heading is assumed to be in
            degrees. Otherwise in radians.
        """
        f_imu = np.asarray_chkfinite(f_imu, dtype=np.float64).reshape(3)
        w_imu = np.asarray_chkfinite(w_imu, dtype=np.float64).reshape(3)
        head = float(head)

        if degrees:
            w_imu = np.radians(w_imu)
        if head_degrees:
            head = np.radians(head)

        R_bn = _rot_matrix_from_quaternion(self._q).T

        # Reference vectors expressed in NED frame
        v1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # direction of gravity
        v2 = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # direction of north

        v1_mes = -_normalize(f_imu)
        v1_est = R_bn @ v1

        delta_head = head - _gamma_from_quaternion(self._q)
        v2_mes = np.array([np.cos(delta_head), -np.sin(delta_head), 0.0])

        # postpone rotation to after cross product
        w_mes = _cross(v1_mes, v1_est) + R_bn @ _cross(v2_mes, v2)

        self._q, self._bias, self._error = self._update(
            self._dt, self._q, self._bias, w_imu, w_mes, self._Kp, self._Ki
        )
        return self

    def euler(self, degrees: bool = True) -> NDArray[np.float64]:
        """
        Current attitude estimate as Euler angles in ZYX convention, see Notes.

        Parameters
        ----------
        degrees : bool
            Whether to return the Euler angles in degrees (`True`) or radians (`False`).

        Returns
        -------
        euler : numpy.ndarray
            Euler angle vector, specifically: alpha (roll), beta (pitch) and gamma (yaw)
            in that order.

        Notes
        -----
        The Euler angles describe how to transition from the 'NED' frame to the 'body'
        frame through three consecutive intrinsic and passive rotations in the ZYX order:
            1. A rotation by an angle gamma (often called yaw) about the z-axis.
            2. A subsequent rotation by an angle beta (often called pitch) about the y-axis.
            3. A final rotation by an angle alpha (often called roll) about the x-axis.

        This sequence of rotations is used to describe the orientation of the 'body' frame
        relative to the 'NED' frame in 3D space.

        Intrinsic rotations mean that the rotations are with respect to the changing
        coordinate system; as one rotation is applied, the next is about the axis of
        the newly rotated system.

        Passive rotations mean that the frame itself is rotating, not the object
        within the frame.
        """
        euler = _euler_from_quaternion(self._q)
        if degrees:
            euler = np.degrees(euler)
        return euler  # type: ignore[no-any-return]  # numpy funcs declare Any as return when given scalar-like

    def quaternion(self) -> NDArray[np.float64]:
        """
        Current attitude estimate as unit quaternion (from-body-to-NED).
        """
        return self._q.copy()

    def error(self) -> NDArray[np.float64]:
        """
        Current error estimate.
        """
        return self._error.copy()

    def bias(self) -> NDArray[np.float64]:
        """
        Current bias estimate.
        """
        return self._bias.copy()
