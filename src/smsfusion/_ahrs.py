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
    AHRS algorithm based on the non-linear observer filter by Mahony et. al. [1]_.

    The algorithm use specific force (acceleration), angular rate,
    and compass heading measurements.

    Parameters
    ----------
    fs : float
        Sampling rate (Hz).
    x0 : array-like, shape (7,)
        Initial state vector containing the following elements in order:

        * Attitude as unit quaternion (4 elements).
        * Gyroscope bias in x, y, z directions (3 elements).

        See Notes for details.
    Kp : float
        Error gain factor.
    Ki : float
        Bias gain factor.

    Notes
    -----
    The quaternion provided as part of the initial state will be normalized to
    ensure unity.

    If initial state is unknown, use [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].

    References
    ----------
    .. [1] Mahony, R., T. Hamel and J.M. Pflimlin, "Nonlinear Complementary
       Filters on the Special Orthogonal Group", IEEE Transactions on Automatic
       Control, vol. 53(5), pp. 1203-1218, 2008.
    """

    def __init__(
        self,
        fs: float,
        x0: ArrayLike,
        Kp: float,
        Ki: float,
    ) -> None:
        self._fs = fs
        self._dt = 1.0 / fs

        self._x0 = np.asarray_chkfinite(x0).reshape(7).copy()
        self._q_nm = _normalize(self._x0[:4])
        self._bias_gyro = self._x0[4:]
        self._error = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        self._Kp = Kp
        self._Ki = Ki

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
        Update the attitude estimate with new measurements.

        Implemented as static method so that Numba JIT compiler can be used.

        Parameters
        ----------
        dt : float
            Increment step size in time.
        q : numpy.ndarray, shape (4,)
            Current attitude as unit quaternion.
        bias : numpy.ndarray, shape (3,)
            Current angular rate bias.
        w_imu : numpy.ndarray, shape (3,)
            Angular rate measurements in radians. Assumed to be measured by
            a strapdown gyroscope, and thus to be in the body frame of reference.
        w_mes : numpy.ndarray, shape (3,)
            "Corrective" angular rate measurements by other sensors, i.e.,
            accelerometer and compass.
        Kp : float
            Error gain factor (cf. proportional).
        Ki : float
            Bias gain factor (cf. integral).

        Returns
        -------
        q : numpy.ndarray, shape (4,)
            Updated attitude as unit quaternion.
        bias : numpy.ndarray, shape (3,)
            Updated angular rate bias.
        w_mes : numpy.ndarray, shape (3,)
            ``w_mes`` passed through for convinience downstream.
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
        Update the attitude estimate with new measurements.

        Parameters
        ----------
        f_imu : array_like, shape (3,)
            Specific force measurements (i.e., accelerations + gravity), given
            as [f_x, f_y, f_z]^T where f_x, f_y and f_z are
            acceleration measurements in x-, y-, and z-direction, respectively.
        w_imu : array_like, shape (3,)
            Angular rate measurements, given as [w_x, w_y, w_z]^T where
            w_x, w_y and w_z are angular rates about the x-, y-,
            and z-axis, respectively.
        head : float
            Compass heading measurement. Assumes right-hand rule about the NED
            z-axis. Thus, the commonly used clockwise compass heading.
        degrees : bool, default True, meaning degrees/s
            Specify whether the angular rates, ``w_imu``, are in degrees/s or radians/s.
        head_degrees : bool, default True.
            Specify whether the compass heading, ``head`` is in degrees or radians.

        Returns
        -------
        AidedINS
            A reference to the instance itself after the update.
        """
        f_imu = np.asarray_chkfinite(f_imu, dtype=np.float64).reshape(3)
        w_imu = np.asarray_chkfinite(w_imu, dtype=np.float64).reshape(3)
        head = float(head)

        if degrees:
            w_imu = np.radians(w_imu)
        if head_degrees:
            head = np.radians(head)

        R_nm = _rot_matrix_from_quaternion(self._q_nm)

        # Reference vectors expressed in NED frame
        v1_ref_n = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # direction of gravity
        v2_ref_n = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # direction of north

        v1_meas_m = -_normalize(f_imu)
        v1_est_m = R_nm.T @ v1_ref_n

        delta_head = head - _gamma_from_quaternion(self._q_nm)
        v2_meas_n = np.array([np.cos(delta_head), -np.sin(delta_head), 0.0])

        # postpone rotation to after cross product
        w_mes = _cross(v1_meas_m, v1_est_m) + R_nm.T @ _cross(v2_meas_n, v2_ref_n)

        self._q_nm, self._bias_gyro, self._error = self._update(
            self._dt, self._q_nm, self._bias_gyro, w_imu, w_mes, self._Kp, self._Ki
        )
        return self

    @property
    def x(self) -> NDArray[np.float64]:
        """
        Get current state vector estimate.

        Returns
        -------
        numpy.ndarray, shape (7,)
            State vector, containing the following elements in order:

            * Attitude as unit quaternion (4 elements).
            * Gyroscope bias in x, y, z directions (3 elements).
        """
        return np.concatenate((self._q_nm, self._bias_gyro))

    def euler(self, degrees: bool = True) -> NDArray[np.float64]:
        """
        Get current attitude estimate as Euler angles in ZYX convention, see Notes.

        Parameters
        ----------
        degrees : bool, default True.
            Specify whether to return the Euler angles in degrees or radians.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Euler angles, specifically: alpha (roll), beta (pitch) and gamma (yaw)
            in that order.

        Notes
        -----
        The Euler angles describe how to transition from the 'NED' frame to the 'body'
        frame through three consecutive intrinsic and passive rotations in the ZYX order:

        #. A rotation by an angle gamma (often called yaw) about the z-axis.
        #. A subsequent rotation by an angle beta (often called pitch) about the y-axis.
        #. A final rotation by an angle alpha (often called roll) about the x-axis.

        This sequence of rotations is used to describe the orientation of the 'body' frame
        relative to the 'NED' frame in 3D space.

        Intrinsic rotations mean that the rotations are with respect to the changing
        coordinate system; as one rotation is applied, the next is about the axis of
        the newly rotated system.

        Passive rotations mean that the frame itself is rotating, not the object
        within the frame.
        """
        theta = _euler_from_quaternion(self._q_nm)

        if degrees:
            theta = (180.0 / np.pi) * theta

        return theta  # type: ignore[no-any-return]  # numpy funcs declare Any as return when given scalar-like

    def quaternion(self) -> NDArray[np.float64]:
        """
        Get the current attitude estimate as unit quaternion (from-body-to-NED).

        Returns
        -------
        numpy.ndarray, shape (4,)
            Attitude as unit quaternion (from-body-to-NED).
        """
        return self._q_nm.copy()  # type: ignore[no-any-return]  # numpy funcs declare Any

    def bias_gyro(self, degrees: bool = False) -> NDArray[np.float64]:
        """
        Get the current angular rate bias estimate.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Current angular rate bias.
        """
        b_gyro = self._bias_gyro.copy()
        if degrees:
            b_gyro = (180.0 / np.pi) * b_gyro
        return b_gyro

    def error(self) -> NDArray[np.float64]:
        """
        Get the current error estimate.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Current error.
        """
        return self._error.copy()
