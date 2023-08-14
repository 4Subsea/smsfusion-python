from warnings import warn

import numpy as np
from numba import njit

from . import numba_optimized as nob


class AHRS:
    """
    Base class for AHRS algorithms based on Mahony et. al

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

    def __init__(self, fs, Kp, Ki, q_init=None, bias_init=None):
        self._dt = 1.0 / fs

        self._Kp = Kp
        self._Ki = Ki

        self._q = self._q_init(q_init)
        self._bias = self._bias_init(bias_init)
        self._error = np.array([0.0, 0.0, 0.0], dtype=float).reshape(1, 3)

    @staticmethod
    def _q_init(self, q_init):
        if q_init is not None:
            q_init = np.asarray_chkfinite(q_init, dtype=float)
            q_abs = np.sqrt(np.dot(q_init, q_init))
            if (0.99 > q_abs) or (q_abs > 1.01):
                warn("'q_init' is not a unit quaternion.")
            q_init /= q_abs
        else:
            q_init = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return q_init.reshape(1, 4)

    def _bias_init(self, bias_init):
        if bias_init is not None:
            self._bias = np.asarray_chkfinite(bias_init, dtype=float)
        else:
            bias_init = np.array([0.0, 0.0, 0.0], dtype=float)
        return bias_init.reshape(1, 3)

    @staticmethod
    @njit
    def _update(dt, q, bias, omega, omega_corr, Kp, Ki):
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

        q = q + (nob._angular_matrix_from_quaternion(q) @ omega) * dt
        q = nob._normalize(q)
        return q, bias, omega_corr

    def update(self, meas, degrees=True):
        """
        Update the attitude estimate with new measurements from the IMU and compass.

        Parameters
        ----------
        meas : ndarray
            Measurements as array of shape (N, 7), where N is the number of measurement
            samples. Each measurement sample should include Ax, Ay, Az, Gx, Gy, Gz
            and head (in that order) (see Notes).
        degrees : bool
            If True (default), the rotation rates are assumed to be in
            degrees/s. Otherwise in radians/s.

        Notes
        -----
        The following measurements are needed to do an update of the filter:

            * Ax: Accelerations along the x-axis. (Unit not important)
            * Ay: Accelerations along the y-axis. (Unit not important)
            * Az: Accelerations along the z-axis. (Unit not important)
            * Gx: Rotation rate about the x-axis.
            * Gy: Rotation rate about the y-axis.
            * Gz: Rotation rate about the z-axis.
            * head: Heading measurement. Assumes right hand-rule about the "sensor-origin"
              z-axis.

        """
        meas = np.asarray_chkfinite(meas, dtype=float).copy().reshape(-1, 7)
        acc = meas[:, 0:3]
        omega = meas[:, 3:6]
        head = meas[:, 6]
        n = meas.shape[0]

        v01 = np.array([0.0, 0.0, 1.0], dtype=float)  # inertial direction of gravity
        v2_est_o = np.array([1.0, 0.0, 0.0], dtype=float)

        if degrees:
            omega = np.radians(omega)
            head = np.radians(head)

        q_prev = self._q[-1]
        bias_prev = self._bias[-1]

        q_update = np.zeros((n, 4))
        bias_update = np.zeros((n, 3))
        error_update = np.zeros((n, 3))
        for i in range(n):
            delta_i = head[i] - nob._gamma_from_quaternion(q_prev)

            v1_meas_i = nob._normalize(acc[i])
            v1_est_i = nob._rot_matrix_from_quaternion(q_prev) @ v01

            # postpone rotation to after cross product
            v2_meas_o_i = np.array([np.cos(delta_i), -np.sin(delta_i), 0.0])

            omega_meas_i = nob._cross(
                v1_meas_i, v1_est_i
            ) + nob._rot_matrix_from_quaternion(q_prev) @ nob._cross(
                v2_meas_o_i, v2_est_o
            )

            q_update[i], bias_update[i], error_update[i] = self._update(
                self._dt, q_prev, bias_prev, omega[i], omega_meas_i, self._Kp, self._Ki
            )

            q_prev = q_update[i]
            bias_prev = bias_update[i]

        self._q = q_update
        self._bias = bias_update
        self._error = error_update

        return

    def _attitude_rad(self):
        """
        Get current attitude (Euler angles) estimate.

        Parameters
        ----------
        degrees : bool
            If True (default), the angles are in degrees. Otherwise in radians.

        Return
        ------
        alpha_beta_gamma : ndarray
            Euler angle about x-axis (alpha-roll), y-axis (beta-pitch), and z-axis
            (gamma-yaw).

        """
        attitude = np.zeros((len(self._q), 3))
        for i, q_i in enumerate(self._q):
            attitude[i] = nob._euler_from_quaternion(q_i)
        return attitude

    @property
    def q(self):
        """
        Get current attitude (quaternion) estimate.
        """
        return self._q.copy()

    @property
    def error(self):
        """
        Get current error estimate.
        """
        return self._error.copy()

    @property
    def bias(self):
        """
        Get current bias estimate.
        """
        return self._bias.copy()
