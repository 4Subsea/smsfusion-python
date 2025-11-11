from abc import ABC, abstractmethod

import numpy as np

from smsfusion._transforms import _rot_matrix_from_euler


class DOF(ABC):
    """
    Abstract base class for degree of freedom (DOF) signal generators.
    """

    @abstractmethod
    def _y(self, fs, n):
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    def _dydt(self, fs, n):
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    def _d2ydt2(self, fs, n):
        raise NotImplementedError("Not implemented.")

    def _time(self, fs, n):
        dt = 1.0 / fs
        t = dt * np.arange(n)
        return t

    def __call__(self, fs, n):
        """
        Generate a length-n signal and its first and second time derivative.

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz.
        n : int
            Number of samples to generate.

        Returns
        -------
        y : numpy.ndarray, shape (n,)
            Signal.
        dydt : numpy.ndarray, shape (n,)
            First time derivative of signal.
        d2ydt2 : numpy.ndarray, shape (n,)
            Second time derivative of signal.
        """
        y = self._y(fs, n)
        dydt = self._dydt(fs, n)
        d2ydt2 = self._d2ydt2(fs, n)
        return y, dydt, d2ydt2


class SineDOF(DOF):
    """
    1D sine wave DOF signal generator.

    Defined as:

        y(t) = A * sin(w * t + phi) + B
        dy(t)/dt = A * w * cos(w * t + phi)
        d2y(t)/dt2 = -A * w^2 * sin(w * t + phi)

    where,

    - A  : Amplitude of the sine wave.
    - w  : Angular frequency of the sine wave.
    - phi: Phase offset of the sine wave.
    - B  : Offset of the sine wave.

    Parameters
    ----------
    amp : float, default 1.0
        Amplitude of the sine wave. Default is 1.0.
    omega : float, default 1.0
        Angular frequency of the sine wave in rad/s. Default is 1.0 rad/s.
    phase : float, default 0.0
        Phase offset of the sine wave. Default is 0.0.
    offset : float, default 0.0
        Offset of the sine wave. Default is 0.0.
    hz : bool, optional
        If True, interpret `omega` as frequency in Hz. If False, interpret as angular
        frequency in radians per second. Default is False.
    phase_degrees : bool, optional
        If True, interpret `phase` in degrees. If False, interpret in radians.
        Default is False.
    """

    def __init__(self, amp=1.0, omega=1.0, phase=0.0, offset=0.0, phase_degrees=False):
        self._amp = amp
        self._omega = omega
        self._phase = np.deg2rad(phase) if phase_degrees else phase
        self._offset = offset

    def _y(self, fs, n):
        t = self._time(fs, n)
        y = self._amp * np.sin(self._omega * t + self._phase) + self._offset
        return y

    def _dydt(self, fs, n):
        t = self._time(fs, n)
        dydt = self._amp * self._omega * np.cos(self._omega * t + self._phase)
        return dydt

    def _d2ydt2(self, fs, n):
        t = self._time(fs, n)
        d2ydt2 = -self._amp * self._omega**2 * np.sin(self._omega * t + self._phase)
        return d2ydt2


class ConstantDOF(DOF):
    """
    1D constant DOF signal generator.

    Defined as:

        y(t) = C
        dy(t)/dt = 0
        d2y(t)/dt2 = 0

    where,

    - C  : Constant value of the signal.

    Parameters
    ----------
    value : float, default 0.0
        Constant value of the signal. Default is 0.0.
    """

    def __init__(self, value=0.0):
        self._value = value

    def _y(self, fs, n):
        return np.full(int(n), self._value)

    def _dydt(self, fs, n):
        return np.zeros(n)

    def _d2ydt2(self, fs, n):
        return np.zeros(n)


class LinearRampUp(DOF):
    """
    Linear ramp-up wrapper for DOF signals.

    Parameters
    ----------
    dof : _DOF
        The DOF signal to wrap with a linear ramp-up.
    t_start : float, default 0.0
        The start time of the ramp-up in seconds. Default is 0.0, i.e., the ramp-up
        starts immediately.
    ramp_length : float, default 1.0
        The duration of the ramp-up in seconds. Default is 1.0 second.
    """

    def __init__(self, dof: DOF, t_start=0.0, ramp_length=1.0):
        self._dof = dof
        self._t_start = t_start
        self._ramp_length = ramp_length

    def _ramp_up(self, fs, n):
        t = self._time(fs, n)
        ramp_up = np.clip((t - self._t_start) / self._ramp_length, 0.0, 1.0)
        return ramp_up

    def _y(self, fs, n):
        ramp_up = self._ramp_up(fs, n)
        return ramp_up * self._dof(fs, n)[0]

    def _dydt(self, fs, n):
        ramp_up = self._ramp_up(fs, n)
        return ramp_up * self._dof(fs, n)[1]

    def _d2ydt2(self, fs, n):
        ramp_up = self._ramp_up(fs, n)
        return ramp_up * self._dof(fs, n)[2]


class IMUSimulator:
    """
    IMU simulator.

    Parameters
    ----------
    pos_x : float or DOF, default 0.0
        X position signal.
    pos_y : float or DOF, default 0.0
        Y position signal.
    pos_z : float or DOF, default 0.0
        Z position signal.
    alpha : float or DOF, default 0.0
        Roll signal.
    beta : float or DOF, default 0.0
        Pitch signal
    gamma : float or DOF, default 0.0
        Yaw signal
    degrees: bool, default False
        Whether to interpret the Euler angle signals as degrees (True) or radians (False).
        Default is False.
    g : float, default 9.80665
        The gravitational acceleration. Default is 'standard gravity' of 9.80665.
    nav_frame : str, default "NED"
        Navigation frame. Either "NED" (North-East-Down) or "ENU" (East-North-Up).
        Default is "NED".
    """

    def __init__(
        self,
        pos_x: float | DOF = 0.0,
        pos_y: float | DOF = 0.0,
        pos_z: float | DOF = 0.0,
        alpha: float | DOF = 0.0,
        beta: float | DOF = 0.0,
        gamma: float | DOF = 0.0,
        degrees=False,
        g=9.80665,
        nav_frame="NED",
    ):
        self._pos_x = pos_x if isinstance(pos_x, DOF) else ConstantDOF(pos_x)
        self._pos_y = pos_y if isinstance(pos_y, DOF) else ConstantDOF(pos_y)
        self._pos_z = pos_z if isinstance(pos_z, DOF) else ConstantDOF(pos_z)
        self._alpha = alpha if isinstance(alpha, DOF) else ConstantDOF(alpha)
        self._beta = beta if isinstance(beta, DOF) else ConstantDOF(beta)
        self._gamma = gamma if isinstance(gamma, DOF) else ConstantDOF(gamma)
        self._degrees = degrees
        self._nav_frame = nav_frame.lower()

        if self._nav_frame == "ned":
            self._g_n = np.array([0.0, 0.0, g])
        elif self._nav_frame == "enu":
            self._g_n = np.array([0.0, 0.0, -g])
        else:
            raise ValueError("Invalid navigation frame. Must be 'NED' or 'ENU'.")

    def _specific_force_body(self, pos, acc, euler):
        """
        Specific force in the body frame.

        Parameters
        ----------
        pos : ndarray, shape (n, 3)
            Position [x, y, z]^T in meters.
        vel : ndarray, shape (n, 3)
            Velocity [x_dot, y_dot, z_dot]^T in meters per second.
        acc : ndarray, shape (n, 3)
            Acceleration [x_ddot, y_ddot, z_ddot]^T in meters per second squared.
        euler : ndarray, shape (n, 3)
            Euler angles [alpha, beta, gamma]^T in radians.
        """
        n = pos.shape[0]
        f_b = np.zeros((n, 3))

        for i in range(n):
            R_i = _rot_matrix_from_euler(euler[i])
            f_b[i] = R_i.T.dot(acc[i] - self._g_n)

        return f_b

    def _angular_velocity_body(self, euler, euler_dot):
        """
        Angular velocity in the body frame.

        Parameters
        ----------
        euler : ndarray, shape (n, 3)
            Euler angles [alpha, beta, gamma]^T in radians.
        euler_dot : ndarray, shape (n, 3)
            Time derivatives of Euler angles [alpha_dot, beta_dot, gamma_dot]^T
            in radians per second.
        """
        alpha, beta, _ = euler.T
        alpha_dot, beta_dot, gamma_dot = euler_dot.T

        w_x = alpha_dot - np.sin(beta) * gamma_dot
        w_y = np.cos(alpha) * beta_dot + np.sin(alpha) * np.cos(beta) * gamma_dot
        w_z = -np.sin(alpha) * beta_dot + np.cos(alpha) * np.cos(beta) * gamma_dot

        w_b = np.column_stack([w_x, w_y, w_z])

        return w_b

    def __call__(self, fs: float, n: int, degrees=None):
        """
        Generate a length-n gyroscope signal and corresponding Euler angles.

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz.
        n : int
            Number of samples to generate.
        degrees : bool, optional
            Whether to return Euler angles and angular velocities in degrees and
            degrees per second (True) or radians and radians per second (False).

        Returns
        -------
        t : ndarray, shape (n,)
            Time vector in seconds.
        euler : ndarray, shape (n, 3)
            Simulated (ZYX) Euler angles [roll, pitch, yaw]^T.
        w_b : ndarray, shape (n, 3)
            Simulated angular velocities, [w_x, w_y, w_z]^T, in the body frame.
        """
        if degrees is None:
            degrees = self._degrees

        # Time
        dt = 1.0 / fs
        t = dt * np.arange(n)

        # DOFs and corresponding rates and accelerations
        pos_x, pos_x_dot, pos_x_ddot = self._pos_x(fs, n)
        pos_y, pos_y_dot, pos_y_ddot = self._pos_y(fs, n)
        pos_z, pos_z_dot, pos_z_ddot = self._pos_z(fs, n)
        alpha, alpha_dot, _ = self._alpha(fs, n)
        beta, beta_dot, _ = self._beta(fs, n)
        gamma, gamma_dot, _ = self._gamma(fs, n)

        pos = np.column_stack([pos_x, pos_y, pos_z])
        vel = np.column_stack([pos_x_dot, pos_y_dot, pos_z_dot])
        acc = np.column_stack([pos_x_ddot, pos_y_ddot, pos_z_ddot])
        euler = np.column_stack([alpha, beta, gamma])
        euler_dot = np.column_stack([alpha_dot, beta_dot, gamma_dot])

        if self._degrees:
            euler = np.deg2rad(euler)
            euler_dot = np.deg2rad(euler_dot)

        # IMU measurements (i.e., specific force and angular velocity in body frame)
        f_b = self._specific_force_body(pos, acc, euler)
        w_b = self._angular_velocity_body(euler, euler_dot)

        if degrees:
            euler = np.rad2deg(euler)
            w_b = np.rad2deg(w_b)

        return t, pos, vel, euler, f_b, w_b
