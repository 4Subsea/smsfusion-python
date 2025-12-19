from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from smsfusion._transforms import _rot_matrix_from_euler


class DOF(ABC):
    """
    Abstract base class for degree of freedom (DOF) signal generators.
    """

    @abstractmethod
    def _y(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    def _dydt(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    def _d2ydt2(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("Not implemented.")

    def y(self, t: ArrayLike) -> NDArray[np.float64]:
        """
        Generates y(t) signal.

        Parameters
        ----------
        t : array_like, shape (n,)
            Time vector in seconds.
        """
        t = np.asarray_chkfinite(t)
        return self._y(t)

    def dydt(self, t: ArrayLike) -> NDArray[np.float64]:
        """
        Generates dy(t)/dt signal.

        Parameters
        ----------
        t : array_like, shape (n,)
            Time vector in seconds.
        """
        t = np.asarray_chkfinite(t)
        return self._dydt(t)

    def d2ydt2(self, t):
        """
        Generates d2y(t)/dt2 signal.

        Parameters
        ----------
        t : array_like, shape (n,)
            Time vector in seconds.
        """
        t = np.asarray_chkfinite(t)
        return self._d2ydt2(t)

    def __call__(
        self, t: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Generates y(t), dy(t)/dt, and d2y(t)/dt2 signals.

        Parameters
        ----------
        t : array_like, shape (n,)
            Time vector in seconds.

        Returns
        -------
        y : ndarray, shape (n,)
            DOF signal y(t).
        dydt : ndarray, shape (n,)
            Time derivative, dy(t)/dt, of DOF signal.
        d2ydt2 : ndarray, shape (n,)
            Second time derivative, d2y(t)/dt2, of DOF signal.
        """
        y = self._y(t)
        dydt = self._dydt(t)
        d2ydt2 = self._d2ydt2(t)

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
    - B  : Constant offset of the sine wave.

    Parameters
    ----------
    amp : float, default 1.0
        Amplitude of the sine wave. Default is 1.0.
    freq : float, default 1.0
        Frequency of the sine wave in rad/s. Default is 1.0 rad/s.
    freq_hz : bool, optional
        If True, interpret `omega` as frequency in Hz. If False, interpret as angular
        frequency in radians per second. Default is False.
    phase : float, default 0.0
        Phase offset of the sine wave. Default is 0.0.
    phase_degrees : bool, optional
        If True, interpret `phase` in degrees. If False, interpret in radians.
        Default is False.
    offset : float, default 0.0
        Offset of the sine wave. Default is 0.0.
    """

    def __init__(
        self,
        amp: float = 1.0,
        freq: float = 1.0,
        freq_hz: bool = False,
        phase: float = 0.0,
        phase_degrees: bool = False,
        offset: float = 0.0,
    ) -> None:
        self._amp = amp
        self._w = 2.0 * np.pi * freq if freq_hz else freq
        self._phase = np.deg2rad(phase) if phase_degrees else phase
        self._offset = offset

    def _y(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        y = self._amp * np.sin(self._w * t + self._phase) + self._offset
        return y

    def _dydt(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        dydt = self._amp * self._w * np.cos(self._w * t + self._phase)
        return dydt

    def _d2ydt2(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        d2ydt2 = -self._amp * self._w**2 * np.sin(self._w * t + self._phase)
        return d2ydt2


class ConstantDOF(DOF):
    """
    1D constant DOF signal generator.

    Defined as:

        y(t) = C
        dy(t)/dt = 0
        d2y(t)/dt2 = 0

    where,

    - C : Constant value of the signal.

    Parameters
    ----------
    value : float, default 0.0
        Constant value of the signal. Default is 0.0.
    """

    def __init__(self, value: float = 0.0) -> None:
        self._value = value

    def _y(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.full_like(t, self._value)

    def _dydt(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros_like(t)

    def _d2ydt2(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros_like(t)


class BeatDOF(DOF):
    """
    1D beat signal DOF generator.

    Defined as:

        y = A * sin(f_beat / 2.0 * t) * cos(f_main * t + phi) + B

    where,

    - A      : Amplitude of the sine waves.
    - w_main : Angular frequency of the main sine wave.
    - w_beat : Angular frequency of the beat sine wave.
    - phi    : Phase offset of the main sine wave.
    - B      : Constant offset of the beat signal.

    Parameters
    ----------
    f_main : float
        The main frequency of the sinusoidal signal, y(t).
    f_beat : float
        The beating frequency, which controls the variation in amplitude.
    freq_hz : bool, default True.
        Whether the frequencies, ``f_main`` and ``f_beat``, are in Hz or rad/s (default).
    """

    def __init__(
        self,
        amp: float = 1.0,
        freq_main: float = 1.0,
        freq_beat: float = 0.1,
        freq_hz: bool = False,
        phase: float = 0.0,
        phase_degrees: bool = False,
        offset: float = 0.0,
    ) -> None:
        self._amp = amp
        self._w_main = 2.0 * np.pi * freq_main if freq_hz else freq_main
        self._w_beat = 2.0 * np.pi * freq_beat if freq_hz else freq_beat
        self._phase = np.deg2rad(phase) if phase_degrees else phase
        self._offset = offset

    def _y(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        main = np.cos(self._w_main * t + self._phase)
        beat = np.sin(self._w_beat / 2.0 * t)
        y = beat * main
        return y  # type: ignore[no-any-return]
    
    def _dydt(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        main = np.cos(self._f_main * t + self._phase)
        beat = np.sin(self._f_beat / 2.0 * t)
        dmain = -self._f_main * np.sin(self._f_main * t + self._phase)
        dbeat = self._f_beat / 2.0 * np.cos(self._f_beat / 2.0 * t)

        dydt = dbeat * main + beat * dmain
        return dydt  # type: ignore[no-any-return]

    def _d2ydt2(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        main = np.cos(self._f_main * t + self._phase)
        beat = np.sin(self._f_beat / 2.0 * t)
        dmain = -self._f_main * np.sin(self._f_main * t + self._phase)
        dbeat = self._f_beat / 2.0 * np.cos(self._f_beat / 2.0 * t)
        d2main = -((self._f_main) ** 2) * np.cos(self._f_main * t + self._phase)
        d2beat = -((self._f_beat / 2.0) ** 2) * np.sin(self._f_beat / 2.0 * t)

        d2ydt2 = dbeat * dmain + d2beat * main + beat * d2main + dbeat * dmain

        return d2ydt2  # type: ignore[no-any-return]


# class LinearRampUp(DOF):
#     """
#     Linear ramp-up wrapper for DOF signals.

#     Parameters
#     ----------
#     dof : _DOF
#         The DOF signal to wrap with a linear ramp-up.
#     t_start : float, default 0.0
#         The start time of the ramp-up in seconds. Default is 0.0, i.e., the ramp-up
#         starts immediately.
#     ramp_length : float, default 1.0
#         The duration of the ramp-up in seconds. Default is 1.0 second.
#     """

#     def __init__(self, dof: DOF, t_start=0.0, ramp_length=1.0):
#         self._dof = dof
#         self._t_start = t_start
#         self._ramp_length = ramp_length
#         self._t_end = t_start + ramp_length

#     def _y_ramp(self, t):
#         ramp_up = np.clip((t - self._t_start) / self._ramp_length, 0.0, 1.0)
#         return ramp_up

#     def _dydt_ramp(self, t):

#         # dydt_ramp = np.ones_like(t)
#         # dydt_ramp = np.where(t < self._t_start, 0.0, dydt_ramp)
#         # dydt_ramp = np.where(t > self._t_start + self._ramp_length, 0.0, dydt_ramp)

#         # dydt_ramp = np.where(
#         #     (t >= self._t_start) & (t <= self._t_start + self._ramp_length),
#         #     1.0 / self._ramp_length,
#         #     0.0,
#         # )
#         # return dydt_ramp

#     def _y(self, t):
#         ramp_up = self._ramp_up(t)
#         return ramp_up * self._dof._y(t)

#     def _dydt(self, t):
#         ramp_up = self._ramp_up(t)
#         return ramp_up * self._dof._dydt(t)

#     def _d2ydt2(self, t):
#         ramp_up = self._ramp_up(t)
#         return ramp_up * self._dof._d2ydt2(t)


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
        degrees: bool = False,
        g: float = 9.80665,
        nav_frame: str = "NED",
    ) -> None:
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

    def _specific_force_body(
        self,
        pos: NDArray[np.float64],
        acc: NDArray[np.float64],
        euler: NDArray[np.float64],
    ) -> NDArray[np.float64]:
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

    def _angular_velocity_body(
        self, euler: NDArray[np.float64], euler_dot: NDArray[np.float64]
    ) -> NDArray[np.float64]:
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

    def __call__(self, fs: float, n: int, degrees: bool | None = None):
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
            Defaults to the value specified at initialization.

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
        pos_x, pos_x_dot, pos_x_ddot = self._pos_x(t)
        pos_y, pos_y_dot, pos_y_ddot = self._pos_y(t)
        pos_z, pos_z_dot, pos_z_ddot = self._pos_z(t)
        alpha, alpha_dot, _ = self._alpha(t)
        beta, beta_dot, _ = self._beta(t)
        gamma, gamma_dot, _ = self._gamma(t)

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
