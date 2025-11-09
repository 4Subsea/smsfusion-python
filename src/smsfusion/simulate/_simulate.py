import numpy as np


class GyroSimulator:
    """
    Gyroscope simulator.

    Parameters
    ----------
    alpha_sim : callable
        Roll angle and angular velocity simulator.
    beta_sim : callable
        Pitch angle and angular velocity simulator.
    gamma_sim : callable
        Yaw angle and angular velocity simulator.
    degrees: bool
        Whether to interpret the simulated angles in degrees (True) or radians (False).
    """

    def __init__(
        self,
        alpha_sim,
        beta_sim,
        gamma_sim,
        degrees=False,
    ):
        self._alpha_sim = alpha_sim
        self._beta_sim = beta_sim
        self._gamma_sim = gamma_sim
        self._degrees = degrees

    def _angular_velocity_body(self, euler, euler_dot):
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
            Whether to return angles and angular velocities in degrees and degrees
            per second (True) or radians and radians per second (False).

        Returns
        -------
        t : ndarray, shape (n,)
            Time vector in seconds.
        euler : ndarray, shape (n, 3)
            Simulated Euler angles (roll, pitch, yaw).
        w_b : ndarray, shape (n, 3)
            Simulated angular velocities in the body frame.
        """
        if degrees is None:
            degrees = self._degrees

        # Time
        dt = 1.0 / fs
        t = dt * np.arange(n)

        # Euler angles and Euler rates
        alpha, alpha_dot = self._alpha_sim(fs, n)
        beta, beta_dot = self._beta_sim(fs, n)
        gamma, gamma_dot = self._gamma_sim(fs, n)
        euler = np.column_stack([alpha, beta, gamma])
        euler_dot = np.column_stack([alpha_dot, beta_dot, gamma_dot])

        w_b = self._angular_velocity_body(euler, euler_dot)

        if degrees:
            euler = np.rad2deg(euler)
            w_b = np.rad2deg(w_b)

        return t, euler, w_b


class Sine1DSimulator:
    """
    Sine wave simulator for 1D signals.

    Parameters
    ----------
    omega : float
        Angular frequency of the sine wave. If `hz` is True, this is interpreted
        as frequency in Hz; otherwise, it is interpreted as angular frequency in
        radians per second.
    amp : float, optional
        Amplitude of the sine wave. Default is 1.0.
    phase : float, optional
        Phase offset of the sine wave. Default is 0.0.
    hz : bool, optional
        If True, interpret `omega` as frequency in Hz. If False, interpret as angular
        frequency in radians per second. Default is False.
    phase_degrees : bool, optional
        If True, interpret `phase` in degrees. If False, interpret in radians.
        Default is False.
    """

    def __init__(self, omega, amp=1.0, phase=0.0, hz=False, phase_degrees=False):
        self._w = 2.0 * np.pi * omega if hz else omega
        self._amp = amp
        self._phase = np.deg2rad(phase) if phase_degrees else phase

    def __call__(self, fs, n):
        """
        Generate a sine wave and its derivative.

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz.
        n : int
            Number of samples to generate.
        """
        dt = 1.0 / fs
        t = dt * np.arange(n)

        y = self._amp * np.sin(self._w * t + self._phase)
        dydt = self._amp * self._w * np.cos(self._w * t + self._phase)

        return y, dydt


class Constant1DSimulator:
    """
    Constant value simulator for 1D signals.

    Parameters
    ----------
    const : float
        Constant value to simulate.
    """

    def __init__(self, const):
        self._const = const

    def __call__(self, fs, n):
        """
        Generate a constant signal and its derivative (always zero).

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz.
        n : int
            Number of samples to generate.
        """
        y = self._const * np.ones(int(n))
        dydt = np.zeros_like(y)

        return y, dydt
