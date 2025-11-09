import numpy as np


class GyroSimulator:
    """
    Gyroscope simulator.

    Parameters
    ----------
    alpha : SineSignal
        Roll signal.
    beta : SineSignal
        Pitch signal
    gamma : SineSignal
        Yaw signal
    degrees: bool
        Whether to interpret the Euler angle signals as degrees (True) or radians (False).
    """

    def __init__(self, alpha, beta, gamma, degrees=False):
        self._alpha_sig = alpha
        self._beta_sig = beta
        self._gamma_sig = gamma
        self._degrees = degrees
        self._rad_scale = np.pi / 180.0 if degrees else 1.0

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
        _, alpha, alpha_dot = self._alpha_sig(fs, n)
        _, beta, beta_dot = self._beta_sig(fs, n)
        _, gamma, gamma_dot = self._gamma_sig(fs, n)
        euler = np.column_stack([alpha, beta, gamma])
        euler_dot = np.column_stack([alpha_dot, beta_dot, gamma_dot])

        if self._degrees:
            euler = np.deg2rad(euler)
            euler_dot = np.deg2rad(euler_dot)

        w_b = self._angular_velocity_body(euler, euler_dot)

        if degrees:
            euler = np.rad2deg(euler)
            w_b = np.rad2deg(w_b)

        return t, euler, w_b


class SineSignal:
    """
    1D sine wave simulator.

    Defined as:

        y(t) = A * sin(w * t + phi) + B
        dy(t)/dt = A * w * cos(w * t + phi)

    where,

        A: Amplitude of the sine wave.
        w: Angular frequency of the sine wave.
        phi: Phase offset of the sine wave.
        B: Offset of the sine wave.

    Parameters
    ----------
    amp : float, default 1.0
        Amplitude of the sine wave. Default is 1.0.
    omega : float, default 1.0
        Angular frequency of the sine wave in rad/s. Default is 1.0
        radians per second.
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

        y = self._amp * np.sin(self._omega * t + self._phase) + self._offset
        dydt = self._amp * self._omega * np.cos(self._omega * t + self._phase)

        return t, y, dydt


# class Constant1DSimulator:
#     """
#     Constant value simulator for 1D signals.

#     Parameters
#     ----------
#     const : float
#         Constant value to simulate.
#     """

#     def __init__(self, const):
#         self._const = const

#     def __call__(self, fs, n):
#         """
#         Generate a constant signal and its derivative (always zero).

#         Parameters
#         ----------
#         fs : float
#             Sampling frequency in Hz.
#         n : int
#             Number of samples to generate.
#         """
#         y = self._const * np.ones(int(n))
#         dydt = np.zeros_like(y)

#         return y, dydt
