import numpy as np


class GyroSimulator:
    """
    Coning trajectory generator and IMU (gyro) simulator.

    Simulates an IMU sensor with its z-axis tilted an angle beta (constant) with
    respect to the inertial frame's z-axis. The sensor rotates about its z-axis
    with a constant rate (the spin rate), while also spinning around the inertial
    frame's z-axis with a constant rate (the precession rate). The IMU sensor's
    z-axis will thus trace out a cone shape, with the half-angle defined by beta.

    Parameters
    ----------
    alpha_sim : callable
        Roll angle and angular velocity simulator.
    beta_sim : callable
        Pitch angle and angular velocity simulator.
    gamma_sim : callable
        Yaw angle and angular velocity simulator.
    """

    def __init__(
        self,
        alpha_sim,
        beta_sim,
        gamma_sim,
    ):
        self._alpha_sim = alpha_sim
        self._beta_sim = beta_sim
        self._gamma_sim = gamma_sim

    def _angular_velocity_body(self, euler, euler_dot):
        alpha, beta, _ = euler.T
        alpha_dot, beta_dot, gamma_dot = euler_dot.T

        w_x = alpha_dot - np.sin(beta) * gamma_dot
        w_y = np.cos(alpha) * beta_dot + np.sin(alpha) * np.cos(beta) * gamma_dot
        w_z = -np.sin(alpha) * beta_dot + np.cos(alpha) * np.cos(beta) * gamma_dot

        w_b = np.column_stack([w_x, w_y, w_z])

        return w_b

    def __call__(self, fs: float, n: int):
        """
        Generate a length-n coning trajectory and corresponding body-frame angular
        velocities as measured by an IMU sensor.

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz.
        n : int
            Number of samples to generate.

        Returns
        -------
        t : ndarray, shape (n,)
            Time vector in seconds.
        euler_zyx : ndarray, shape (n, 3)
            ZYX Euler angles [yaw, pitch, roll] in radians.
        omega_b : ndarray, shape (n, 3)
            Body angular velocity [p, q, r] in rad/s (IMU gyro measurements).
        """

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

        return t, euler, w_b


class Sine1DSimulator:
    def __init__(self, omega, amp=0.0, phase=0.0, hz=False, phase_degrees=False):
        self._w = 2.0 * np.pi * omega if hz else omega
        self._amp = amp
        self._phase = np.deg2rad(phase) if phase_degrees else phase

    def __call__(self, fs, n):
        dt = 1.0 / fs
        t = dt * np.arange(n)

        y = self._amp * np.sin(self._w * t + self._phase)
        dydt = self._amp * self._w * np.cos(self._w * t + self._phase)
        
        return y, dydt
    

class Constant1DSimulator:
    def __init__(self, const):
        self._const = const

    def __call__(self, fs, n):
        y = self._const * np.ones(int(n))
        dydt = np.zeros_like(y)
        
        return y, dydt