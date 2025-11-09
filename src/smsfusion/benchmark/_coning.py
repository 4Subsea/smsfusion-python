import numpy as np


class ConingSimulator:
    """
    Coning trajectory generator and IMU (gyro) simulator.

    Simulates an IMU sensor with its z-axis tilted an angle beta (constant) with
    respect to the inertial frame's z-axis. The sensor rotates about its z-axis
    with a constant rate (the spin rate), while also spinning around the inertial
    frame's z-axis with a constant rate (the precession rate). The IMU sensor's
    z-axis will thus trace out a cone shape, with the half-angle defined by beta.

    Parameters
    ----------
    omega_prec : float
        Precession angular velocity in rad/s. I.e., the rate at which the IMU sensor's
        z-axis rotates about the inertial frame's z-axis.
    omega_spin : float
        Spin angular velocity in rad/s. I.e., the rate at which the IMU sensor
        rotates about its own z-axis.
    beta : float
        Cone half-angle in radians. I.e., the angle between the IMU sensor's z-axis
        and the inertial frame's z-axis.
    degrees: bool
        Whether to interpret beta in degrees (True) or radians (False), and angular
        velocities in deg/s (True) or rad/s (False).
    """

    def __init__(
        self,
        omega_prec: float = 180.0,
        omega_spin: float = 360.0,
        beta: float = 45.0,
        degrees: bool = True,
    ):
        self._beta = beta
        self._w_prec = omega_prec
        self._w_spin = omega_spin
        self._psi0 = 0.0
        self._phi0 = 0.0

        if degrees:
            self._beta = np.deg2rad(self._beta)
            self._w_prec = np.deg2rad(self._w_prec)
            self._w_spin = np.deg2rad(self._w_spin)

    @staticmethod
    def _rot_matrix_from_euler_zyz(psi, theta, phi):
        """
        Euler ZYZ rotation matrix:
            R = Rz(psi) @ Ry(theta) @ Rz(phi)

        Parameters
        ----------
        psi, theta, phi : array_like
            Euler angles (ZYZ) in radians. Broadcasting is supported.

        Returns
        -------
        R : ndarray
            Rotation matrix/matrices of shape (..., 3, 3).
        """
        cpsi, spsi = np.cos(psi), np.sin(psi)
        ctheta, stheta = np.cos(theta), np.sin(theta)
        cphi, sphi = np.cos(phi), np.sin(phi)

        R11 = cpsi * ctheta * cphi - spsi * sphi
        R12 = -cpsi * ctheta * sphi - spsi * cphi
        R13 = cpsi * stheta
        R21 = spsi * ctheta * cphi + cpsi * sphi
        R22 = -spsi * ctheta * sphi + cpsi * cphi
        R23 = spsi * stheta
        R31 = -stheta * cphi
        R32 = stheta * sphi
        R33 = ctheta

        n = len(psi)
        R = np.empty((n, 3, 3), dtype="float64")

        R[..., 0, 0] = R11
        R[..., 0, 1] = R12
        R[..., 0, 2] = R13
        R[..., 1, 0] = R21
        R[..., 1, 1] = R22
        R[..., 1, 2] = R23
        R[..., 2, 0] = R31
        R[..., 2, 1] = R32
        R[..., 2, 2] = R33

        return R

    @staticmethod
    def _euler_from_rot_matrix_zyx(R):
        R11 = R[:, 0, 0]
        R21 = R[:, 1, 0]
        R31 = R[:, 2, 0]
        R32 = R[:, 2, 1]
        R33 = R[:, 2, 2]
        yaw = np.arctan2(R21, R11)
        pitch = -np.arcsin(R31)
        roll = np.arctan2(R32, R33)

        euler_zyx = np.column_stack([roll, pitch, yaw])

        return euler_zyx

    def _body_rates_from_euler_zyz(self, psi, theta, phi):
        p = -self._w_prec * np.sin(theta) * np.cos(phi)
        q = self._w_prec * np.sin(theta) * np.sin(phi)
        r = self._w_spin + self._w_prec * np.cos(theta)  # constant
        r = np.full_like(p, r)
        w_b = np.column_stack([p, q, r])

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

        # ZYZ Euler angles
        psi = self._psi0 + self._w_prec * t  # precession angle
        theta = self._beta * np.ones_like(t)  # constant cone half-angle
        phi = self._phi0 + self._w_spin * t  # spin angle

        # Rotation matrix (body-to-inertial)
        R = self._rot_matrix_from_euler_zyz(psi, theta, phi)

        # Extract ZYX Euler angles (yaw, pitch, roll)
        euler_zyx = self._euler_from_rot_matrix_zyx(R)

        # Body frame angular velocities from ZYZ Euler angle rates
        w_b = self._body_rates_from_euler_zyz(psi, theta, phi)

        return t, euler_zyx, w_b


class ConingSimulator2:
    """
    Coning trajectory generator and IMU (gyro) simulator.

    Parameters
    ----------
    omega_prec : float
        Precession angular velocity in rad/s.
    omega_spin : float
        Spin angular velocity in rad/s.
    beta : float
        Cone half-angle in radians.
    degrees: bool
        Whether to interpret beta in degrees (True) or radians (False), and angular
        velocities in deg/s (True) or rad/s (False).
    """

    def __init__(
        self, omega_prec: float = 1.0, omega_spin: float = 2.0, degrees: bool = True
    ):
        self._w_prec = omega_prec
        self._w_spin = omega_spin

        if degrees:
            self._w_prec = np.deg2rad(self._w_prec)
            self._w_spin = np.deg2rad(self._w_spin)

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

        # ZYX Euler angles
        alpha0 = 0.0
        gamma0 = 0.0
        beta0 = 0.0
        alpha_dot = self._w_spin
        gamma_dot = self._w_prec
        alpha = alpha0 + alpha_dot * t  # spin angle
        gamma = gamma0 + gamma_dot * t  # precession angle
        beta = beta0 * np.ones_like(t)  # constant
        euler = np.column_stack([alpha, beta, gamma])

        # Body frame angular velocities from ZYZ Euler angle rates
        w_x = alpha_dot * np.ones_like(t)
        w_y = gamma_dot * np.sin(alpha)
        w_z = gamma_dot * np.cos(alpha)
        w_b = np.column_stack([w_x, w_y, w_z])

        return t, euler, w_b


class ConingSimulator3:
    """
    Coning trajectory generator and IMU (gyro) simulator.

    Simulates an IMU sensor with its z-axis tilted an angle beta (constant) with
    respect to the inertial frame's z-axis. The sensor rotates about its z-axis
    with a constant rate (the spin rate), while also spinning around the inertial
    frame's z-axis with a constant rate (the precession rate). The IMU sensor's
    z-axis will thus trace out a cone shape, with the half-angle defined by beta.

    Parameters
    ----------
    psi_fun : callable
        Precession angle in radians as a function of time. I.e., rotation of IMU
        sensor's z-axis about the inertial frame's z-axis.
    psi_dot_fun : callable
        Precession angular velocity in rad/s as a function of time.
    phi_fun : callable
        Spin angle in radians as a function of time.
    phi_dot_fun : callable
        Spin angular velocity in rad/s as a function of time. I.e., the rate at
        which the IMU sensor rotates about its own z-axis.
    beta : float
        Cone half-angle in radians (constant). I.e., the angle between the IMU sensor's
        z-axis and the inertial frame's z-axis.
    degrees: bool
        Whether to interpret beta in degrees (True) or radians (False), and angular
        velocities in deg/s (True) or rad/s (False).
    """

    def __init__(
        self,
        psi_sim,
        phi_sim,
        beta: float = np.pi/4.0,
        degrees: bool = False,
    ):
        self._beta = (np.pi / 180.0) * beta if degrees else beta
        self._psi_sim = psi_sim
        self._phi_sim = phi_sim
        self._psi0 = 0.0
        self._phi0 = 0.0

    @staticmethod
    def _rot_matrix_from_euler_zyz(psi, theta, phi):
        """
        Euler ZYZ rotation matrix:
            R = Rz(psi) @ Ry(theta) @ Rz(phi)

        Parameters
        ----------
        psi, theta, phi : array_like
            Euler angles (ZYZ) in radians. Broadcasting is supported.

        Returns
        -------
        R : ndarray
            Rotation matrix/matrices of shape (..., 3, 3).
        """
        cpsi, spsi = np.cos(psi), np.sin(psi)
        ctheta, stheta = np.cos(theta), np.sin(theta)
        cphi, sphi = np.cos(phi), np.sin(phi)

        R11 = cpsi * ctheta * cphi - spsi * sphi
        R12 = -cpsi * ctheta * sphi - spsi * cphi
        R13 = cpsi * stheta
        R21 = spsi * ctheta * cphi + cpsi * sphi
        R22 = -spsi * ctheta * sphi + cpsi * cphi
        R23 = spsi * stheta
        R31 = -stheta * cphi
        R32 = stheta * sphi
        R33 = ctheta

        n = len(psi)
        R = np.empty((n, 3, 3), dtype="float64")

        R[..., 0, 0] = R11
        R[..., 0, 1] = R12
        R[..., 0, 2] = R13
        R[..., 1, 0] = R21
        R[..., 1, 1] = R22
        R[..., 1, 2] = R23
        R[..., 2, 0] = R31
        R[..., 2, 1] = R32
        R[..., 2, 2] = R33

        return R

    @staticmethod
    def _euler_from_rot_matrix_zyx(R):
        R11 = R[:, 0, 0]
        R21 = R[:, 1, 0]
        R31 = R[:, 2, 0]
        R32 = R[:, 2, 1]
        R33 = R[:, 2, 2]
        yaw = np.arctan2(R21, R11)
        pitch = -np.arcsin(R31)
        roll = np.arctan2(R32, R33)

        euler_zyx = np.column_stack([roll, pitch, yaw])

        return euler_zyx

    def _body_rates_from_euler_zyz(self, theta, phi, psi_dot, phi_dot):
        p = -psi_dot * np.sin(theta) * np.cos(phi)
        q = psi_dot * np.sin(theta) * np.sin(phi)
        r = phi_dot + psi_dot * np.cos(theta)
        r = np.full_like(p, r)
        w_b = np.column_stack([p, q, r])

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

        # ZYZ Euler angles
        psi, psi_dot = self._psi_sim(fs, n)  # precession angle and rate
        phi, phi_dot = self._phi_sim(fs, n)
        theta = self._beta * np.ones_like(t)  # constant cone half-angle

        # Rotation matrix (body-to-inertial)
        R = self._rot_matrix_from_euler_zyz(psi, theta, phi)

        # Extract ZYX Euler angles (yaw, pitch, roll)
        euler_zyx = self._euler_from_rot_matrix_zyx(R)

        # Body frame angular velocities from ZYZ Euler angle rates
        w_b = self._body_rates_from_euler_zyz(theta, phi, psi_dot, phi_dot)

        return t, euler_zyx, w_b


# class IMUSimulator:
#     """
#     Coning trajectory generator and IMU (gyro) simulator.

#     Simulates an IMU sensor with its z-axis tilted an angle beta (constant) with
#     respect to the inertial frame's z-axis. The sensor rotates about its z-axis
#     with a constant rate (the spin rate), while also spinning around the inertial
#     frame's z-axis with a constant rate (the precession rate). The IMU sensor's
#     z-axis will thus trace out a cone shape, with the half-angle defined by beta.

#     Parameters
#     ----------
#     alpha_sim : callable
#         Roll angle and angular velocity simulator.
#     beta_sim : callable
#         Pitch angle and angular velocity simulator.
#     gamma_sim : callable
#         Yaw angle and angular velocity simulator.
#     """

#     def __init__(
#         self,
#         alpha_sim,
#         beta_sim,
#         gamma_sim,
#     ):
#         self._alpha_sim = alpha_sim
#         self._beta_sim = beta_sim
#         self._gamma_sim = gamma_sim

#     def _angular_velocity_body(self, euler, euler_dot):
#         alpha, beta, _ = euler.T
#         alpha_dot, beta_dot, gamma_dot = euler_dot.T

#         w_x = alpha_dot - np.sin(beta) * gamma_dot
#         w_y = np.cos(alpha) * beta_dot + np.sin(alpha) * np.cos(beta) * gamma_dot
#         w_z = -np.sin(alpha) * beta_dot + np.cos(alpha) * np.cos(beta) * gamma_dot

#         w_b = np.column_stack([w_x, w_y, w_z])

#         return w_b

#     def __call__(self, fs: float, n: int):
#         """
#         Generate a length-n coning trajectory and corresponding body-frame angular
#         velocities as measured by an IMU sensor.

#         Parameters
#         ----------
#         fs : float
#             Sampling frequency in Hz.
#         n : int
#             Number of samples to generate.

#         Returns
#         -------
#         t : ndarray, shape (n,)
#             Time vector in seconds.
#         euler_zyx : ndarray, shape (n, 3)
#             ZYX Euler angles [yaw, pitch, roll] in radians.
#         omega_b : ndarray, shape (n, 3)
#             Body angular velocity [p, q, r] in rad/s (IMU gyro measurements).
#         """

#         # Time
#         dt = 1.0 / fs
#         t = dt * np.arange(n)

#         # Euler angles and Euler rates
#         alpha, alpha_dot = self._alpha_sim(fs, n)
#         beta, beta_dot = self._beta_sim(fs, n)
#         gamma, gamma_dot = self._gamma_sim(fs, n)
#         euler = np.column_stack([alpha, beta, gamma])
#         euler_dot = np.column_stack([alpha_dot, beta_dot, gamma_dot])

#         w_b = self._angular_velocity_body(euler, euler_dot)

#         return t, euler, w_b


# class Sine1DSimulator:
#     def __init__(self, omega, phase, freq_hz=False, phase_degrees=False):
#         self._w = 2.0 * np.pi * omega if freq_hz else omega
#         self._phase = np.deg2rad(phase) if phase_degrees else phase

#     def __call__(self, fs, n):
#         dt = 1.0 / fs
#         t = dt * np.arange(n)

#         y = np.sin(self._w * t + self._phase)
#         dydt = self._w * np.cos(self._w * t + self._phase)
        
#         return y, dydt
