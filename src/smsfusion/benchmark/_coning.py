import numpy as np


# class ConingTrajectorySimulator:
#     """
#     Coning trajectory generator and IMU simulator.

#     A coning trajectory is defined as a circular motion of a vector, r, of constant
#     amplitude, making a constant angle, theta, with respect to a fixed axis. Here,
#     the fixed axis is the z-axis.

#     Let,
#     - R be the amplitude of the vector, r.
#     - theta be the coning (half) angle. I.e., the angle between r and the z-axis.
#     - phi be the heading angle. I.e., the angle between the projection of r onto the
#       x-y plane and the x-axis.
    
#     Then, the vector, r, can be expressed as:

#         r(t) = R * [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)]^T

#     """


#     def __init__(self):
#         pass

#     def __call__(self, n: int):
#         pass


# class ConingTrajectorySimulator:
#     """
#     Coning trajectory generator and IMU simulator.
#     """


#     def __init__(self):
#         pass

#     def __call__(self, n: int):
#         pass


class ConingSimulator:
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

    def __init__(self, omega_prec: float = 1.0, omega_spin: float = 2.0, beta: float = 10.0, degrees: bool = True):
        self._beta = beta
        self._w_prec = omega_prec
        self._w_spin = omega_spin
        self._psi0 = 0.0
        self._phi0 = 0.0

        if degrees:
            self._beta = np.deg2rad(self._beta)
            self._w_prec = np.deg2rad(self._w_prec)
            self._w_spin = np.deg2rad(self._w_spin)

    # @staticmethod
    # def _rot_matrix_from_euler_zxz(psi: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    #     cpsi, spsi = np.cos(psi), np.sin(psi)
    #     ctheta, stheta = np.cos(theta), np.sin(theta)
    #     cphi, sphi = np.cos(phi), np.sin(phi)

    #     R_zyz = np.stack([
    #         np.stack([cpsi * cphi - spsi * ctheta * sphi, -cpsi * sphi - spsi * ctheta * cphi, spsi * stheta], axis=-1),
    #         np.stack([spsi * cphi + cpsi * ctheta * sphi, -spsi * sphi + cpsi * ctheta * cphi, -cpsi * stheta], axis=-1),
    #         np.stack([stheta * sphi, stheta * cphi, ctheta], axis=-1)
    #     ], axis=-2)

    #     return R_zyz

    # def _body_rates_from_euler_zxz(self, psi, theta, phi):
    #     p = self._w_prec * np.sin(theta) * np.sin(phi)
    #     q = self._w_prec * np.sin(theta) * np.cos(phi)
    #     r = self._w_spin + self._w_prec * np.cos(theta)
    #     r = np.full_like(p, r)
    #     w_b = np.column_stack([p, q, r])

    #     return w_b

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

        R = np.stack([
            np.stack([ cpsi*ctheta*cphi - spsi*sphi,   -cpsi*ctheta*sphi - spsi*cphi,   cpsi*stheta ], axis=-1),
            np.stack([ spsi*ctheta*cphi + cpsi*sphi,   -spsi*ctheta*sphi + cpsi*cphi,   spsi*stheta ], axis=-1),
            np.stack([        -stheta*cphi,               stheta*sphi,       ctheta     ], axis=-1)
        ], axis=-2)
        return R

    @staticmethod
    def _euler_from_rot_matrix_zyx(R):
        R11 = R[:, 0, 0]
        R21 = R[:, 1, 0]
        R31 = R[:, 2, 0]
        R32 = R[:, 2, 1]
        R33 = R[:, 2, 2]
        yaw   = np.arctan2(R21, R11)
        # pitch = -np.arcsin(np.clip(R31, -1.0, 1.0))
        pitch = -np.arcsin(R31)
        roll  = np.arctan2(R32, R33)

        euler_zyx = np.column_stack([roll, pitch, yaw])

        return euler_zyx

    def _body_rates_from_euler_zyz(self, psi, theta, phi):
        p = -self._w_prec * np.sin(theta) * np.cos(phi)
        q = self._w_prec * np.sin(theta) * np.sin(phi)
        r = self._w_spin + self._w_prec * np.cos(theta)
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

    def __init__(self, omega_prec: float = 1.0, omega_spin: float = 2.0, degrees: bool = True):
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
