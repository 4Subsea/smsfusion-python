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


class ConingTrajectorySimulator:
    """
    Coning trajectory generator and IMU (gyro) simulator.

    Internally uses 313 Euler angles with constant nutation (cone half-angle) β,
    constant precession rate Ω about inertial Z, and constant spin rate σ about
    the body Z. Orientation is then converted to ZYX (yaw, pitch, roll).

    What you get from calling the simulator:
      - t: (n,) time array [s]
      - euler_zyx: (n, 3) array of [yaw, pitch, roll] in radians (ZYX convention)
      - omega_b: (n, 3) array of body angular rates [p, q, r] in rad/s
                 (these are exactly what a 3-axis gyro in the body frame measures)

    Parameters
    ----------
    beta : float, rad
        Cone half-angle (nutation). β = 0 is pure spin with no coning.
    omega_prec : float, rad/s
        Precession rate Ω about the inertial Z axis.
    sigma_spin : float, rad/s
        Spin rate σ about the body Z axis.
    dt : float, s
        Sampling interval.
    psi0 : float, rad
        Initial precession angle ψ(0).
    phi0 : float, rad
        Initial spin angle φ(0).

    Notes
    -----
    313 angles are:
        ψ(t) = psi0 + Ω t      (precession)
        θ(t) = β                (nutation, constant)
        φ(t) = phi0 + σ t       (spin)
    Rotation matrix (body->inertial): R = Rz(ψ) Rx(β) Rz(φ)

    ZYX (yaw–pitch–roll) extraction from R:
        yaw   = atan2(R21, R11)
        pitch = -arcsin(R31)
        roll  = atan2(R32, R33)

    Body angular velocity (what gyros measure), for 313 with θ=β const:
        p = Ω sinβ sinφ
        q = Ω sinβ cosφ
        r = σ + Ω cosβ
    """

    def __init__(self, omega_prec: float = 1.0, omega_spin: float = 2.0, beta: float = 10.0):
        self._beta = beta
        self._w_prec = omega_prec
        self._w_spin = omega_spin
        self._psi0 = 0.0
        self._phi0 = 0.0

    def __call__(self, fs: float, n: int):
        """
        Generate a length-n coning trajectory sample.

        Parameters
        ----------
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
        psi = self._psi0 + self._w_prec * t  # precession
        theta = self._beta * np.ones_like(t)  # constant nutation
        phi = self._phi0 + self._w_spin * t  # spin

        cpsi, spsi = np.cos(psi), np.sin(psi)
        cth,  sth  = np.cos(theta), np.sin(theta)
        cphi, sphi = np.cos(phi), np.sin(phi)

        # Rotation matrices (vectorized): R = Rz(psi) @ Rx(theta) @ Rz(phi)
        # Build component matrices for each t: shape (n, 3, 3)
        Rz_psi = np.stack([
            np.stack([ cpsi, -spsi, np.zeros_like(t)], axis=-1),
            np.stack([ spsi,  cpsi, np.zeros_like(t)], axis=-1),
            np.stack([ np.zeros_like(t), np.zeros_like(t), np.ones_like(t)], axis=-1)
        ], axis=-2)  # (n, 3, 3)

        Rx_th = np.array([[1.0, 0.0, 0.0],
                          [0.0,  cth, -sth],
                          [0.0,  sth,  cth]], dtype=float)  # (3,3) constant

        Rz_phi = np.stack([
            np.stack([ cphi, -sphi, np.zeros_like(t)], axis=-1),
            np.stack([ sphi,  cphi, np.zeros_like(t)], axis=-1),
            np.stack([ np.zeros_like(t), np.zeros_like(t), np.ones_like(t)], axis=-1)
        ], axis=-2)  # (n, 3, 3)

        # Multiply R = Rz(psi) @ Rx(th) @ Rz(phi) for all t using einsum
        # First A = Rz(psi) @ Rx(th) -> (n,3,3)
        A = np.einsum('nij,jk->nik', Rz_psi, Rx_th)
        # Then R = A @ Rz(phi) -> (n,3,3)
        R = np.einsum('nij,njk->nik', A, Rz_phi)

        # Extract ZYX (yaw, pitch, roll) from R
        R11, R21, R31 = R[:, 0, 0], R[:, 1, 0], R[:, 2, 0]
        R32, R33      = R[:, 2, 1], R[:, 2, 2]

        yaw   = np.arctan2(R21, R11)
        pitch = -np.arcsin(np.clip(R31, -1.0, 1.0))
        roll  = np.arctan2(R32, R33)

        euler_zyx = np.stack([yaw, pitch, roll], axis=-1)

        # IMU gyros (body angular velocity) for uniform coning
        p = self.omega_prec * sth * sphi
        q = self.omega_prec * sth * cphi
        r = self.sigma_spin + self.omega_prec * cth
        omega_b = np.stack([p, q, np.full_like(p, r)], axis=-1)

        return t, euler_zyx, omega_b
