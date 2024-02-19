import numpy as np

from .._ins import gravity
from .._transforms import _angular_matrix_from_quaternion, _rot_matrix_from_quaternion


class PVASignal202402A:
    """
    Position, velocity and attitude (PVA) and IMU signals (without noise) as implemented
    by Fossen.
    """

    def __init__(self, mu=None):
        self._mu = mu
        self._g_ned = np.array([0.0, 0.0, gravity(mu)])

    def simulate(self, x0, fs, n):
        dt = 1.0 / fs
        t = np.arange(0.0, n * dt, dt)
        x0 = np.asarray_chkfinite(x0).reshape(10)

        R_bn = _rot_matrix_from_quaternion
        T = _angular_matrix_from_quaternion

        # Linear acceleration
        acc = np.array(
            [
                0.1 * np.sin(0.1 * t),
                0.1 * np.cos(0.1 * t),
                0.05 * np.sin(0.05 * t),
            ]
        )

        # Angular rate
        ang_rate = np.array(
            [
                0.01 * np.cos(0.2 * t),
                -0.02 * np.sin(0.1 * t),
                0.01 * np.sin(0.1 * t),
            ]
        )

        # Simulate
        x_k = x0
        x = np.zeros((n, 10))
        f = np.zeros((n, 3))
        w = np.zeros((n, 3))
        for k, (a_k, w_k) in enumerate(zip(acc.T, ang_rate.T)):

            # Current state and IMU measurements
            x[k, :] = x_k  # State
            f[k, :] = a_k - R_bn(x_k[6:10]).T @ self._g_ned  # Specific force
            w[k, :] = w_k  # Angular rate

            # Propagate state vector
            x_dot_k = np.r_[
                x_k[3:6],
                R_bn(x_k[6:10]) @ f[k, :] + self._g_ned,
                T(x_k[6:10]) @ w[k, :],
            ]
            x_k = x_k + x_dot_k * dt

        return x, f, w
