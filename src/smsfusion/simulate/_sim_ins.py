import numpy as np

from .._ins import gravity
from .._transforms import _angular_matrix_from_quaternion, _rot_matrix_from_quaternion


class IMUSignalFossen:

    def __init__(self, mu=None):
        # self._x0 = np.asarray_chkfinite(x0).reshape(16)
        # self._x = self._x0.copy()
        self._mu = mu
        # x = np.zeros(16)
        # x[6:10] = [1, 0, 0, 0]
        self._g_ned = np.array([0.0, 0.0, gravity(mu)])
        # g_body = _rot_matrix_from_quaternion(x[6:10]).T @ g_ned

    def simulate(self, fs, n):
        dt = 1.0 / fs
        t = np.arange(0.0, n * dt, dt)

        # Initial state
        x0 = np.zeros(16)
        x0[6:10] = np.asarray([1, 0, 0, 0])

        # Simulate
        x_pred = x0
        x = np.zeros((n, 16))
        # x[0, :] = x0
        acc = np.zeros((n, 3))
        f = np.zeros((n, 3))
        w = np.zeros((n, 3))
        for k in range(0, n):

            x[k, :] = x_pred

            # Rotation matrix from body to NED
            R_bn_k = _rot_matrix_from_quaternion(x[k, 6:10])
            T_k = _angular_matrix_from_quaternion(x[k, 6:10])

            # Gravity vector in body frame
            g_body_k = R_bn_k.T @ self._g_ned

            # Acceleration
            acc[k, :] = np.array(
                [
                    0.1 * np.sin(0.1 * t[k]),
                    0.1 * np.cos(0.1 * t[k]),
                    0.05 * np.sin(0.05 * t[k]),
                ]
            )

            # Specific force (i.e., acceleration - gravity)
            f[k, :] = acc[k, :] - g_body_k

            # Rotation rate
            w[k, :] = np.array(
                [
                    0.01 * np.cos(0.2 * t[k]),
                    -0.02 * np.sin(0.1 * t[k]),
                    0.01 * np.sin(0.1 * t[k]),
                ]
            )

            x_dot_k = np.r_[
                x[k, 3:6], R_bn_k @ f[k, :] + self._g_ned, T_k @ w[k, :], np.zeros(6)
            ]

            # Propagate signal vector
            x_pred = x[k, :] + x_dot_k * dt

        return x, f, w
