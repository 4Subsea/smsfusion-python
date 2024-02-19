import numpy as np

from .._ins import gravity
from .._transforms import _angular_matrix_from_quaternion, _rot_matrix_from_quaternion


class IMUSignal202402A:
    """
    IMU signal as implemented by Fossen.
    """

    def __init__(self, mu=None):
        self._mu = mu
        self._g_ned = np.array([0.0, 0.0, gravity(mu)])

    def simulate(self, x0, fs, n):
        dt = 1.0 / fs
        t = np.arange(0.0, n * dt, dt)
        x0 = np.asarray_chkfinite(x0).reshape(10)

        # Simulate
        x_pred = x0
        x = np.zeros((n, 10))
        f = np.zeros((n, 3))
        w = np.zeros((n, 3))
        for k in range(0, n):

            pos_k = x_pred[0:3]
            vel_k = x_pred[3:6]
            q_k = x_pred[6:10]

            # Rotation matrix from body to NED
            R_bn_k = _rot_matrix_from_quaternion(q_k)
            T_k = _angular_matrix_from_quaternion(q_k)

            # Gravity vector in body frame
            g_body_k = R_bn_k.T @ self._g_ned

            # Acceleration
            acc_k = np.array(
                [
                    0.1 * np.sin(0.1 * t[k]),
                    0.1 * np.cos(0.1 * t[k]),
                    0.05 * np.sin(0.05 * t[k]),
                ]
            )

            # Rotation rate
            gyro_k = np.array(
                [
                    0.01 * np.cos(0.2 * t[k]),
                    -0.02 * np.sin(0.1 * t[k]),
                    0.01 * np.sin(0.1 * t[k]),
                ]
            )

            x[k, :] = np.r_[pos_k, vel_k, q_k]
            f[k, :] = acc_k - g_body_k  # Specific force (i.e., acceleration - gravity)
            w[k, :] = gyro_k

            # Propagate signal vector
            x_dot_k = np.r_[vel_k, R_bn_k @ f[k, :] + self._g_ned, T_k @ w[k, :]]
            x_pred = x[k, :] + x_dot_k * dt

        return x, f, w
