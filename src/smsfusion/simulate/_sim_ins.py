import numpy as np

from .._ins import gravity
from .._transforms import _angular_matrix_from_quaternion, _rot_matrix_from_quaternion


def pva_signal_202402A(fs, n):
    dt = 1.0 / fs
    t = np.arange(0.0, n * dt, dt)

    g_ned = np.array([0.0, 0.0, gravity()])

    # Alias for transformation nmatrices
    R_bn = _rot_matrix_from_quaternion  # rotation matrix body-to-ned
    T = _angular_matrix_from_quaternion  # angular rates to quaternion rates

    # Linear acceleration signal (expressed in body frame)
    acc = np.array(
        [
            0.1 * np.sin(0.1 * t),
            0.1 * np.cos(0.1 * t),
            0.05 * np.sin(0.05 * t),
        ]
    )

    # Angular rate signal (expressed in body frame)
    ang_rate = np.array(
        [
            0.01 * np.cos(0.2 * t),
            -0.02 * np.sin(0.1 * t),
            0.01 * np.sin(0.1 * t),
        ]
    )

    # Simulate
    x0 = np.zeros(10)
    x0[6:10] = [1.0, 0.0, 0.0, 0.0]
    x_pred = x0
    x, f, w = [], [], []
    for k, (a_k, w_k) in enumerate(zip(acc.T, ang_rate.T)):

        # Update current state
        x_k = x_pred
        v_k = x_k[3:6]
        q_k = x_k[6:10]
        x.append(x_k)

        # Specific force
        f_k = a_k - R_bn(x_k[6:10]).T @ g_ned
        f.append(f_k)

        # Propagate state vector
        x_dot_k = np.r_[v_k, R_bn(q_k) @ a_k, T(q_k) @ w_k]
        x_pred = x_k + x_dot_k * dt

    x = np.asarray(x).reshape(-1, 10)
    f = np.asarray(f).reshape(-1, 3)
    w = ang_rate.reshape(-1, 3)

    return x, f, w


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
