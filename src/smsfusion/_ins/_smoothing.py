import numpy as np

from ._common import _correct_quat_with_gibbs2
from ._pvamekf import PVAMEKF, _state_transition_matrix_update
from .._transforms import _rot_matrix_from_quaternion


class FixedIntervalSmoother:
    def __init__(self, mekf: PVAMEKF):
        self._mekf = mekf

        # Buffers with estimates from the forward pass
        self._p_buf = []
        self._v_buf = []
        self._q_buf = []
        self._bg_buf = []
        self._dx_buf = []
        self._P_buf = []
        self._dvel_buf = []
        self._dtheta_buf = []

        # Smoothed state and covariance estimates
        self._p_n = np.empty((0, 3), dtype="float64")
        self._v_n = np.empty((0, 3), dtype="float64")
        self._q_nb = np.empty((0, 4), dtype="float64")
        self._bg_b = np.empty((0, 3), dtype="float64")
        self._P = np.empty((0, *self._mekf._P.shape), dtype="float64")

    def update(self, *args, **kwargs):
        """
        Update with IMU and aiding measurements.
        """
        self._mekf.update(*args, **kwargs)
        self._p_buf.append(self._mekf.position())
        self._v_buf.append(self._mekf.velocity())
        self._q_buf.append(self._mekf.quaternion())
        self._bg_buf.append(self._mekf.bias_gyro(degrees=False))
        self._dx_buf.append(self._mekf._dx.copy())
        self._P_buf.append(self._mekf.P)
        self._dvel_buf.append(self._mekf._dvel.copy())
        self._dtheta_buf.append(self._mekf._dtheta.copy())
        return self

    def _smooth(self):
        n_samples = len(self._q_buf)

        if n_samples == 0:
            pass
        elif n_samples == 1:
            self._p_n = np.array(self._p_buf)
            self._v_n = np.array(self._v_buf)
            self._q_nb = np.array(self._q_buf)
            self._bg_b = np.array(self._bg_buf)
            self._P = np.array(self._P_buf)
        elif n_samples != len(self._p_n):
            p_n, v_n, q_nb, bg_b, P = _rts_backward_sweep(
                self._p_buf,
                self._v_buf,
                self._q_buf,
                self._bg_buf,
                self._P_buf,
                self._dx_buf,
                self._dvel_buf,
                self._dtheta_buf,
                self._mekf._phi,
                self._mekf._Q,
            )
            self._p_n = np.array(p_n, dtype="float64")
            self._v_n = np.array(v_n, dtype="float64")
            self._q_nb = np.array(q_nb, dtype="float64")
            self._bg_b = np.array(bg_b, dtype="float64")
            self._P = np.array(P, dtype="float64")


def _rts_backward_sweep(p_n, v_n, q_nb, bg_b, P, dx, dvel, dtheta, phi_k, Q):
    """
    Perform a backward sweep with the Rauch-Tung-Striebel (RTS) algorithm.
    """

    p_n = p_n.copy()
    v_n = v_n.copy()
    q_nb = q_nb.copy()
    bg_b = bg_b.copy()
    P = P.copy()
    dx = dx.copy()

    # Backward sweep
    n = len(q_nb)
    for k in range(n - 2, -1, -1):

        # Update step k state space and calculate a priori covariance for step k + 1
        R_nb = _rot_matrix_from_quaternion(q_nb[k])
        _state_transition_matrix_update(phi_k, dtheta[k + 1], dtheta[k + 1], R_nb)
        P_prior_kp1 = phi_k @ P[k] @ phi_k.T + Q

        # Smoothed error-state estimate and corresponding covariance
        A = P[k] @ phi_k.T @ np.linalg.inv(P_prior_kp1)
        ddx = A @ dx[k + 1]
        dx[k] += ddx
        P[k] += A @ (P[k + 1] - P_prior_kp1) @ A.T

        # Update smoothed state estimates
        p_n[k] += ddx[0:3]
        v_n[k] += ddx[3:6]
        _correct_quat_with_gibbs2(q_nb[k], ddx[6:9])
        bg_b[k] += ddx[9:12]

    return p_n, v_n, q_nb, bg_b, P
