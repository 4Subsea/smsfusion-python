import numpy as np

from ._pvamekf import PVAMEKF


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
        self._p_nb = np.empty((0, 3), dtype="float64")
        self._v_nb = np.empty((0, 3), dtype="float64")
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
            self._q_nb = np.array(self._q_buf)
            self._bg_b = np.array(self._b_buf)
            self._P = np.array(self._P_buf)
        elif n_samples != len(self._q_nb):
            q_nb, bg_b, P = _rts_backward_sweep(
                self._q_buf,
                self._b_buf,
                self._P_buf,
                self._dtheta_buf,
                self._dx_buf,
                self._mekf._phi,
                self._mekf._Q,
            )
            self._q_nb = np.array(q_nb, dtype="float64")
            self._bg_b = np.array(bg_b, dtype="float64")
            self._P = np.array(P, dtype="float64")

def _rts_backward_sweep(q_nb, bg_b, P, dtheta, dx, phi_k, Q):
    """
    Perform a backward sweep with the Rauch-Tung-Striebel (RTS) algorithm.
    """

    q_nb = q_nb.copy()
    bg_b = bg_b.copy()
    P = P.copy()
    dx = dx.copy()

    # Backward sweep
    n = len(q_nb)
    for k in range(n - 2, -1, -1):

        # Update step k state space and calculate a priori covariance for step k + 1
        _state_transition_update(phi_k, dtheta[k + 1])
        P_prior_kp1 = phi_k @ P[k] @ phi_k.T + Q

        # Smoothed error-state estimate and corresponding covariance
        A = P[k] @ phi_k.T @ np.linalg.inv(P_prior_kp1)
        ddx = A @ dx[k + 1]
        dx[k] += ddx
        P[k] += A @ (P[k + 1] - P_prior_kp1) @ A.T

        _correct_quat_with_gibbs2(q_nb[k], ddx[0:3])
        bg_b[k] += ddx[3:6]

    return q_nb, bg_b, P
