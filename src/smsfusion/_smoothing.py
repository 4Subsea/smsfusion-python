import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._vectorops import _normalize, _quaternion_product


class FixedIntervalSmoother:
    def __init__(self, ains):
        self._ains = ains

    def smooth(
        self,
        f_imu: NDArray,
        w_imu: NDArray,
        degrees: bool = False,
        pos: NDArray | None = None,
        pos_var: NDArray | None = None,
        vel: NDArray | None = None,
        vel_var: NDArray | None = None,
        head: float | None = None,
        head_var: float | None = None,
        head_degrees: bool = True,
        g_ref: bool = False,
        g_var: NDArray | None = None,
    ):
        f_imu = np.asarray_chkfinite(f_imu).reshape(-1, 3)
        w_imu = np.asarray_chkfinite(w_imu).reshape(-1, 3)
        if pos is not None:
            pos = np.asarray_chkfinite(pos).reshape(-1, 3)
            pos_var = np.asarray_chkfinite(pos_var).reshape(-1, 3)
        else:
            pos = [None] * len(f_imu)
            pos_var = [None] * len(f_imu)
        if vel is not None:
            vel = np.asarray_chkfinite(vel).reshape(-1, 3)
            vel_var = np.asarray_chkfinite(vel_var).reshape(-1, 3)
        else:
            vel = [None] * len(f_imu)
            vel_var = [None] * len(f_imu)
        if head is not None:
            head = np.asarray_chkfinite(head).reshape(-1, 1)
            head_var = np.asarray_chkfinite(head_var).reshape(-1, 1)
        else:
            head = [None] * len(f_imu)
            head_var = [None] * len(f_imu)
        if g_ref:
            g_var = np.asarray_chkfinite(g_var).reshape(-1, 3)

        x = np.zeros((len(f_imu), *self._ains.x.shape))
        dx = np.zeros((len(f_imu), self._ains.P.shape[0]))
        phi = np.zeros((len(f_imu), *self._ains.P.shape))
        P = np.zeros((len(f_imu), *self._ains.P.shape))
        P_prior = np.zeros((len(f_imu), *self._ains.P.shape))

        # Forward sweep
        for k in range(len(f_imu)):
            P_prior[k, :, :] = self._ains.P_prior
            phi[k, :, :] = self._ains._I + self._ains._dt * self._ains._F  # state transition matrix
            self._ains.update(
                f_imu[k],
                w_imu[k],
                degrees=degrees,
                pos=pos[k],
                pos_var=pos_var[k],
                vel=vel[k],
                vel_var=vel_var[k],
                head=head[k],
                head_var=head_var[k],
                head_degrees=head_degrees,
                g_ref=g_ref,
                g_var=g_var,
            )

            P[k, :, :] = self._ains.P
            x[k, :] = self._ains.x
            dx[k, :] = self._ains._dx_fwd

        x_fwd = x.copy()
        P_fwd = P.copy()

        # Backward sweep
        dP = np.zeros_like(P[0])
        for k in range(len(f_imu) - 2, -1, -1):

            A = P[k] @ phi[k+1].T @ np.linalg.inv(P_prior[k+1])
            ddx = A @ dx[k+1]  # error-state smoothing correction
            dP = A @ dP @ A.T
            P[k] = P[k] + dP

            # Update total state
            dda = ddx[6:9]
            ddq = (1.0 / np.sqrt(4.0 + dda.T @ dda)) * np.r_[2.0, dda]
            x[k, :3] = x[k, :3] + ddx[:3]
            x[k, 3:6] = x[k, 3:6] + ddx[3:6]
            x[k, 6:10] = _quaternion_product(x[k, 6:10], ddq)
            x[k, 6:10] = _normalize(x[k, 6:10])
            x[k, -3:] = x[k, -3:] + ddx[-3:]
            if not self._ains._ignore_bias_acc:
                x[k, 10:13] = x[k, 10:13] + ddx[9:12]

            dx[k] = dx[k] + ddx

        self._x_fwd = x_fwd
        self._P_fwd = P_fwd
        self._x = x
        self._P = P
