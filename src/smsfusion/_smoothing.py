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

        phi_fwd = np.zeros((len(f_imu), *self._ains.P.shape))
        x_fwd = np.zeros((len(f_imu), *self._ains.x.shape))
        P_fwd = np.zeros((len(f_imu), *self._ains.P.shape))
        x_smth = np.zeros((len(f_imu), *self._ains.x.shape))
        P_smth = np.zeros((len(f_imu), *self._ains.P.shape))
        dx_smth = np.zeros((len(f_imu), self._ains.P.shape[0]))

        # Forward sweep
        for k in range(len(f_imu)):
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

            phi_fwd[k, :, :] = self._ains._phi
            P_fwd[k, :, :] = self._ains.P
            x_fwd[k, :] = self._ains.x

        # Backward sweep
        for k in range(len(f_imu) - 2, -1, -1):
            # A_k = P_fwd[k] @ phi_fwd[k+1].T @ np.linalg.inv(P_fwd[k+1])
            # x_smth[k, :] = x_fwd[k] + A_k @ (x_smth[k+1] - x_fwd[k+1])
            # P_smth[k] = P_fwd[k] + A_k @ (P_smth[k+1] - P_fwd[k+1]) @ A_k.T

            A_k = P_fwd[k] @ phi_fwd[k+1].T @ np.linalg.inv(P_fwd[k+1])
            dx_smth[k] = A_k @ (dx_smth[k+1])
            P_smth[k] = P_fwd[k] + A_k @ (P_smth[k+1] - P_fwd[k+1]) @ A_k.T

            da = dx_smth[k][6:9]
            dq = (1.0 / np.sqrt(4.0 + da.T @ da)) * np.r_[2.0, da]
            x_smth[k, :3] = x_fwd[k, :3] + dx_smth[k, :3]
            x_smth[k, 3:6] = x_fwd[k, 3:6] + dx_smth[k, 3:6]
            x_smth[k, 6:10] = _quaternion_product(x_fwd[k, 6:10], dq)
            x_smth[k, 6:10] = _normalize(x_smth[k, 6:10])
            x_smth[k, -3:] = x_fwd[k, -3:] + dx_smth[k, -3:]
            if not self._ains._ignore_bias_acc:
                x_smth[k, 10:13] = x_fwd[k, 10:13] + dx_smth[k, 9:12]

        self._x_fwd = x_fwd
        self._x = x_smth
        self._P = P_smth
