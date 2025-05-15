import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._vectorops import _normalize, _quaternion_product


# class FixedIntervalSmoothing:
#     def __init__(self, ains):
#         self._ains = ains

#     def update(
#         self,
#         f_imu: NDArray,
#         w_imu: NDArray,
#         degrees: bool = False,
#         pos: NDArray | None = None,
#         pos_var: NDArray | None = None,
#         vel: NDArray | None = None,
#         vel_var: NDArray | None = None,
#         head: float | None = None,
#         head_var: float | None = None,
#         head_degrees: bool = True,
#         g_ref: bool = False,
#         g_var: NDArray | None = None,
#     ):
#         f_imu = np.asarray_chkfinite(f_imu).reshape(-1, 3)
#         w_imu = np.asarray_chkfinite(w_imu).reshape(-1, 3)

#         # pos = np.asarray_chkfinite(pos).reshape(-1, 3) if pos else None
#         # pos_var = np.asarray_chkfinite(pos).reshape(-1, 3) if pos_var else None
#         # vel = np.asarray_chkfinite(vel).reshape(-1, 3) if vel else None
#         # vel_var = np.asarray_chkfinite(vel_var).reshape(-1, 3) if vel_var else None
#         # head = np.asarray_chkfinite(head).reshape(-1) if head else None
#         # head_var = np.asarray_chkfinite(head_var).reshape(-1, 3) if head_var else None
#         # g_var = np.asarray_chkfinite(g_var).reshape(-1, 3) if g_var else None

#         if pos is not None:
#             pos = np.asarray_chkfinite(pos).reshape(-1, 3)
#             pos_var = np.asarray_chkfinite(pos_var).reshape(-1, 3)
#         else:
#             pos = [None] * len(f_imu)
#             pos_var = [None] * len(f_imu)
#         if vel is not None:
#             vel = np.asarray_chkfinite(vel).reshape(-1, 3)
#             vel_var = np.asarray_chkfinite(vel_var).reshape(-1, 3)
#         else:
#             vel = [None] * len(f_imu)
#             vel_var = [None] * len(f_imu)
#         if head is not None:
#             head = np.asarray_chkfinite(head).reshape(-1, 1)
#             head_var = np.asarray_chkfinite(head_var).reshape(-1, 1)
#         else:
#             head = [None] * len(f_imu)
#             head_var = [None] * len(f_imu)
#         if g_ref:
#             g_var = np.asarray_chkfinite(g_var).reshape(-1, 3)

#         x = np.zeros((len(f_imu), *self._ains.x.shape))
#         dx = np.zeros((len(f_imu), self._ains.P.shape[0]))
#         phi = np.zeros((len(f_imu), *self._ains.P.shape))
#         P = np.zeros((len(f_imu), *self._ains.P.shape))
#         P_prior = np.zeros((len(f_imu), *self._ains.P.shape))

#         # Forward sweep
#         for k in range(len(f_imu)):
#             P_prior[k, :, :] = self._ains.P_prior
#             phi[k, :, :] = self._ains._I + self._ains._dt * self._ains._F  # state transition matrix
#             self._ains.update(
#                 f_imu[k],
#                 w_imu[k],
#                 degrees=degrees,
#                 pos=pos[k],
#                 pos_var=pos_var[k],
#                 vel=vel[k],
#                 vel_var=vel_var[k],
#                 head=head[k],
#                 head_var=head_var[k],
#                 head_degrees=head_degrees,
#                 g_ref=g_ref,
#                 g_var=g_var,
#             )

#             P[k, :, :] = self._ains.P
#             x[k, :] = self._ains.x
#             dx[k, :] = self._ains._dx_fwd

#         x_fwd = x.copy()
#         P_fwd = P.copy()

#         # Backward sweep
#         dP_prev = np.zeros_like(P[0])
#         for k in range(len(f_imu) - 2, -1, -1):

#             A = P[k] @ phi[k+1].T @ np.linalg.inv(P_prior[k+1])
#             ddx = A @ dx[k+1]
#             dP = A @ dP_prev @ A.T
#             P[k] = P[k] + dP

#             dda = ddx[6:9]
#             ddq = (1.0 / np.sqrt(4.0 + dda.T @ dda)) * np.r_[2.0, dda]
#             x[k, :3] = x[k, :3] + ddx[:3]
#             x[k, 3:6] = x[k, 3:6] + ddx[3:6]
#             x[k, 6:10] = _quaternion_product(x[k, 6:10], ddq)
#             x[k, 6:10] = _normalize(x[k, 6:10])
#             x[k, -3:] = x[k, -3:] + ddx[-3:]
#             if not self._ains._ignore_bias_acc:
#                 x[k, 10:13] = x[k, 10:13] + ddx[9:12]

#             dx[k] = dx[k] + ddx
#             dP_prev = dP

#         self._x_fwd = x_fwd
#         self._P_fwd = P_fwd
#         self._x = x
#         self._P = P


# class FixedIntervalSmoothing:
#     def __init__(self, ains):
#         self._ains = ains
#         self._n_samples = 0
#         self._x = []
#         self._dx = []
#         self._P = []
#         self._P_prior = []
#         self._phi = []

#     def update(self, *args, **kwargs):
#         self._ains.update(*args, **kwargs)
#         self._n_samples += 1
#         self._x.append(self._ains.x)
#         self._dx.append(self._ains._dx_fwd)
#         self._P.append(self._ains.P)
#         self._P_prior.append(self._ains._P_prior_fwd)
#         self._phi.append(self._ains._phi_fwd)

#     def backward_sweep(self):

#         x = np.asarray_chkfinite(self._x)
#         dx = np.asarray_chkfinite(self._dx)
#         P = np.asarray_chkfinite(self._P)
#         P_prior = np.asarray_chkfinite(self._P_prior)
#         phi = np.asarray_chkfinite(self._phi)

#         # Backward sweep
#         dP = np.zeros_like(P[0])
#         for k in range(self._n_samples - 2, -1, -1):

#             A = P[k] @ phi[k + 1].T @ np.linalg.inv(P_prior[k + 1])
#             ddx = A @ dx[k + 1]
#             dx[k] = dx[k] + ddx
#             dP = A @ dP @ A.T
#             P[k] = P[k] + dP

#             dda = ddx[6:9]
#             ddq = (1.0 / np.sqrt(4.0 + dda.T @ dda)) * np.r_[2.0, dda]
#             x[k, :3] = x[k, :3] + ddx[:3]
#             x[k, 3:6] = x[k, 3:6] + ddx[3:6]
#             x[k, 6:10] = _quaternion_product(x[k, 6:10], ddq)
#             x[k, 6:10] = _normalize(x[k, 6:10])
#             x[k, -3:] = x[k, -3:] + ddx[-3:]
#             if not self._ains._ignore_bias_acc:
#                 x[k, 10:13] = x[k, 10:13] + ddx[9:12]

#         self._x_smth = x
#         self._P_smth = P


def backward_sweep(
    x: NDArray,
    dx: NDArray,
    P: NDArray,
    P_prior: NDArray,
    phi: NDArray,
) -> tuple[NDArray, NDArray]:
    """
    Perform a backward sweep using the RTS algorithm for fixed-interval smoothing.

    Parameters
    ----------
    x : NDArray
        The state vector.
    dx : NDArray
        The error state vector.
    P : NDArray
        The covariance matrix.
    P_prior : NDArray
        The a priori covariance matrix.
    phi : NDArray
        The state transition matrix.

    Returns
    -------
    tuple[NDArray, NDArray]
        The smoothed state vector and covariance matrix.
    """

    ignore_bias_acc = dx.shape[1] == 15

    # Backward sweep
    dP = np.zeros_like(P[0])
    for k in range(len(x) - 2, -1, -1):

        A = P[k] @ phi[k + 1].T @ np.linalg.inv(P_prior[k + 1])
        ddx = A @ dx[k + 1]
        dx[k] = dx[k] + ddx
        dP = A @ dP @ A.T
        P[k] = P[k] + dP

        dda = ddx[6:9]
        ddq = (1.0 / np.sqrt(4.0 + dda.T @ dda)) * np.r_[2.0, dda]
        x[k, :3] = x[k, :3] + ddx[:3]
        x[k, 3:6] = x[k, 3:6] + ddx[3:6]
        x[k, 6:10] = _quaternion_product(x[k, 6:10], ddq)
        x[k, 6:10] = _normalize(x[k, 6:10])
        x[k, -3:] = x[k, -3:] + ddx[-3:]
        if not ignore_bias_acc:
            x[k, 10:13] = x[k, 10:13] + ddx[9:12]

    return x, P
