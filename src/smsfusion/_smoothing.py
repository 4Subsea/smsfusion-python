from warnings import warn

import numpy as np
from numpy.typing import NDArray

from ._vectorops import _normalize, _quaternion_product


class FixedIntervalSmoother:
    def __init__(self, ains):
        warn(
            "FixedIntervalSmoother is experimental and may change or be removed in the future.",
            UserWarning,
        )
        self._ains = ains
        self._x, self._dx, self._P, self._P_prior, self._phi = [], [], [], [], []

    def update(self, *args, **kwargs):
        self._P_prior.append(self._ains.P_prior)
        self._ains.update(*args, **kwargs)
        self._x.append(self._ains.x)
        self._dx.append(self._ains._dx_fwd)
        self._P.append(self._ains.P)
        self._phi.append(self._ains._phi_fwd)

    def smooth(self):
        x, P = backward_sweep(self._x, self._dx, self._P, self._P_prior, self._phi)
        return x, P


def backward_sweep(
    x: NDArray,
    dx: NDArray,
    P: NDArray,
    P_prior: NDArray,
    phi: NDArray,
) -> tuple[NDArray, NDArray]:
    """
    Perform a fixed-interval smoothing backward sweep using the RTS algorithm.

    Parameters
    ----------
    x : NDArray, shape (n_samples, 16)
        The state vector.
    dx : NDArray, shape (n_samples, 15) or (n_samples, 12)
        The error state vector.
    P : NDArray, shape (n_samples, 15, 15) or (n_samples, 12, 12)
        The covariance matrix.
    P_prior : NDArray, shape (n_samples, 15, 15) or (n_samples, 12, 12)
        The a priori covariance matrix.
    phi : NDArray, shape (n_samples, 15, 15) or (n_samples, 12, 12)
        The state transition matrix.

    Returns
    -------
    tuple[NDArray, NDArray]
        The smoothed state vector and covariance matrix.
    """

    x = np.asarray_chkfinite(x).copy()
    dx = np.asarray_chkfinite(dx).copy()
    P = np.asarray_chkfinite(P).copy()
    P_prior = np.asarray_chkfinite(P_prior).copy()
    phi = np.asarray_chkfinite(phi).copy()

    ignore_bias_acc = dx.shape[1] == 15

    # Backward sweep
    for k in range(len(x) - 2, -1, -1):

        A = P[k] @ phi[k].T @ np.linalg.inv(P_prior[k + 1])
        ddx = A @ dx[k + 1]
        dx[k, :] += ddx
        P[k, :, :] += A @ (P[k+1] - P_prior[k+1]) @ A.T

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
