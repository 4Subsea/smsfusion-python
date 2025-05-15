from warnings import warn

import numpy as np
from numpy.typing import NDArray

from ._ins import AidedINS
from ._vectorops import _normalize, _quaternion_product


class FixedIntervalSmoother:
    """
    Fixed-interval smoothing for AidedINS based on the RTS algorithm (see [1] for
    details).

    References
    ----------
    [1] R. G. Brown and P. Y. C. Hwang, "Random signals and applied Kalman
        filtering with MATLAB exercises", 4th ed. Wiley, pp. 208-212, 2012.
    """

    def __init__(self):
        warn(
            "FixedIntervalSmoother is experimental and may change or be removed in the future.",
            UserWarning,
        )
        self._x, self._dx, self._P, self._P_prior, self._phi = [], [], [], [], []

    def append(self, ains):
        if not isinstance(ains, AidedINS):
            raise TypeError(f"Expected AidedINS instance, got {type(ains).__name__}")
        self._x.append(ains.x)
        self._P.append(ains.P)
        self._dx.append(ains._dx_smth.copy())
        self._P_prior.append(ains._P_prior_smth.copy())
        self._phi.append(ains._phi_smth.copy())

    def clear(self):
        self._x.clear()
        self._dx.clear()
        self._P.clear()
        self._P_prior.clear()
        self._phi.clear()

    def smooth(self):
        """
        Smooths the state estimates and error covariances using a backward sweep
        with the RTS algorithm (see [1] for details).

        References
        ----------
        [1] R. G. Brown and P. Y. C. Hwang, "Random signals and applied Kalman
            filtering with MATLAB exercises", 4th ed. Wiley, pp. 208-212, 2012.
        """
        self._x, self._P = self._backward_sweep(
            self._x, self._dx, self._P, self._P_prior, self._phi
        )

    @property
    def x(self):
        return np.asarray_chkfinite(self._x).copy()

    @property
    def P(self):
        return np.asarray_chkfinite(self._P).copy()

    @staticmethod
    def _backward_sweep(
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

        # Backward sweep
        for k in range(len(x) - 2, -1, -1):

            A = P[k] @ phi[k].T @ np.linalg.inv(P_prior[k + 1])
            ddx = A @ dx[k + 1]
            dx[k, :] += ddx
            P[k, :, :] += A @ (P[k + 1] - P_prior[k + 1]) @ A.T

            dda = ddx[6:9]
            ddq = (1.0 / np.sqrt(4.0 + dda.T @ dda)) * np.r_[2.0, dda]
            x[k, :3] = x[k, :3] + ddx[:3]
            x[k, 3:6] = x[k, 3:6] + ddx[3:6]
            x[k, 6:10] = _quaternion_product(x[k, 6:10], ddq)
            x[k, 6:10] = _normalize(x[k, 6:10])
            x[k, -3:] = x[k, -3:] + ddx[-3:]
            if dx.shape[1] == 15:
                x[k, 10:13] = x[k, 10:13] + ddx[9:12]

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

    ignore_bias_acc = dx.shape[1] == 15

    # Backward sweep
    for k in range(len(x) - 2, -1, -1):

        A = P[k] @ phi[k].T @ np.linalg.inv(P_prior[k + 1])
        ddx = A @ dx[k + 1]
        dx[k, :] += ddx
        P[k, :, :] += A @ (P[k + 1] - P_prior[k + 1]) @ A.T

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
