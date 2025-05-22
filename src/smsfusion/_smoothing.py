from warnings import warn

import numpy as np
from numpy.typing import NDArray

from ._ins import AidedINS
from ._vectorops import _normalize, _quaternion_product


class FixedIntervalSmoother:
    """
    Fixed-interval smoothing layer for AidedINS.

    This class wraps an instance of AidedINS, and stores a time-ordered buffer of
    state and error covariance estimates for each updated time step. A backward sweep
    using the RTS algorithm [1] is performed to refine the Kalman filter estimates
    before returning them to the user.

    Parameters
    ----------
    ains : AidedINS or AHRS or VRU
        The AidedINS (AINS) instance.
    include_cov : bool, default True
        Whether to include the error covariance matrix, P, in the smoothing process.
        Excluding the covariance matrix will have no impact on the smoothed state
        estimates, and it will speed up the computations. Thus, if smoothed covariance
        estimates are not needed, this parameter can be set to ``False`` for improved
        performance.

    References
    ----------
    [1] R. G. Brown and P. Y. C. Hwang, "Random signals and applied Kalman
        filtering with MATLAB exercises", 4th ed. Wiley, pp. 208-212, 2012.
    """

    def __init__(self, ains, include_cov: bool = True):
        warn(
            "FixedIntervalSmoother is experimental and may change or be removed in the future.",
            UserWarning,
        )
        self._ains = ains
        self._include_cov = include_cov
        self._x_buf = []
        self._P_buf = []
        self._dx_buf = []
        self._P_prior_buf = []
        self._phi_buf = []
        self._is_smoothed = False

    def update(self, *args, **kwargs):
        """
        Update the AINS with measurements, and append the current AINS state to
        the smoother's internal buffer.

        Parameters
        ----------
        *args : tuple
            Positional arguments to be passed on to ``ains.update()``.
        **kwargs : dict
            Keyword arguments to be passed on to ``ains.update()``.
        """
        self._P_prior_buf.append(self._ains.P_prior)
        self._ains.update(*args, **kwargs)
        self._x_buf.append(self._ains.x)
        self._P_buf.append(self._ains.P)
        self._dx_buf.append(self._ains._dx_smth.copy())
        self._phi_buf.append(self._ains._phi_smth.copy())
        self._is_smoothed = False
        return self

    def clear(self):
        """
        Clear the internal buffer and reset the smoother so it is ready to process
        a new interval of data.
        """
        self._x_buf.clear()
        self._dx_buf.clear()
        self._P_buf.clear()
        self._P_prior_buf.clear()
        self._phi_buf.clear()
        self._is_smoothed = False

    def _smooth(func):
        def wrapper(self, *args, **kwargs):
            if not self._is_smoothed:
                self._x, self._P = self._backward_sweep(
                    self._x_buf,
                    self._dx_buf,
                    self._P_buf,
                    self._P_prior_buf,
                    self._phi_buf,
                    include_cov=self._include_cov,
                )
                self._is_smoothed = True
            return func(self, *args, **kwargs)
        return wrapper

    @property
    @_smooth
    def x(self):
        """
        Smoothed AINS state estimates.

        Returns
        -------
        np.ndarray, shape (N, 15) or (N, 12)
            State estimates for each of the N time steps where the AINS/smoother
            is updated.
        """

        return np.asarray_chkfinite(self._x).copy()

    @property
    @_smooth
    def P(self):
        """
        Smoothed AINS error covariance matrices.

        Returns
        -------
        np.ndarray, shape (N, 15, 15) or (N, 12, 12)
            Error covariance matrix estimates for each of the N time steps where
            the AINS/smoother is updated.
        """
        if not self._include_cov:
            raise ValueError(
                "Error covariance matrix is excluded from the smoothing process. "
                "Set ``include_cov=True`` during initialization to include it."
            )
        return np.asarray_chkfinite(self._P).copy()

    @staticmethod
    def _backward_sweep(
        x: NDArray,
        dx: NDArray,
        P: NDArray,
        P_prior: NDArray,
        phi: NDArray,
        include_cov: bool,
    ) -> tuple[NDArray, NDArray]:
        """
        Perform a fixed-interval smoothing backward sweep using the RTS algorithm [1].

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
        include_cov : bool
            Whether to include the error covariance matrix in the smoothing process.

        Returns
        -------
        x_smth : NDArray, shape (n_samples, 15) or (n_samples, 12)
            The smoothed state vector.
        P_smth : NDArray, shape (n_samples, 15, 15) or (n_samples, 12, 12)
            The smoothed covariance matrix if include_cov is True, otherwise None.

        References
        ----------
        [1] R. G. Brown and P. Y. C. Hwang, "Random signals and applied Kalman
            filtering with MATLAB exercises", 4th ed. Wiley, pp. 208-212, 2012.
        """

        x = np.asarray_chkfinite(x).copy()
        dx = np.asarray_chkfinite(dx).copy()
        P = np.asarray_chkfinite(P).copy()

        # Backward sweep
        for k in range(len(x) - 2, -1, -1):
            A = P[k] @ phi[k].T @ np.linalg.inv(P_prior[k + 1])
            ddx = A @ dx[k + 1]
            dx[k, :] += ddx
            if include_cov:
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

        if not include_cov:
            P = None

        return x, P
