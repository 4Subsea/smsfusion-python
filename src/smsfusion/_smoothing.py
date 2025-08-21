from typing import Self
from warnings import warn

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ._ins import AHRS, VRU, AidedINS
from ._transforms import _euler_from_quaternion
from ._vectorops import _normalize, _quaternion_product


class FixedIntervalSmoother:
    """
    Fixed-interval smoothing for AidedINS.

    This class wraps an instance of AidedINS (or a subclass like AHRS or VRU),
    and maintains a time-ordered buffer of state and error covariance estimates
    as measurements are processed via the ``update()`` method. A backward sweep
    over the buffered data using the Rauch-Tung-Striebel (RTS) algorithm [1] is
    performed to refine the filter estimates.

    Parameters
    ----------
    ains : AidedINS or AHRS or VRU
        The underlying AidedINS instance used for forward filtering.
    cov_smoothing : bool, default True
        Whether to include the error covariance matrix, `P`, in the smoothing process.
        Disabling the covariance smoothing has no effect on the smoothed state estimates,
        and can reduce computation time if smoothed covariances are not required.

    References
    ----------
    [1] R. G. Brown and P. Y. C. Hwang, "Random signals and applied Kalman
        filtering with MATLAB exercises", 4th ed. Wiley, pp. 208-212, 2012.
    """

    def __init__(self, ains: AidedINS | AHRS | VRU, cov_smoothing: bool = True) -> None:
        warn(
            "FixedIntervalSmoother is experimental and may change or be removed in the future.",
            UserWarning,
        )
        self._ains = ains
        self._cov_smoothing = cov_smoothing

        # Buffers for storing state and covariance estimates from forward sweep
        self._x_buf = []  # state estimates (w/o smoothing)
        self._P_buf = []  # error covariance estimates (w/o smoothing)
        self._dx_buf = []  # error-state estimates (w/o smoothing)
        self._P_prior_buf = []  # a priori error covariance estimates (w/o smoothing)
        self._phi_buf = []  # state transition matrix

        # Smoothed state and covariance estimates
        self._x = np.empty((0, 16), dtype="float64")
        self._P = np.empty((0, *self._ains.P.shape), dtype="float64")

    @property
    def ains(self) -> AidedINS | AHRS | VRU:
        """
        The underlying AidedINS instance used for forward filtering.

        Returns
        -------
        AidedINS or AHRS or VRU
            The AidedINS instance.
        """
        return self._ains

    def update(self, *args, **kwargs) -> Self:
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
        self._dx_buf.append(self._ains._dx.copy())
        self._phi_buf.append(self._ains._phi.copy())
        return self

    def clear(self) -> None:
        """
        Clear the internal buffer of state estimates. This resets the smoother,
        and prepares for a new interval of measurements.
        """
        self._x_buf.clear()
        self._dx_buf.clear()
        self._P_buf.clear()
        self._P_prior_buf.clear()
        self._phi_buf.clear()

    def _smooth(self):
        n_samples = len(self._x_buf)
        if n_samples == 0:
            self._x = np.empty((0, 16), dtype="float64")
            self._P = np.empty((0, *self._ains.P.shape), dtype="float64")
        elif n_samples == 1:
            self._x = np.asarray(self._x_buf)
            self._P = np.asarray(self._P_buf)
        elif n_samples != len(self._x):
            x, P = _rts_backward_sweep(
                self._x_buf,
                self._dx_buf,
                self._P_buf,
                self._P_prior_buf,
                self._phi_buf,
                self._cov_smoothing,
            )
            self._x = np.asarray(x)
            self._P = np.asarray(P)

    @property
    def x(self) -> NDArray:
        """
        Smoothed state vector estimates.

        Returns
        -------
        np.ndarray, shape (N, 15) or (N, 12)
            State estimates for each of the N time steps where the smoother has
            been updated with measurements.
        """
        self._smooth()
        return self._x.copy()

    @property
    def P(self) -> NDArray:
        """
        Error covariance matrix estimates.

        If ``cov_smoothing=True``, smoothed error covariance estimates are returned.
        Otherwise, the forward filter covariance estimates are returned.

        Returns
        -------
        np.ndarray, shape (N, 15, 15) or (N, 12, 12)
            Error covariance matrix estimates for each of the N time steps where
            the smoother has been updated with measurements.
        """
        self._smooth()
        return self._P.copy()

    def position(self) -> NDArray:
        """
        Smoothed position estimates.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Position estimates for each of the N time steps where the smoother has
            been updated with measurements.
        """
        x = self.x
        if x.size == 0:
            return np.empty((0, 3), dtype="float64")
        return self.x[:, :3]

    def velocity(self) -> NDArray:
        """
        Smoothed velocity estimates.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Velocity estimates for each of the N time steps where the smoother has
            been updated with measurements.
        """
        x = self.x
        if x.size == 0:
            return np.empty((0, 3), dtype="float64")
        return self.x[:, 3:6]

    def quaternion(self) -> NDArray:
        """
        Smoothed unit quaternion estimates.

        Returns
        -------
        np.ndarray, shape (N, 4)
            Unit quaternion estimates for each of the N time steps where the smoother has
            been updated with measurements.
        """
        x = self.x
        if x.size == 0:
            return np.empty((0, 4), dtype="float64")
        return self.x[:, 6:10]

    def bias_acc(self) -> NDArray:
        """
        Smoothed accelerometer bias estimates.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Accelerometer bias estimates for each of the N time steps where the smoother has
            been updated with measurements.
        """
        x = self.x
        if x.size == 0:
            return np.empty((0, 3), dtype="float64")
        return x[:, 10:13]

    def bias_gyro(self, degrees: bool = False) -> NDArray:
        """
        Smoothed gyroscope bias estimates.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Gyroscope bias estimates for each of the N time steps where the smoother has
            been updated with measurements.
        """
        x = self.x
        if x.size == 0:
            return np.empty((0, 3), dtype="float64")

        bg = self.x[:, 13:16]
        return np.degrees(bg) if degrees else bg

    def euler(self, degrees: bool = False) -> NDArray:
        """
        Smoothed Euler angles estimates.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Euler angles estimates for each of the N time steps where the smoother has
            been updated with measurements.
        """
        q = self.quaternion()
        if q.size == 0:
            return np.empty((0, 3), dtype="float64")

        theta = np.array([_euler_from_quaternion(q_i) for q_i in q])
        return np.degrees(theta) if degrees else theta


@njit  # type: ignore[misc]
def _rts_backward_sweep(
    x: list[NDArray],
    dx: list[NDArray],
    P: list[NDArray],
    P_prior: list[NDArray],
    phi: list[NDArray],
    cov_smoothing: bool,
) -> tuple[list[NDArray], list[NDArray]]:
    """
    Perform a backward sweep with the RTS algorithm [1].

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
    cov_smoothing : bool
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

    x = x.copy()
    dx = dx.copy()
    P = P.copy()

    q_prealloc = np.array([2.0, 0.0, 0.0, 0.0])  # Preallocation

    # Backward sweep
    for k in range(len(x) - 2, -1, -1):
        # Smoothed error-state estimate and corresponding covariance
        A = P[k] @ phi[k].T @ np.linalg.inv(P_prior[k + 1])
        ddx = A @ dx[k + 1]
        dx[k] += ddx
        if cov_smoothing:
            P[k] += A @ (P[k + 1] - P_prior[k + 1]) @ A.T

        # Reset
        dda = ddx[6:9]
        q_prealloc[1:] = dda
        ddq = (1.0 / np.sqrt(4.0 + dda.T @ dda)) * q_prealloc
        x[k][:3] = x[k][:3] + ddx[:3]
        x[k][3:6] = x[k][3:6] + ddx[3:6]
        x[k][6:10] = _quaternion_product(x[k][6:10], ddq)
        x[k][6:10] = _normalize(x[k][6:10])
        x[k][-3:] = x[k][-3:] + ddx[-3:]
        if dx[k].size == 15:
            x[k][10:13] = x[k][10:13] + ddx[9:12]

    return x, P
