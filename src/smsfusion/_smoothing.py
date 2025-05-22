from warnings import warn

import numpy as np
from numpy.typing import NDArray

from ._ins import AidedINS
from ._vectorops import _normalize, _quaternion_product


class FixedIntervalSmoother:
    """

    Fixed-interval smoother for AidedINS based on the RTS algorithm [1].

    This class stores a time-ordered buffer of state and error covariance estimates
    from an AidedINS instance. After completing the normal forward filtering sweep
    with the AidedINS, the smoother can be used to perform a backward sweep with
    the Rauch-Tung-Striebel (RTS) algorithm [1] to refine the estimates.

    The user is expected to call `append()` at each time step, after `AidedINS.update()`,
    to copy the current states and covariances into the smoother's internal buffer.

    Once all time steps have been appended, call `smooth()` to run fixed-interval
    smoothing. The smoothed state and error covariance estimates are then available
    through the `x` and `P` attributes. Reset the smoother's buffer using
    `clear()` before appending a new interval of data.

    Parameters
    ----------
    ains : AidedINS or AHRS or VRU
        The AINS instance to use for smoothing.
    include_cov : bool, default True
        Whether to include the error covariance matrix, P, in the smoothing process.
        Excluding the covariance matrix will have no impact on the smoothed state
        estimates, and it will speed up the computations. Thus, if smoothed covariance
        estimates are not needed, this parameter may be set to False for improved
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
        Update AINS state estimates with IMU and aiding sensor measurements.

        Parameters
        ----------
        *args : tuple
            Positional arguments to be passed to the AINS update method.
        **kwargs : dict
            Keyword arguments to be passed to the AINS update method.
        """
        self._P_prior_buf.append(self._ains.P_prior)
        self._ains.update(*args, **kwargs)
        self._x_buf.append(self._ains.x)
        self._P_buf.append(self._ains.P)
        self._dx_buf.append(self._ains._dx_smth.copy())
        self._phi_buf.append(self._ains._phi_smth.copy())
        self._is_smoothed = False
        return self

    # def append(self, ains):
    #     """
    #     Copy the current states and error covariances of the given AidedINS instance,
    #     and store them in the smoother's buffer for later smoothing.

    #     Should be called once per time step; i.e., after every update of the AidedINS
    #     instance.

    #     Parameters
    #     ----------
    #     ains : AidedINS or AHRS or VRU
    #         The AidedINS instance to extract the current states and covariance
    #         matrices from. These are stored in the smoother's buffer for later smoothing.
    #     """
    #     if not isinstance(ains, AidedINS):
    #         raise TypeError(f"Expected AidedINS instance, got {type(ains).__name__}")
    #     self._x_buf.append(ains.x)
    #     self._P_buf.append(ains.P)
    #     self._dx_buf.append(ains._dx_smth.copy())
    #     self._P_prior_buf.append(ains._P_prior_smth.copy())
    #     self._phi_buf.append(ains._phi_smth.copy())

    def clear(self):
        """
        Clear the internal buffer of stored AINS states and error covariances.

        Resets the smoother so it is ready for smoothing a new interval of data.
        """
        self._x_buf.clear()
        self._dx_buf.clear()
        self._P_buf.clear()
        self._P_prior_buf.clear()
        self._phi_buf.clear()
        # self._x.clear()
        # self._P.clear()
        self._is_smoothed = False

    # def smooth(self):
    #     """
    #     Perform fixed-interval smoothing of the AINS state and error covariance
    #     estimates using a backward sweep with the Rauch-Tung-Striebel (RTS) algorithm
    #     (see [1] for details).

    #     This method processes the internal buffer of forward-pass estimates that were
    #     collected using `append()` and refines them by incorporating future information.

    #     The smoothed state and error covariance estimates can then be accessed
    #     through the `x` and `P` attributes.

    #     This method should be called only once for each interval of data. Clear
    #     the smoother's buffer using `clear()` if you want to smooth a new interval
    #     of data.

    #     References
    #     ----------
    #     [1] R. G. Brown and P. Y. C. Hwang, "Random signals and applied Kalman
    #         filtering with MATLAB exercises", 4th ed. Wiley, pp. 208-212, 2012.
    #     """
    #     self._x, self._P = self._backward_sweep(
    #         self._x_buf, self._dx_buf, self._P_buf, self._P_prior_buf, self._phi_buf
    #     )

    # def _smooth(self):
    #     if not self._is_smoothed:
    #         self._x, self._P = self._backward_sweep(
    #             self._x_buf,
    #             self._dx_buf,
    #             self._P_buf,
    #             self._P_prior_buf,
    #             self._phi_buf,
    #             include_cov=self._include_cov,
    #         )
    #         self._is_smoothed = True

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
        Smoothed state estimates.

        Note that `smooth()` must be called before these state estimates are updated.

        Returns
        -------
        np.ndarray, shape (N, 15) or (N, 12)
            State estimates for each of the N appended time steps.
        """

        return np.asarray_chkfinite(self._x).copy()

    @property
    @_smooth
    def P(self):
        """
        Smoothed error covariances.

        Note that `smooth()` must be called before these error covariances are updated.

        Returns
        -------
        np.ndarray, shape (N, 15, 15) or (N, 12, 12)
            Error covariances for each of the N appended time steps.
        """
        if not self._include_cov:
            raise ValueError(
                "Covariance matrix is not included in the smoothing process. "
                "Set include_cov=True when initializing the FixedIntervalSmoother."
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
