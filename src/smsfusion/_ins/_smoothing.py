import numpy as np
from numba import njit
from numpy.typing import NDArray

from .._transforms import _euler_from_quaternion, _rot_matrix_from_quaternion
from ._common import _update_quaternion_with_gibbs2
from ._pvamekf import PVAMEKF, _state_transition_matrix_update


class FixedIntervalSmoother:
    """
    Fixed-interval smoothing for PVAMEKF.

    This class wraps an instance of PVAMEKF, and maintains a time-ordered buffer
    of state and error covariance estimates as measurements are processed via
    the ``update()`` method. A backward sweep over the buffered data using the
    Rauch-Tung-Striebel (RTS) algorithm [1] is performed to refine the filter
    estimates.

    Parameters
    ----------
    mekf : PVAMEKF
        The underlying PVAMEKF instance used for forward filtering.
    cov_smoothing : bool, default True
        Whether to include the error covariance matrix, `P`, in the smoothing process.
        Disabling the covariance smoothing has no effect on the smoothed state estimates,
        and can reduce computation time if smoothed covariances are not required.

    References
    ----------
    [1] R. G. Brown and P. Y. C. Hwang, "Random signals and applied Kalman
        filtering with MATLAB exercises", 4th ed. Wiley, pp. 208-212, 2012.
    """

    def __init__(self, mekf: PVAMEKF, cov_smoothing: bool = True):
        self._mekf = mekf
        self._mekf._keep_smoothing_params = True
        self._cov_smoothing = cov_smoothing

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
        self._P_buf.append(self._mekf.P)
        self._dx_buf.append(self._mekf._dx_copy)
        self._dvel_buf.append(self._mekf._dvel_copy)
        self._dtheta_buf.append(self._mekf._dtheta_copy)
        return self

    def _smooth(self):
        n_samples = len(self._q_buf)
        if n_samples != len(self._p_n):
            self._p_n, self._v_n, self._q_nb, self._bg_b, self._P = _rts_backward_sweep(
                np.array(self._p_buf),
                np.array(self._v_buf),
                np.array(self._q_buf),
                np.array(self._bg_buf),
                np.array(self._P_buf),
                np.array(self._dx_buf),
                self._dvel_buf,
                self._dtheta_buf,
                self._mekf._phi,
                self._mekf._Q,
                self._cov_smoothing,
            )

    def quaternion(self) -> NDArray[np.float64]:
        """
        Smoothed quaternion estimates.

        Returns
        -------
        np.ndarray, shape (N, 4)
            Quaternion estimates for each of the N time steps where the smoother has
            been updated with measurements.
        """
        self._smooth()
        return self._q_nb.copy()

    def euler(self, degrees: bool = False):
        """
        Smoothed Euler angles estimates.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Euler angles estimates for each of the N time steps where the smoother has
            been updated with measurements.
        """
        self._smooth()
        if self._q_nb.size == 0:
            return np.empty((0, 3), dtype="float64")

        theta = np.array([_euler_from_quaternion(q_i) for q_i in self._q_nb])

        return np.degrees(theta) if degrees else theta

    def position(self) -> NDArray[np.float64]:
        """
        Smoothed position estimates.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Position estimates for each of the N time steps where the smoother has
            been updated with measurements.
        """
        self._smooth()
        return self._p_n.copy()

    def velocity(self) -> NDArray[np.float64]:
        """
        Smoothed velocity estimates.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Velocity estimates for each of the N time steps where the smoother has
            been updated with measurements.
        """
        self._smooth()
        return self._v_n.copy()

    def bias_gyro(self) -> NDArray[np.float64]:
        """
        Smoothed gyroscope bias estimates.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Gyroscope bias estimates for each of the N time steps where the smoother has
            been updated with measurements.
        """
        self._smooth()
        return self._bg_b.copy()

    @property
    def P(self) -> NDArray[np.float64]:
        """
        Smoothed error covariance estimates.

        Returns
        -------
        np.ndarray, shape (N, 12, 12)
            Error covariance estimates for each of the N time steps where the smoother has
            been updated with measurements.
        """
        self._smooth()
        return self._P.copy()


@njit  # type: ignore[misc]
def _rts_backward_sweep(
    p_n: NDArray[np.float64],
    v_n: NDArray[np.float64],
    q_nb: NDArray[np.float64],
    bg_b: NDArray[np.float64],
    P: NDArray[np.float64],
    dx: NDArray[np.float64],
    dvel: NDArray[np.float64],
    dtheta: NDArray[np.float64],
    phi_k: NDArray[np.float64],
    Q: NDArray[np.float64],
    cov_smoothing: bool = True,
):
    """
    Perform a backward sweep with the Rauch-Tung-Striebel (RTS) algorithm.
    """

    # Backward sweep
    n = len(q_nb)
    for k in range(n - 2, -1, -1):

        # Update state space model for step k
        R_nb_k = _rot_matrix_from_quaternion(q_nb[k])
        _state_transition_matrix_update(phi_k, dvel[k + 1], dtheta[k + 1], R_nb_k)

        # Calculate a priori error covariance matrix for step k + 1
        P_prior_kp1 = phi_k @ P[k] @ phi_k.T + Q

        # Smoothed error-state estimate and corresponding covariance
        A = P[k] @ phi_k.T @ np.linalg.inv(P_prior_kp1)
        ddx_k = A @ dx[k + 1]
        dx[k] += ddx_k
        if cov_smoothing:
            P[k] += A @ (P[k + 1] - P_prior_kp1) @ A.T

        # Update smoothed state estimates
        p_n[k] += ddx_k[0:3]
        v_n[k] += ddx_k[3:6]
        _update_quaternion_with_gibbs2(q_nb[k], ddx_k[6:9])
        bg_b[k] += ddx_k[9:12]

    return p_n, v_n, q_nb, bg_b, P
