from typing import Self

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from smsfusion._transforms import _euler_from_quaternion
from smsfusion._vectorops import _skew_symmetric

from ._aiding import _aiding_update_gref, _aiding_update_head
from ._common import (
    _dhda_head,
    _gref_b_from_quat,
    _nz2vg,
    _project_covariance_ahead,
    _update_quaternion_with_gibbs2,
    _update_quaternion_with_rotvec,
)

_P0 = (
    (1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0e-6, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0e-6, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 1.0e-6, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-6),
)


@njit  # type: ignore[misc]
def _state_transition_matrix_init(
    dt: float,
    dtheta: NDArray[np.float64],
    gbc: float,
) -> NDArray[np.float64]:
    """
    State transition matrix.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    dtheta : ndarray, shape (3,)
        Attitude increment measurement (coning integral).
    gbc : float
        Gyro bias correlation time in seconds.

    Returns
    -------
    ndarray, shape (6, 6)
        State transition matrix.
    """
    phi = np.eye(6)
    phi[0:3, 0:3] -= _skew_symmetric(dtheta)  # NB! update each time step
    phi[0:3, 3:6] -= dt * np.eye(3)
    phi[3:6, 3:6] -= dt * np.eye(3) / gbc
    return phi


@njit  # type: ignore[misc]
def _state_transition_matrix_update(
    phi: NDArray[np.float64],
    dtheta: NDArray[np.float64],
) -> None:
    """
    Update the state transition matrix in place.

    Parameters
    ----------
    phi : ndarray, shape (6, 6)
        State transition matrix to be updated in place.
    dtheta : ndarray, shape (3,)
        Attitude increment measurement (coning integral).
    """
    dtx, dty, dtz = dtheta

    # phi[0:3, 0:3] = np.eye(3) - dt * S(w_b)
    phi[0, 1] = dtz
    phi[0, 2] = -dty
    phi[1, 0] = -dtz
    phi[1, 2] = dtx
    phi[2, 0] = dty
    phi[2, 1] = -dtx


@njit  # type: ignore[misc]
def _process_noise_covariance_matrix(
    dt: float, arw: float, gbs: float, gbc: float
) -> NDArray[np.float64]:
    """
    Process noise covariance matrix.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    arw : float
        Angular random walk (gyroscope noise density) in rad/√Hz.
    gbs : float
        Gyro bias stability (bias instability) in rad/s.
    gbc : float
        Gyro bias correlation time in seconds.

    Returns
    -------
    Q : ndarray, shape (6, 6)
        Process noise covariance matrix.
    """
    Q = np.zeros((6, 6))
    Q[0:3, 0:3] = dt * arw**2 * np.eye(3)
    Q[3:6, 3:6] = dt * (2.0 * gbs**2 / gbc) * np.eye(3)
    return Q


def _measurement_matrix_init(
    q_nb: NDArray[np.float64], nav_frame_factor: float
) -> NDArray[np.float64]:
    """
    Measurement matrix.

    Parameters
    ----------
    q_nb : ndarray, shape (4,)
        Unit quaternion.
    nav_frame_factor: float
        Gravity direction along the navigation frame's z-axis. +1.0 for 'NED' and
        -1.0 for 'ENU'.

    Returns
    -------
    ndarray, shape (4, 6)
        Linearized measurement matrix.
    """
    vg_b = _gref_b_from_quat(q_nb, nav_frame_factor)  # gravity reference vector
    H = np.zeros((4, 6))
    H[0:1, 0:3] = _dhda_head(q_nb)  # heading
    H[1:4, 0:3] = _skew_symmetric(vg_b)  # gravity reference vector
    return H


@njit  # type: ignore[misc]
def _reset(
    dx: NDArray[np.float64], q_nb: NDArray[np.float64], bg_b: NDArray[np.float64]
) -> None:
    """
    Reset state (in place).

    Parameters
    ----------
    dx : ndarray, shape (6,)
        Error state vector containing the corrections to be applied to the state
        estimates. Will be reset to zero after applying the corrections.
    q_nb : ndarray, shape (4,)
        Attitude state estimate parameterized as a unit quaternion to be reset in place.
    bg_b : ndarray, shape (3,)
        Gyroscope bias state estimate to be reset in place.
    """
    _update_quaternion_with_gibbs2(q_nb, dx[0:3])  # -> update q_nb (in place)
    bg_b[:] += dx[3:6]
    dx[:] = 0.0


class VRU:
    """
    Vertical Reference Unit (VRU).

    This class provides attitude and gyro bias estimation using a multiplicative
    extended Kalman filter (MEKF).

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    q0 : array_like, shape (4,), optional
        Initial attitude estimate as a unit quaternion (qw, qx, qy, qz). Defaults
        to the identity quaternion (1.0, 0.0, 0.0, 0.0) (i.e., no rotation).
    bg0 : array_like, shape (3,), optional
        Initial gyroscope bias estimate (bgx, bgy, bgz) in rad/s. Defaults to zero bias.
    P0 : array_like, shape (6, 6), optional
        Initial (a priori) estimate of the error covariance matrix. Defaults to
        a small diagonal matrix (1e-6 * np.eye(6)).
    gyro_noise_density : float, optional
        Gyroscope noise density (angular random walk) in (rad/s)/√Hz. Defaults to
        0.00005 (rad/s)/√Hz (SMS Motion 2 noise level).
    gyro_bias_stability : float, optional
        Gyroscope bias stability in rad/s. Defaults to 0.00005 rad/s (SMS Motion 2
        noise level).
    gyro_bias_corr_time : float, optional
        Gyroscope bias correlation time in seconds. Defaults to 50.0 s.
    nav_frame : {'NED', 'ENU'}, optional
        Specifies the assumed inertial-like 'navigation' frame. Should be 'NED' (North-East-Down)
        (default) or 'ENU' (East-North-Up). The body's (or IMU sensor's) degrees of freedom
        will be expressed relative to this frame.
    """

    def __init__(
        self,
        fs: float,
        q0: ArrayLike = (1.0, 0.0, 0.0, 0.0),
        bg0: ArrayLike = (0.0, 0.0, 0.0),
        P0: ArrayLike = _P0,
        gyro_noise_density: float = 0.00005,
        gyro_bias_stability: float = 0.00005,
        gyro_bias_corr_time: float = 50.0,
        nav_frame: str = "NED",
    ) -> None:
        self._fs = fs
        self._dt = 1.0 / fs
        self._nav_frame = nav_frame.lower()
        self._nz2vg = _nz2vg(self._nav_frame)

        # IMU noise parameters
        self._arw = gyro_noise_density  # angular random walk
        self._gbs = gyro_bias_stability  # gyro bias stability
        self._gbc = gyro_bias_corr_time  # gyro bias correlation time

        # State and covariance estimates
        self._q_nb = np.asarray_chkfinite(q0).reshape(4).copy()
        self._bg_b = np.asarray_chkfinite(bg0).reshape(3).copy()
        self._P = np.asarray_chkfinite(P0).reshape(6, 6).copy()
        self._dx = np.zeros(6)

        # Discrete state-space model
        self._phi = _state_transition_matrix_init(self._dt, np.zeros(3), self._gbc)
        self._Q = _process_noise_covariance_matrix(
            self._dt, self._arw, self._gbs, self._gbc
        )
        self._H = _measurement_matrix_init(self._q_nb, self._nz2vg)

    def quaternion(self) -> NDArray[np.float64]:
        """
        Copy of the attitude estimate expressed as a unit quaternion.
        """
        return self._q_nb.copy()

    def euler(self, degrees: bool = False) -> NDArray[np.float64]:
        """
        Copy of the attitude estimate expressed as Euler angles (roll, pitch, yaw).

        Parameters
        ----------
        degrees : bool, optional
            Whether to return the Euler angles in degrees or radians. Defaults to
            radians.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Euler angles (roll, pitch, yaw).
        """

        theta = _euler_from_quaternion(self._q_nb)

        if degrees:
            theta = (180.0 / np.pi) * theta

        return theta

    def bias_gyro(self, degrees=False) -> NDArray[np.float64]:
        """
        Copy of the gyroscope bias estimate in rad/s or deg/s depending on the
        ``degrees`` flag.

        Parameters
        ----------
        degrees : bool, optional
            Whether to return the bias in deg/s or rad/s. Defaults to rad/s.
        """
        bg_b = self._bg_b.copy()
        if degrees:
            bg_b = (180.0 / np.pi) * bg_b
        return bg_b

    @property
    def P(self) -> NDArray[np.float64]:
        """
        Copy of the error covariance matrix estimate.
        """
        return self._P.copy()

    def update(
        self,
        dvel: ArrayLike,
        dtheta: ArrayLike,
        degrees: bool = False,
        head: float | None = None,
        head_var: float | None = None,
        head_degrees: bool = False,
        gref: bool = False,
        gref_var: ArrayLike | None = None,
    ) -> Self:
        """
        Update state estimates with IMU and aiding measurements.

        Parameters
        ----------
        dvel : array_like, shape (3,)
            Velocity increment (sculling integral) in m/s.
        dtheta : array_like, shape (3,)
            Attitude increment (coning integral) in radians or degrees depending
            on the ``degrees`` flag.
        degrees : bool, optional
            Specifies whether the unit of the attitude increment, ``dtheta``, is
            degrees or radians. Defaults to radians.
        head : float, optional
            Heading measurement in radians or degrees depending on the ``head_degrees``
            flag. I.e., the yaw angle of the 'body' frame relative to the assumed
            'navigation' frame ('NED' or 'ENU') specified during initialization.
            If ``None``, compass aiding is not used.
        head_var : float, optional
            Variance of heading measurement noise in radians^2 or degrees^2 depending
            on the ``head_degrees`` flag. Ignored if ``head`` is ``None``.
        head_degrees : bool, default False
            Specifies whether the unit of ``head`` and ``head_var`` are in degrees
            and degrees^2, or radians and radians^2. Defaults to radians and radians^2.
        gref : bool, optional
            Specifies whether to use accelerometer measurements (dvel) and the known
            direction of gravity as aiding. Defaults to ``False``.
        gref_var : array_like, shape (3,), optional
            Variance of gravity reference vector measurement noise (dimensionless).
            Required for gravity reference vector aiding.

        Returns
        -------
        VRU
            A reference to the instance itself after the update.
        """

        dvel = np.asarray(dvel)
        dtheta = np.asarray(dtheta)

        if degrees:
            dtheta = (np.pi / 180.0) * dtheta

        dtheta = dtheta - self._dt * self._bg_b

        # Update state-space model
        _state_transition_matrix_update(self._phi, dtheta)  # -> update phi (in place)

        # Project (a priori) state estimates ahead
        _update_quaternion_with_rotvec(self._q_nb, dtheta)  # -> update q_nb (in place)

        # Project (a priori) error covariance matrix estimate ahead
        _project_covariance_ahead(self._P, self._phi, self._Q)  # -> update P (in place)

        if head is not None:
            if head_var is None:
                raise ValueError("'head_var' is required for heading aiding.")

            # Update measurement matrix (heading row)
            self._H[0, 0:3] = _dhda_head(self._q_nb)

            # Update (a posteriori) estimates with heading aiding
            _aiding_update_head(  # -> update dx and P (in place)
                self._dx,
                self._P,
                self._H[0],
                self._q_nb,
                head,
                head_var,
                head_degrees,
            )

        if gref is True:
            if gref_var is None:
                raise ValueError("'gref_var' is required for gravity reference aiding.")

            # Update measurement matrix (gravity reference vector rows)
            vg_b = _gref_b_from_quat(self._q_nb, self._nz2vg)
            self._H[1:4, 0:3] = _skew_symmetric(vg_b)

            # Update (a posteriori) estimates with gravity reference vector aiding
            _aiding_update_gref(  # -> update dx and P (in place)
                self._dx,
                self._P,
                self._H[1:4],
                vg_b,
                dvel,
                np.asarray(gref_var),
            )

        # Reset state -> update q_nb, bg_b and dx (in place)
        _reset(self._dx, self._q_nb, self._bg_b)

        return self
