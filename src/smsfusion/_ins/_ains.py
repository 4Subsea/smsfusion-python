from typing import Self

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from smsfusion._transforms import _euler_from_quaternion, _rot_matrix_from_quaternion
from smsfusion._vectorops import _skew_symmetric

from ._aiding import (
    _aiding_update_gref,
    _aiding_update_head,
    _aiding_update_pos,
    _aiding_update_vel,
)
from ._common import (
    _dhda_head,
    _gref_b_from_quat,
    _nz2vg,
    _project_covariance_ahead,
    _update_quaternion_with_gibbs2,
    _update_quaternion_with_rotvec,
)

_P0 = (
    (1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-6, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-6, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-6),
)


def _state_transition_matrix_init(
    dt: float,
    dvel: NDArray[np.float64],
    dtheta: NDArray[np.float64],
    R_nb: NDArray[np.float64],
    gbc: float,
) -> NDArray[np.float64]:
    """
    State transition matrix.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    dvel : ndarray, shape (3,)
        Velocity increment measurement (sculling integral).
    dtheta : ndarray, shape (3,)
        Attitude increment measurement (coning integral).
    R_nb : ndarray, shape (3, 3)
        Rotation matrix from body to navigation frame.
    gbc : float
        Gyro bias correlation time in seconds.

    Returns
    -------
    ndarray, shape (12, 12)
        State transition matrix.
    """
    phi = np.eye(12)
    phi[0:3, 3:6] += dt * np.eye(3)
    phi[3:6, 6:9] -= R_nb @ _skew_symmetric(dvel)  # NB! update each time step
    phi[6:9, 6:9] -= _skew_symmetric(dtheta)  # NB! update each time step
    phi[6:9, 9:12] -= dt * np.eye(3)
    phi[9:12, 9:12] -= dt * np.eye(3) / gbc
    return phi


@njit  # type: ignore[misc]
def _state_transition_matrix_update(
    phi: NDArray[np.float64],
    dvel: NDArray[np.float64],
    dtheta: NDArray[np.float64],
    R_nb: NDArray[np.float64],
) -> None:
    """
    Update the state transition matrix in place.

    Parameters
    ----------
    phi : ndarray, shape (12, 12)
        State transition matrix to be updated in place.
    dvel : ndarray, shape (3,)
        Velocity increment measurement (sculling integral).
    dtheta : ndarray, shape (3,)
        Attitude increment measurement (coning integral).
    R_nb : ndarray, shape (3, 3)
        Rotation matrix from body to navigation frame.
    """
    dtx, dty, dtz = dtheta
    dvx, dvy, dvz = dvel

    r00, r01, r02 = R_nb[0]
    r10, r11, r12 = R_nb[1]
    r20, r21, r22 = R_nb[2]

    # phi[6:9, 6:9] = np.eye(3) - dt * S(w_b)
    phi[6, 7] = dtz
    phi[6, 8] = -dty
    phi[7, 6] = -dtz
    phi[7, 8] = dtx
    phi[8, 6] = dty
    phi[8, 7] = -dtx

    # phi[3:6, 6:9] = -dt * R_nb @ S(f_b)
    phi[3, 6] = -dvz * r01 + dvy * r02
    phi[4, 6] = -dvz * r11 + dvy * r12
    phi[5, 6] = -dvz * r21 + dvy * r22
    phi[3, 7] = dvz * r00 - dvx * r02
    phi[4, 7] = dvz * r10 - dvx * r12
    phi[5, 7] = dvz * r20 - dvx * r22
    phi[3, 8] = -dvy * r00 + dvx * r01
    phi[4, 8] = -dvy * r10 + dvx * r11
    phi[5, 8] = -dvy * r20 + dvx * r21


def _process_noise_covariance_matrix(
    dt: float, vrw: float, arw: float, gbs: float, gbc: float
) -> NDArray[np.float64]:
    """
    Process noise covariance matrix.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    vrw : float
        Velocity random walk (accelerometer noise density) in m/s/√Hz.
    arw : float
        Angular random walk (gyroscope noise density) in rad/√Hz.
    gbs : float
        Gyro bias stability (bias instability) in rad/s.
    gbc : float
        Gyro bias correlation time in seconds.

    Returns
    -------
    ndarray, shape (12, 12)
        Process noise covariance matrix.
    """
    Q = np.zeros((12, 12))
    Q[3:6, 3:6] = dt * vrw**2 * np.eye(3)
    Q[6:9, 6:9] = dt * arw**2 * np.eye(3)
    Q[9:12, 9:12] = dt * (2.0 * gbs**2 / gbc) * np.eye(3)
    return Q


def _measurement_matrix_init(
    q_nb: NDArray[np.float64], lever_arm: NDArray[np.float64], nav_frame_factor: float
) -> NDArray[np.float64]:
    """
    Measurement matrix.

    Parameters
    ----------
    q_nb : ndarray, shape (4,)
        Unit quaternion.
    lever_arm : ndarray, shape(3,)
        Lever-arm vector describing the location of position aiding (in meters) relative
        to the IMU expressed in the IMU/body reference frame. For instance, the location
        of the GNSS antenna relative to the IMU.
    nav_frame_factor: float
        Gravity direction along the navigation frame's z-axis. +1.0 for 'NED' and
        -1.0 for 'ENU'.

    Returns
    -------
    ndarray, shape (7, 12)
        Linearized measurement matrix.
    """
    vg_b = _gref_b_from_quat(q_nb, nav_frame_factor)  # gravity reference vector
    H = np.zeros((10, 12))
    H[0:3, 0:3] = np.eye(3)  # position
    H[0:3, 6:9] = -_rot_matrix_from_quaternion(q_nb) @ _skew_symmetric(lever_arm)
    H[3:6, 3:6] = np.eye(3)  # velocity
    H[6:7, 6:9] = _dhda_head(q_nb)  # heading
    H[7:10, 6:9] = _skew_symmetric(vg_b)  # gravity reference vector
    return H


def _gravity_nav(g: float, nav_frame: str) -> NDArray[np.float64]:
    """
    Gravity vector expressed in the navigation frame ('NED' or 'ENU').

    Parameters
    ----------
    g : float
        Gravitational acceleration in m/s^2.
    nav_frame : {'NED', 'ENU'}
        Navigation frame in which the gravity vector is expressed.

    Returns
    -------
    ndarray, shape (3,)
        Gravity vector expressed in the navigation frame.
    """
    if nav_frame.lower() == "ned":
        g_n = np.array([0.0, 0.0, g])
    elif nav_frame.lower() == "enu":
        g_n = np.array([0.0, 0.0, -g])
    else:
        raise ValueError(f"Unknown navigation frame: {nav_frame}.")
    return g_n


@njit  # type: ignore[misc]
def _reset(
    dx: NDArray[np.float64],
    p_n: NDArray[np.float64],
    v_n: NDArray[np.float64],
    q_nb: NDArray[np.float64],
    bg_b: NDArray[np.float64],
) -> None:
    """
    Reset state (in place).

    Parameters
    ----------
    p_n : ndarray, shape (3,)
        Position state estimate to be reset in place.
    v_n : ndarray, shape (3,)
        Velocity state estimate to be reset in place.
    q_nb : ndarray, shape (4,)
        Attitude state estimate parameterized as a unit quaternion to be reset in place.
    bg_b : ndarray, shape (3,)
        Gyroscope bias state estimate to be reset in place.
    dx : ndarray, shape (9,)
        Error state vector containing the corrections to be applied to the state
        estimates. Will be reset to zero after applying the corrections.
    """
    p_n[:] += dx[0:3]
    v_n[:] += dx[3:6]
    _update_quaternion_with_gibbs2(q_nb, dx[6:9])  # -> update q_nb (in place)
    bg_b[:] += dx[9:12]
    dx[:] = 0.0


@njit  # type: ignore[misc]
def _project_state_ahead(
    p_n: NDArray[np.float64],
    v_n: NDArray[np.float64],
    q_nb: NDArray[np.float64],
    R_nb: NDArray[np.float64],
    dvel: NDArray[np.float64],
    dtheta: NDArray[np.float64],
    dt: float,
    dvel_g_corr: NDArray[np.float64],
) -> None:
    """
    Project state estimates ahead (in place).

    References
    ----------
    .. [1] https://www.vectornav.com/resources/inertial-navigation-primer/math-fundamentals/math-coning (Eq. 3-5)
    """
    dvel_corr = R_nb @ dvel + dvel_g_corr
    p_n[:] += dt * v_n + 0.5 * dt * dvel_corr
    v_n[:] += dvel_corr
    _update_quaternion_with_rotvec(q_nb, dtheta)  # -> update q_nb (in place)


class AINS:
    """
    Aided inertial navigation system (AINS).

    This class provides position, velocity, attitude and gyro bias estimation using
    a multiplicative extended Kalman filter (MEKF).

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    p0 : array_like, shape (3,), optional
        Initial position estimate in m. Defaults to origin (0.0, 0.0, 0.0).
    v0 : array_like, shape (3,), optional
        Initial velocity estimate in m/s. Defaults to zero velocity (stationary).
    q0 : array_like, shape (4,), optional
        Initial attitude estimate as a unit quaternion (qw, qx, qy, qz). Defaults
        to the identity quaternion (1.0, 0.0, 0.0, 0.0) (i.e., no rotation).
    bg0 : array_like, shape (3,), optional
        Initial gyroscope bias estimate (bgx, bgy, bgz) in rad/s. Defaults to zero bias.
    P0 : array_like, shape (12, 12), optional
        Initial (a priori) estimate of the error covariance matrix. Defaults to
        a small diagonal matrix (1e-6 * np.eye(12)).
    acc_noise_density : float, optional
        Accelerometer noise density (velocity random walk) in (m/s)/√Hz. Defaults to
        0.0007 (m/s)/√Hz (SMS Motion 2 noise level).
    gyro_noise_density : float, optional
        Gyroscope noise density (angular random walk) in (rad/s)/√Hz. Defaults to
        0.00005 (rad/s)/√Hz (SMS Motion 2 noise level).
    gyro_bias_stability : float, optional
        Gyroscope bias stability in rad/s. Defaults to 0.00005 rad/s (SMS Motion 2
        noise level).
    gyro_bias_corr_time : float, optional
        Gyroscope bias correlation time in seconds. Defaults to 50.0 s.
    g : float, optional
        The gravitational acceleration in m/s^2. Default is 'standard gravity' of
        9.80665 m/s^2.
    nav_frame : {'NED', 'ENU'}, optional
        Specifies the assumed inertial-like 'navigation' frame. Should be 'NED' (North-East-Down)
        (default) or 'ENU' (East-North-Up). The body's (or IMU sensor's) degrees of freedom
        will be expressed relative to this frame.
    lever_arm : array-like, shape (3,), optional
        Lever-arm vector describing the location of position aiding (in meters) relative
        to the IMU expressed in the IMU/body reference frame. For instance, the location
        of the GNSS antenna relative to the IMU. By default it is assumed that the
        aiding position coincides with the IMU's origin.
    """

    def __init__(
        self,
        fs: float,
        p0: ArrayLike = (0.0, 0.0, 0.0),
        v0: ArrayLike = (0.0, 0.0, 0.0),
        q0: ArrayLike = (1.0, 0.0, 0.0, 0.0),
        bg0: ArrayLike = (0.0, 0.0, 0.0),
        P0: ArrayLike = _P0,
        acc_noise_density: float = 0.0007,
        gyro_noise_density: float = 0.00005,
        gyro_bias_stability: float = 0.00005,
        gyro_bias_corr_time: float = 50.0,
        g: float = 9.80665,
        nav_frame: str = "NED",
        lever_arm: ArrayLike = (0.0, 0.0, 0.0),
    ) -> None:
        self._fs = fs
        self._dt = 1.0 / fs
        self._nav_frame = nav_frame.lower()
        self._nz2vg = _nz2vg(self._nav_frame)
        self._g = g
        self._g_n = _gravity_nav(self._g, self._nav_frame)
        self._dvel_g_corr = self._dt * self._g_n
        self._lever_arm = np.asarray_chkfinite(lever_arm).reshape(3).copy()

        # IMU noise parameters
        self._vrw = acc_noise_density  # velocity random walk
        self._arw = gyro_noise_density  # angular random walk
        self._gbs = gyro_bias_stability  # gyro bias stability
        self._gbc = gyro_bias_corr_time  # gyro bias correlation time

        # State and covariance estimates
        self._p_n = np.asarray_chkfinite(p0).reshape(3).copy()
        self._v_n = np.asarray_chkfinite(v0).reshape(3).copy()
        self._q_nb = np.asarray_chkfinite(q0).reshape(4).copy()
        self._bg_b = np.asarray_chkfinite(bg0).reshape(3).copy()
        self._P = np.asarray_chkfinite(P0).reshape(12, 12).copy()
        self._dx = np.zeros(12)

        # Discrete state-space model
        self._phi = _state_transition_matrix_init(
            self._dt,
            np.zeros(3),
            np.zeros(3),
            _rot_matrix_from_quaternion(self._q_nb),
            self._gbc,
        )
        self._Q = _process_noise_covariance_matrix(
            self._dt, self._vrw, self._arw, self._gbs, self._gbc
        )
        self._H = _measurement_matrix_init(self._q_nb, self._lever_arm, self._nz2vg)

    def position(self) -> NDArray[np.float64]:
        """
        Copy of the position estimate in meters.
        """
        return self._p_n.copy()

    def velocity(self) -> NDArray[np.float64]:
        """
        Copy of the velocity estimate in m/s.
        """
        return self._v_n.copy()

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
        pos: ArrayLike | None = (0.0, 0.0, 0.0),
        pos_var: ArrayLike | None = (1_000_000.0, 1_000_000.0, 1_000_000.0),
        vel: ArrayLike | None = (0.0, 0.0, 0.0),
        vel_var: ArrayLike | None = (100.0, 100.0, 100.0),
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
        pos : array-like, shape (3,), optional
            Position aiding measurement in meters. If ``None``, position aiding
            is not used. Defaults to (0.0, 0.0, 0.0) (i.e., origin).
        pos_var : array-like, shape (3,), optional
            Variance of position measurement noise in m^2. Ignored if ``pos`` is
            ``None``. Defaults to (1_000_000.0, 1_000_000.0, 1_000_000.0).
        vel : array-like, shape (3,), optional
            Velocity aiding measurement in m/s. If ``None``, velocity aiding is
            not used. Defaults to (0.0, 0.0, 0.0) (i.e., zero velocity).
        vel_var : array-like, shape (3,), optional
            Variance of velocity measurement noise in (m/s)^2. Ignored if ``vel``
            is ``None``. Defaults to (100.0, 100.0, 100.0).
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
        AINS
            A reference to the instance itself after the update.
        """

        dvel = np.asarray(dvel)
        dtheta = np.asarray(dtheta)

        if degrees:
            dtheta = (np.pi / 180.0) * dtheta

        dtheta = dtheta - self._dt * self._bg_b

        # Update state-space model
        R_nb = _rot_matrix_from_quaternion(self._q_nb)
        _state_transition_matrix_update(self._phi, dvel, dtheta, R_nb)  # -> update phi

        # Project (a priori) state estimates ahead
        _project_state_ahead(  # -> update p_n, v_n, q_nb (in place)
            self._p_n,
            self._v_n,
            self._q_nb,
            R_nb,
            dvel,
            dtheta,
            self._dt,
            self._dvel_g_corr,
        )

        # Project (a priori) error covariance matrix estimate ahead
        _project_covariance_ahead(self._P, self._phi, self._Q)  # -> update P (in place)

        if pos is not None:
            if pos_var is None:
                raise ValueError("'pos_var' is required for position aiding.")

            # Update (a posteriori) estimates with position aiding
            _aiding_update_pos(  # -> update dx and P (in place)
                self._dx,
                self._P,
                self._H[0:3],
                self._p_n,
                np.asarray(pos),
                np.asarray(pos_var),
                R_nb,
                self._lever_arm,
            )

        if vel is not None:
            if vel_var is None:
                raise ValueError("'vel_var' is required for velocity aiding.")

            # Update (a posteriori) estimates with velocity aiding
            _aiding_update_vel(  # -> update dx and P (in place)
                self._dx,
                self._P,
                self._H[3:6],
                self._v_n,
                np.asarray(vel),
                np.asarray(vel_var),
            )

        if head is not None:
            if head_var is None:
                raise ValueError("'head_var' is required for heading aiding.")

            # Update measurement matrix (heading row)
            self._H[6, 6:9] = _dhda_head(self._q_nb)

            # Update (a posteriori) estimates with heading aiding
            _aiding_update_head(  # -> update dx and P (in place)
                self._dx,
                self._P,
                self._H[6],
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
            self._H[7:10, 6:9] = _skew_symmetric(vg_b)

            # Update (a posteriori) estimates with gravity reference vector aiding
            _aiding_update_gref(  # -> update dx and P (in place)
                self._dx,
                self._P,
                self._H[7:10],
                vg_b,
                dvel,
                np.asarray(gref_var),
            )

        # Reset state -> update p_n, v_n, q_nb, bg_b and dx (in place)
        _reset(self._dx, self._p_n, self._v_n, self._q_nb, self._bg_b)

        return self
