import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._transforms import _angular_matrix_from_euler, _rot_matrix_from_euler


def gravity(lat: float | None = None, degrees: bool = True) -> float:
    """
    Calculates the gravitational acceleration based on the World Geodetic System
    (1984) Ellipsoidal Gravity Formula (WGS-84).

    The WGS-84 formula is given by:

        ``g = g_e * (1 - k * sin(lat)^2) / sqrt(1 - e^2 * sin(lat)^2)``

    where,

        ``g_e = 9.780325335903891718546``

        ``k = 0.00193185265245827352087``

        ``e^2 = 0.006694379990141316996137``

    and ``lat`` is the latitude.

    If no latitude is provided, the 'standard gravity', ``g_0``, is returned instead.
    The standard gravity is by definition of the ISO/IEC 8000 given by:

        ``g_0 = 9.80665``

    Parameters
    ----------
    lat : float (optional)
        Latitude. If `lat` is ``None``, the 'standard gravity' is returned.
    degrees : True
        Whether the latitude, `lat`, is given in degrees (``True``) or radians (``False``).
    """
    if lat is None:
        g_0 = 9.80665  # standard gravity in m/s^2
        return g_0

    g_e = 9.780325335903891718546  # gravity at equator
    k = 0.00193185265245827352087  # formula constant
    e_2 = 0.006694379990141316996137  # spheroid's squared eccentricity

    if degrees:
        lat = (np.pi / 180.0) * lat

    g = g_e * (1.0 + k * np.sin(lat) ** 2.0) / np.sqrt(1.0 - e_2 * np.sin(lat) ** 2.0)
    return g  # type: ignore[no-any-return]  # numpy funcs declare Any as return when given scalar-like


class StrapdownINS:
    """
    Inertial navigation system (INS) strapdown algorithm.

    This class provides an interface for estimating position, velocity and attitude
    of a moving body by integrating the 'strapdown navigation equations'.

    Parameters
    ----------
    x0 = array-like (9,)
        Initial state vector as 1-D array of length 9 (see Notes).
    lat : float (optional)
        Latitude used to calculate the gravitational acceleration. If `lat` is ``None``,
        the 'standard gravity' (i.e., 9.80665) is used.

    Notes
    -----
    The state vector should be given as:

        ``x = [p_x, p_y, p_z, v_x, v_y, v_z, alpha, beta, gamma]^T``

    where ``p_x``, ``p_y`` and ``p_z`` are position coordinates (in x-, y- and z-direction),
    ``v_x``, ``v_y`` and ``v_z`` are (linear) velocities (in x-, y- and z-direction),
    and ``alpha``, ``beta`` and ``gamma`` are Euler angles (given in radians).
    """

    def __init__(self, x0: ArrayLike, lat: float | None = None) -> None:
        self._x0 = np.asarray_chkfinite(x0).reshape(9, 1).copy()
        self._x = self._x0.copy()
        self._g = np.array([0, 0, gravity(lat)]).reshape(3, 1)  # gravity vector in NED

    @property
    def _p(self) -> NDArray[np.float64]:
        return self._x[0:3]

    @_p.setter
    def _p(self, p: ArrayLike) -> None:
        self._x[0:3] = p

    @property
    def _v(self) -> NDArray[np.float64]:
        return self._x[3:6]

    @_v.setter
    def _v(self, v: ArrayLike) -> None:
        self._x[3:6] = v

    @property
    def _theta(self) -> NDArray[np.float64]:
        return self._x[6:9]

    @_theta.setter
    def _theta(self, theta: ArrayLike) -> None:
        self._x[6:9] = theta

    @property
    def x(self) -> NDArray[np.float64]:
        """
        Current state vector estimate.

        Given as as:

            ``x = [p_x, p_y, p_z, v_x, v_y, v_z, alpha, beta, gamma]^T``

        where ``p_x``, ``p_y`` and ``p_z`` are position coordinates
        (in x-, y- and z-direction), ``v_x``, ``v_y`` and ``v_z`` are (linear) velocities
        (in x-, y- and z-direction), and ``alpha``, ``beta`` and ``gamma`` are Euler angles
        (given in radians).

        Returns
        -------
        x : ndarray
            State as array of shape (9, 1).
        """
        return self._x.copy()

    def position(self) -> NDArray[np.float64]:
        """
        Current position vector estimate.

        Given as as:

            ``p = [p_x, p_y, p_z]^T``

        where ``p_x``, ``p_y`` and ``p_z`` are position coordinates in x-, y-, and
        z-direction respectively.

        Returns
        -------
        p : ndarray
            Position as array of shape (3, 1).
        """
        return self._p.copy()

    def velocity(self) -> NDArray[np.float64]:
        """
        Current velocity vector estimate.

        Given as as:

            ``v = [v_x, v_y, v_z]^T``

        where ``v_x``, ``v_y`` and ``v_z`` are (linear) velocity components in x-, y-,
        and z-direction respectively.

        Returns
        -------
        v : ndarray
            Velocity as array of shape (3, 1).
        """
        return self._v.copy()

    def attitude(self, degrees: bool = False) -> NDArray[np.float64]:
        """
        Current attitude estimate as vector of Euler angles (i.e., roll, pitch and yaw).

        Given as as:

            ``theta = [alpha, beta, gamma]^T``

        where ``alpha``, ``beta`` and ``gamma`` are the Euler angles (given in radians).

        Returns
        -------
        theta : ndarray
            Attitude as array of shape (3, 1).
        """
        theta = self._theta.copy()

        if degrees:
            theta = (180.0 / np.pi) * theta

        return theta

    def reset(self, x_new: ArrayLike) -> None:
        """
        Reset state.

        Parameters
        ----------
        x_new : array-like (9,)
            New state as 1-D array of length 9 (see Notes).

        Notes
        -----
        The state vector should be given as:

            ``x = [p_x, p_y, p_z, v_x, v_y, v_z, alpha, beta, gamma]^T``

        where ``p_x``, ``p_y`` and ``p_z`` are position coordinates
        (in x-, y- and z-direction), ``v_x``, ``v_y`` and ``v_z`` are (linear) velocities
        (in x-, y- and z-direction), and ``alpha``, ``beta`` and ``gamma`` are Euler angles
        (given in radians).
        """
        self._x = np.asarray_chkfinite(x_new).reshape(9, 1).copy()

    def update(
        self,
        dt: float,
        f_imu: ArrayLike,
        w_imu: ArrayLike,
        degrees: bool = False,
        theta_ext: ArrayLike | None = None,
    ) -> None:
        """
        Update the INS states by integrating the 'strapdown navigation equations'.

        Assuming constant inputs (i.e., accelerations and angular velocity) over
        the sampling period.

        The states are updated according to:

            ``p[k+1] = p[k] + h * v[k] + 0.5 * dt * a[k]``

            ``v[k+1] = v[k] + dt * a[k]``

            ``theta[k+1] = theta[k] + dt * T[k] * w[k]``

        where,

            ``a[k] = R[k] * f_imu[k] + g``

            ``g = [0, 0, 9.81]^T``

        Parameters
        ----------
        dt : float
            Sampling period in seconds.
        f_imu : array-like (3,)
            IMU specific force measurements (i.e., accelerations + gravity). Given as
            ``[f_x, f_y, f_z]^T`` where ``f_x``, ``f_y`` and ``f_z`` are
            acceleration measurements in x-, y-, and z-direction, respectively.
        w_imu : array-like (3,)
            IMU rotation rate measurements. Given as ``[w_x, w_y, w_z]^T`` where
            ``w_x``, ``w_y`` and ``w_z`` are rotation rates about the x-, y-,
            and z-axis, respectively. Unit determined with ``degrees`` keyword argument.
        degrees : bool
            Whether the rotation rates are given in `degrees` (``True``) or `radians`
            (``False``).
        theta_ext : array-like (3,), optional
            Externally provided IMU orientation as Euler angles according to the
            ned-to-body `z-y-x` convention, which is used to calculate the
            ``R`` and ``T`` matrices. If ``None`` (default), the most recent orientation state
            is used instead. Unit must be in `radians`.
        """
        theta = (
            self._theta.flatten()
            if theta_ext is None
            else np.asarray_chkfinite(theta_ext, dtype=float)
        )

        R = _rot_matrix_from_euler(theta).T
        T = _angular_matrix_from_euler(theta)

        f_imu = np.asarray_chkfinite(f_imu, dtype=float).reshape(3, 1)
        w_imu = np.asarray_chkfinite(w_imu, dtype=float).reshape(3, 1)

        if degrees:
            w_imu = (np.pi / 180.0) * w_imu

        # State propagation (assuming constant linear acceleration and angular velocity)
        a = R @ f_imu + self._g
        self._p = self._p + dt * self._v + 0.5 * dt**2 * a
        self._v = self._v + dt * a
        self._theta = self._theta + dt * T @ w_imu
