from __future__ import annotations

from abc import ABC

import numpy as np
from numpy.linalg import inv
from numpy.typing import ArrayLike, NDArray

from ._transforms import (
    _angular_matrix_from_quaternion,
    _euler_from_quaternion,
    _rot_matrix_from_quaternion,
)
from ._vectorops import _normalize, _quaternion_product, _skew_symmetric


def _signed_smallest_angle(angle: float, degrees: bool = True) -> float:
    """
    Convert the given angle to the smallest angle between [-180., 180) degrees.

    Parameters
    ----------
    angle : float
        Value of angle.
    degrees : bool, default True
        Specify whether ``angle`` is given degrees or radians.

    Returns
    -------
    float
        The smallest angle between [-180., 180) degrees (or  [-pi, pi] radians).
    """
    base = 180.0 if degrees else np.pi
    return (angle + base) % (2.0 * base) - base


def gravity(lat: float | None = None, degrees: bool = True) -> float:
    """
    Calculates the gravitational acceleration based on the World Geodetic System
    (1984) Ellipsoidal Gravity Formula (WGS-84).

    The WGS-84 formula is given by::

        g = g_e * (1 - k * sin(lat)^2) / sqrt(1 - e^2 * sin(lat)^2)

    where, ::

        g_e = 9.780325335903891718546
        k = 0.00193185265245827352087
        e^2 = 0.006694379990141316996137

    and ``lat`` is the latitude.

    If no latitude is provided, the 'standard gravity', ``g_0``, is returned instead.
    The standard gravity is by definition of the ISO/IEC 8000 given as
    ``g_0 = 9.80665``.

    Parameters
    ----------
    lat : float, optional
        Latitude. If none provided, the 'standard gravity' is returned.
    degrees : bool, optional
        Specify whether the latitude, ``lat``, is in degrees or radians.
        Applicapble only if ``lat`` is provided.
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


class BaseINS(ABC):
    """
    Abstract class for inertial navigation systems (INS).

    Parameters
    ----------
    x0 : array-like, shape (16,)
        Initial state vector containing the following elements in order:

        * Position in x, y, z directions (3 elements).
        * Velocity in x, y, z directions (3 elements).
        * Attitude as unit quaternion (4 elements).
        * Accelerometer bias in x, y, z directions (3 elements).
        * Gyroscope bias in x, y, z directions (3 elements).

    Notes
    -----
    The quaternion provided as part of the initial state will be normalized to
    ensure unity.
    """

    def __init__(self, x0: ArrayLike) -> None:
        self._x0 = np.asarray_chkfinite(x0).reshape(16).copy()
        self._x0[6:10] = _normalize(self._x0[6:10])
        self._x = self._x0.copy()

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
    def _q(self) -> NDArray[np.float64]:
        return self._x[6:10]

    @_q.setter
    def _q(self, q: ArrayLike) -> None:
        self._x[6:10] = q

    @property
    def _b_acc(self) -> NDArray[np.float64]:
        return self._x[10:13]

    @_b_acc.setter
    def _b_acc(self, b_acc: ArrayLike) -> None:
        self._x[10:13] = b_acc

    @property
    def _b_gyro(self) -> NDArray[np.float64]:
        return self._x[13:16]

    @_b_gyro.setter
    def _b_gyro(self, b_gyro: ArrayLike) -> None:
        self._x[13:16] = b_gyro

    @property
    def x(self) -> NDArray[np.float64]:
        """
        Get current state vector estimate.

        Returns
        -------
        numpy.ndarray, shape (16,)
            State vector, containing the following elements in order:

            * Position in x, y, z directions (3 elements).
            * Velocity in x, y, z directions (3 elements).
            * Attitude as unit quaternion (4 elements).
            * Accelerometer bias in x, y, z directions (3 elements).
            * Gyroscope bias in x, y, z directions (3 elements).
        """
        return self._x.copy()

    def position(self) -> NDArray[np.float64]:
        """
        Get current position estimate.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Position state vector, containing position in x-, y-, and z-direction
            (in that order).
        """
        return self._p.copy()

    def velocity(self) -> NDArray[np.float64]:
        """
        Get current velocity estimate.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Velocity state vector, containing (linear) velocity in x-, y-, and z-direction
            (in that order).
        """
        return self._v.copy()

    def euler(self, degrees: bool = False) -> NDArray[np.float64]:
        """
        Get current attitude estimate as Euler angles (see Notes).

        Parameters
        ----------
        degrees : bool, default False
            Whether to return the Euler angles in degrees or radians.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Euler angles, specifically: alpha (roll), beta (pitch) and gamma (yaw)
            in that order.

        Notes
        -----
        The Euler angles describe how to transition from the 'NED' frame to the 'body'
        frame through three consecutive intrinsic and passive rotations in the ZYX order:

        #. A rotation by an angle gamma (often called yaw) about the z-axis.
        #. A subsequent rotation by an angle beta (often called pitch) about the y-axis.
        #. A final rotation by an angle alpha (often called roll) about the x-axis.

        This sequence of rotations is used to describe the orientation of the 'body' frame
        relative to the 'NED' frame in 3D space.

        Intrinsic rotations mean that the rotations are with respect to the changing
        coordinate system; as one rotation is applied, the next is about the axis of
        the newly rotated system.

        Passive rotations mean that the frame itself is rotating, not the object
        within the frame.
        """
        q = self.quaternion()
        theta = _euler_from_quaternion(q)

        if degrees:
            theta = (180.0 / np.pi) * theta

        return theta  # type: ignore[no-any-return]

    def quaternion(self) -> NDArray[np.float64]:
        """
        Get current attitude estimate as unit quaternion (from-body-to-NED).

        Returns
        -------
        numpy.ndarray, shape (4,)
            Attitude as unit quaternion. Given as ``[q1, q2, q3, q4]``, where
            ``q1`` is the real part and ``q1``, ``q2`` and ``q3`` are the three
            imaginary parts.
        """
        return self._q.copy()

    def bias_acc(self) -> NDArray[np.float64]:
        """
        Get current accelerometer bias estimate.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Accelerometer bias vector, containing biases in x-, y-, and z-direction
            (in that order).
        """
        return self._b_acc.copy()

    def bias_gyro(self, degrees: bool = False) -> NDArray[np.float64]:
        """
        Get current gyroscope bias estimate.

        Parameters
        ----------
        degrees : bool, default False
            Whether to return the bias in deg/s or rad/s.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Gyroscope bias vector, containing biases in x-, y-, and z-direction
            (in that order).
        """
        b_gyro = self._b_gyro.copy()
        if degrees:
            b_gyro = (180.0 / np.pi) * b_gyro
        return b_gyro


class StrapdownINS(BaseINS):
    """
    Strapdown inertial navigation system (INS).

    This class provides an interface for estimating position, velocity and attitude
    of a moving body by integrating the *strapdown navigation equations*.

    Parameters
    ----------
    x0 : array-like, shape (16,)
        Initial state vector containing the following elements in order:

        * Position in x, y, z directions (3 elements).
        * Velocity in x, y, z directions (3 elements).
        * Attitude as unit quaternion (4 elements).
        * Accelerometer bias in x, y, z directions (3 elements).
        * Gyroscope bias in x, y, z directions (3 elements).
    lat : float, optional
        Latitude used to calculate the gravitational acceleration. If none
        provided, the 'standard gravity' is assumed.

    Notes
    -----
    The quaternion provided as part of the initial state will be normalized to
    ensure unity.
    """

    def __init__(self, x0: ArrayLike, lat: float | None = None) -> None:
        self._g = np.array([0, 0, gravity(lat)])  # gravity vector in NED
        super().__init__(x0)

    def reset(self, x_new: ArrayLike) -> None:
        """
        Reset current state with a new one.

        Parameters
        ----------
        x_new : numpy.ndarray, shape (10,)
            New state vector, containing the following elements in order:

            * Position in x-, y-, and z-direction (3 elements).
            * Velocity in x-, y-, and z-direction (3 elements).
            * Attitude as unit quaternion (4 elements). Should be given as
              [q1, q2, q3, q4], where q1 is the real part and q1, q2 and q3
              are the three imaginary parts.

        Notes
        -----
        The quaternion provided as part of the new state will be normalized to
        ensure unity.
        """
        self._x = np.asarray_chkfinite(x_new).reshape(16).copy()
        self._x[6:10] = _normalize(self._x[6:10])

    def update(
        self,
        dt: float,
        f_imu: ArrayLike,
        w_imu: ArrayLike,
        degrees: bool = False,
    ) -> "StrapdownINS":  # TODO: Replace with ``typing.Self`` when Python > 3.11:
        """
        Update the INS states by integrating the *strapdown navigation equations*.

        Assuming constant inputs (i.e., accelerations and angular velocities) over
        the sampling period.

        The states are updated according to::

            p[k+1] = p[k] + h * v[k] + 0.5 * dt * a[k]

            v[k+1] = v[k] + dt * a[k]

            q[k+1] = q[k] + dt * T(q[k]) * w_ins[k]

        with bias compensated IMU measurements::

            f_ins[k] = f_imu[k] - b_acc[k]

            w_ins[k] = w_imu[k] - b_gyro[k]

        and::

            a[k] = R(q[k]) * f_ins[k] + g

            g = [0, 0, 9.81]^T

        Parameters
        ----------
        dt : float
            Sampling period in seconds.
        f_imu : array-like, shape (3,)
            Specific force measurements (i.e., accelerations + gravity), given
            as [f_x, f_y, f_z]^T where f_x, f_y and f_z are
            acceleration measurements in x-, y-, and z-direction, respectively.
        w_imu : array-like, shape (3,)
            Angular rate measurements, given as [w_x, w_y, w_z]^T where
            w_x, w_y and w_z are angular rates about the x-, y-,
            and z-axis, respectively.
        degrees : bool, default False
            Specify whether the angular rates are given in degrees or radians.

        Returns
        -------
        StrapdownINS :
            A reference to the instance itself after the update.
        """
        f_imu = np.asarray_chkfinite(f_imu, dtype=float).reshape(3)
        w_imu = np.asarray_chkfinite(w_imu, dtype=float).reshape(3)

        if degrees:
            w_imu = (np.pi / 180.0) * w_imu

        # Bias compensated IMU measurements
        f_ins = f_imu - self._b_acc
        w_ins = w_imu - self._b_gyro

        R_bn = _rot_matrix_from_quaternion(self._q)  # body-to-ned
        T = _angular_matrix_from_quaternion(self._q)

        # State propagation (assuming constant linear acceleration and angular velocity)
        a = R_bn @ f_ins + self._g
        self._p = self._p + dt * self._v + 0.5 * dt**2 * a
        self._v = self._v + dt * a
        self._q = self._q + dt * T @ w_ins
        self._q = _normalize(self._q)

        return self


def _gibbs(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the scaled Gibbs vector (i.e., 2 x Gibbs vector) from a unit quaternion.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Unit quaternion.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Scaled Gibbs vector.
    """
    return (2.0 / q[0]) * q[1:]  # type: ignore[no-any-return]


def _h(a: NDArray[np.float64]) -> float:
    """
    Compute yaw angle from scaled Gibbs vector, see ref [1]_.

    Parameters
    ----------
    a : numpy.ndarray, shape (3,)
        Scaled Gibbs vector.

    Returns
    -------
    float
        Yaw angle in the NED reference frame.

    References
    ----------
    .. [1] Fossen, T.I., "Handbook of Marine Craft Hydrodynamics and Motion Control",
    2nd Edition, equation 14.251, John Wiley & Sons, 2021.
    """
    a_x, a_y, a_z = a
    u_y = 2.0 * (a_x * a_y + 2.0 * a_z)
    u_x = 4.0 + a_x**2 - a_y**2 - a_z**2
    return np.arctan2(u_y, u_x)  # type: ignore[no-any-return]


def _dhda(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute yaw angle gradient wrt to the scaled Gibbs vector, see ref [1]_.

    Parameters
    ----------
    a : numpy.ndarray, shape (3,)
        Scaled Gibbs vector.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Yaw angle gradient vector.

    References
    ----------
    .. [1] Fossen, T.I., "Handbook of Marine Craft Hydrodynamics and Motion Control",
    2nd Edition, equation 14.254, John Wiley & Sons, 2021.
    """
    a_x, a_y, a_z = a

    u_y = 2.0 * (a_x * a_y + 2.0 * a_z)
    u_x = 4.0 + a_x**2 - a_y**2 - a_z**2
    u = u_y / u_x

    duda_scale = 1.0 / (4.0 + a_x**2 - a_y**2 - a_z**2) ** 2
    duda_x = -2.0 * ((a_x**2 + a_z**2 - 4.0) * a_y + a_y**3 + 4.0 * a_x * a_z)
    duda_y = 2.0 * ((a_y**2 - a_z**2 + 4.0) * a_x + a_x**3 + 4.0 * a_y * a_z)
    duda_z = 4.0 * (a_z**2 + a_x * a_y * a_z + a_x**2 - a_y**2 + 4.0)
    duda = duda_scale * np.array([duda_x, duda_y, duda_z])

    dhda = 1.0 / (1.0 + np.sum(u**2)) * duda

    return dhda  # type: ignore[no-any-return]


class AidedINS(BaseINS):
    """
    Aided inertial navigation system (AINS) using a multiplicative extended
    Kalman filter (MEKF).

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    x0 : array-like, shape (16,)
        Initial state vector containing the following elements in order:

        * Position in x, y, z directions (3 elements).
        * Velocity in x, y, z directions (3 elements).
        * Attitude as unit quaternion (4 elements).
        * Accelerometer bias in x, y, z directions (3 elements).
        * Gyroscope bias in x, y, z directions (3 elements).
    err_acc : dict of {str: float}
        Dictionary containing accelerometer noise parameters with keys:

        * ``N``: White noise power spectral density in (m/s^2)/sqrt(Hz).
        * ``B``: Bias stability in m/s^2.
        * ``tau_cb``: Bias correlation time in seconds.
    err_gyro : dict of {str: float}
        Dictionary containing gyroscope noise parameters with keys:

        * ``N``: White noise power spectral density in (rad/s)/sqrt(Hz).
        * ``B``: Bias stability in rad/s.
        * ``tau_cb``: Bias correlation time in seconds.
    var_pos : array-like, shape (3,)
        Variance of position measurement noise in m^2.
    var_vel : array-like, shape (3,)
        Variance of velocity measurement noise in (m/s)^2.
    var_g : array-like, shape (3,)
        Variance of gravitational reference vector measurement noise in m^2.
    var_compass : float
        Variance of compass measurement noise in rad^2.
    cov_error : array-like, shape (15, 15), optional, default identidy matrix
        A priori estimate of error covariance matrix, **P**.
    """

    _I15 = np.eye(15)

    def __init__(
        self,
        fs: float,
        x0: ArrayLike,
        err_acc: dict[str, float],
        err_gyro: dict[str, float],
        var_pos: ArrayLike,
        var_vel: ArrayLike,
        var_g: ArrayLike,
        var_compass: float,
        cov_error: ArrayLike | None = None,
    ) -> None:
        self._fs = fs
        self._dt = 1.0 / fs
        self._err_acc = err_acc
        self._err_gyro = err_gyro
        self._var_pos = np.asarray_chkfinite(var_pos).reshape(3).copy()
        self._var_vel = np.asarray_chkfinite(var_vel).reshape(3).copy()
        self._var_g = np.asarray_chkfinite(var_g).reshape(3).copy()
        self._var_compass = np.asarray_chkfinite(var_compass).reshape(1).copy()
        super().__init__(x0)

        # Strapdown algorithm
        self._ins = StrapdownINS(self._x0)

        # Initial Kalman filter error covariance
        if cov_error is not None:
            self._P_prior = np.asarray_chkfinite(cov_error).reshape(15, 15).copy()
        else:
            self._P_prior = np.eye(15)
        self._P = self._P_prior.copy()

        # Prepare system matrices
        q0 = self._x0[6:10]
        self._dfdx = self._prep_dfdx_matrix(err_acc, err_gyro, q0)
        self._dfdw = self._prep_dfdw_matrix(q0)
        self._dhdx = self._prep_dhdx_matrix(q0)
        self._W = self._prep_W_matrix(err_acc, err_gyro)

    @property
    def P(self) -> NDArray[np.float64]:
        """
        Get current error covariance matrix, **P**.
        """
        return self._P.copy()

    @staticmethod
    def _prep_dfdx_matrix(
        err_acc: dict[str, float],
        err_gyro: dict[str, float],
        q: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Prepare linearized state matrix
        """

        beta_acc = 1.0 / err_acc["tau_cb"]
        beta_gyro = 1.0 / err_gyro["tau_cb"]

        # Temporary placeholder vectors (to be replaced each timestep)
        f_ins = np.array([0.0, 0.0, 0.0])
        w_ins = np.array([0.0, 0.0, 0.0])

        # Aliases for transformation matrices
        R = _rot_matrix_from_quaternion  # body-to-ned rotation matrix
        S = _skew_symmetric  # skew symmetric matrix

        # State transition matrix
        dfdx = np.zeros((15, 15))
        dfdx[0:3, 3:6] = np.eye(3)
        dfdx[3:6, 6:9] = -R(q) @ S(f_ins)  # NB! update each time step
        dfdx[3:6, 9:12] = -R(q)  # NB! update each time step
        dfdx[6:9, 6:9] = -S(w_ins)  # NB! update each time step
        dfdx[6:9, 12:15] = -np.eye(3)
        dfdx[9:12, 9:12] = -beta_acc * np.eye(3)
        dfdx[12:15, 12:15] = -beta_gyro * np.eye(3)

        return dfdx

    def _update_dfdx_matrix(
        self,
        q: NDArray[np.float64],
        f_ins: NDArray[np.float64],
        w_ins: NDArray[np.float64],
    ) -> None:
        """Update linearized state transition matrix"""
        # Aliases for transformation matrices
        R = _rot_matrix_from_quaternion  # body-to-ned rotation matrix
        S = _skew_symmetric  # skew symmetric matrix

        # Update matrix
        self._dfdx[3:6, 6:9] = -R(q) @ S(f_ins)  # NB! update each time step
        self._dfdx[3:6, 9:12] = -R(q)  # NB! update each time step
        self._dfdx[6:9, 6:9] = -S(w_ins)  # NB! update each time step

    @staticmethod
    def _prep_dfdw_matrix(q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Prepare (white noise) input matrix"""

        # Alias for transformation matrix
        R = _rot_matrix_from_quaternion  # body-to-ned rotation matrix

        # Input (white noise) matrix
        dfdw = np.zeros((15, 12))
        dfdw[3:6, 0:3] = -R(q)  # NB! update each time step
        dfdw[6:9, 3:6] = -np.eye(3)
        dfdw[9:12, 6:9] = np.eye(3)
        dfdw[12:15, 9:12] = np.eye(3)

        return dfdw

    def _update_dfdw_matrix(self, q: NDArray[np.float64]) -> None:
        """Update (white noise) input matrix"""

        # Alias for transformation matrix
        R = _rot_matrix_from_quaternion  # body-to-ned rotation matrix alias

        # Update matrix
        self._dfdw[3:6, 0:3] = -R(q)

    @staticmethod
    def _prep_dhdx_matrix(q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Prepare linearized measurement matrix"""

        # Reference vector
        v01_ned = np.array([0.0, 0.0, 1.0])

        # Aliases for transformation matrices
        R = _rot_matrix_from_quaternion  # body-to-ned rotation matrix
        S = _skew_symmetric  # skew symmetric matrix

        dhdx = np.zeros((10, 15))
        dhdx[0:3, 0:3] = np.eye(3)  # position
        dhdx[3:6, 3:6] = np.eye(3)  # velocity
        dhdx[6:9, 6:9] = S(R(q).T @ v01_ned)  # gravity reference vector
        dhdx[9:10, 6:9] = _dhda(_gibbs(q))  # compass
        return dhdx

    def _update_dhdx_matrix(self, q: NDArray[np.float64]) -> None:
        """Update linearized measurement matrix"""

        # Reference vector
        v01_ned = np.array([0.0, 0.0, 1.0])

        # Aliases for transformation matrices
        R = _rot_matrix_from_quaternion  # body-to-ned rotation matrix
        S = _skew_symmetric  # skew symmetric matrix

        self._dhdx[6:9, 6:9] = S(R(q).T @ v01_ned)  # gravity reference vector
        self._dhdx[9:10, 6:9] = _dhda(_gibbs(q))  # compass

    @staticmethod
    def _prep_W_matrix(
        err_acc: dict[str, float], err_gyro: dict[str, float]
    ) -> NDArray[np.float64]:
        """Prepare white noise power spectral density matrix"""
        N_acc = err_acc["N"]
        sigma_acc = err_acc["B"]
        beta_acc = 1.0 / err_acc["tau_cb"]
        N_gyro = err_gyro["N"]
        sigma_gyro = err_gyro["B"]
        beta_gyro = 1.0 / err_gyro["tau_cb"]

        # White noise power spectral density matrix
        W = np.eye(12)
        W[0:3, 0:3] *= N_acc**2
        W[3:6, 3:6] *= N_gyro**2
        W[6:9, 6:9] *= 2.0 * sigma_acc**2 * beta_acc
        W[9:12, 9:12] *= 2.0 * sigma_gyro**2 * beta_gyro

        return W

    def update(
        self,
        f_imu: ArrayLike,
        w_imu: ArrayLike,
        pos: ArrayLike | None = None,
        vel: ArrayLike | None = None,
        head: float | None = None,
        degrees: bool = False,
        head_degrees: bool = True,
    ) -> "AidedINS":  # TODO: Replace with ``typing.Self`` when Python > 3.11
        """
        Update the AINS state estimates based on measurements.

        Parameters
        ----------
        f_imu : array-like, shape (3,)
            Specific force measurements (i.e., accelerations + gravity), given
            as [f_x, f_y, f_z]^T where f_x, f_y and f_z are
            acceleration measurements in x-, y-, and z-direction, respectively.
        w_imu : array-like, shape (3,)
            Angular rate measurements, given as [w_x, w_y, w_z]^T where
            w_x, w_y and w_z are angular rates about the x-, y-,
            and z-axis, respectively.
        pos : array-like, shape (3,), optional
            Position aiding measurement. If ``None``, position aiding is not used.
        vel : array-like, shape (3,), optional
            Velocity aiding measurement. If ``None``, velocity aiding is not used.
        head : float, optional
            Heading measurement, i.e., yaw angle. If ``None``, compass aiding is not used.
        degrees : bool, default False
            Specifies whether the unit of ``w_imu`` are in degrees or radians.
        head_degrees : bool, default True
            Specifies whether the unit of ``head`` are in degrees or radians.

        Returns
        -------
        MEKF
            A reference to the instance itself after the update.
        """
        f_imu = np.asarray_chkfinite(f_imu, dtype=float).reshape(3).copy()
        w_imu = np.asarray_chkfinite(w_imu, dtype=float).reshape(3).copy()

        if degrees:
            w_imu = (np.pi / 180.0) * w_imu

        # Update current state estimate
        self._x = self._ins.x

        # Current state estimates
        p_ins = self._p
        v_ins = self._v
        q_ins = self._q
        b_acc_ins = self._b_acc
        b_gyro_ins = self._b_gyro

        # Rotation matrix (body-to-ned)
        R_bn = _rot_matrix_from_quaternion(q_ins)

        # Bias compensated IMU measurements
        f_ins = f_imu - b_acc_ins
        w_ins = w_imu - b_gyro_ins

        # Gravity reference vector
        v01 = np.array([0.0, 0.0, 1.0])

        # Measured gravity vector
        v1 = -_normalize(f_ins)

        # Update system matrices
        self._update_dfdx_matrix(q_ins, f_ins, w_ins)
        self._update_dfdw_matrix(q_ins)
        self._update_dhdx_matrix(q_ins)

        dfdx = self._dfdx  # state matrix
        dfdw = self._dfdw  # (white noise) input matrix
        dhdx_ = self._dhdx  # measurement matrix
        W = self._W  # white noise power spectral density matrix
        P_prior = self._P_prior  # error covariance matrix
        I15 = self._I15  # 15x15 identity matrix

        # Position aiding
        dz_temp, var_z_temp, dhdx_temp = [], [], []
        if pos is not None:
            pos = np.asarray_chkfinite(pos, dtype=float).reshape(3).copy()
            delta_pos = pos - p_ins
            dz_temp.append(delta_pos)
            var_z_temp.append(self._var_pos)
            dhdx_temp.append(dhdx_[0:3])

        # Velocity aiding
        if vel is not None:
            vel = np.asarray_chkfinite(vel, dtype=float).reshape(3).copy()
            delta_vel = vel - v_ins
            dz_temp.append(delta_vel)
            var_z_temp.append(self._var_vel)
            dhdx_temp.append(dhdx_[3:6])

        # Gravity reference vector aiding
        delta_g = v1 - R_bn.T @ v01
        dz_temp.append(delta_g)
        var_z_temp.append(self._var_g)
        dhdx_temp.append(dhdx_[6:9])

        # Compass aiding
        if head is not None:
            if head_degrees:
                head = (np.pi / 180.0) * head
            delta_head = _signed_smallest_angle(head - _h(_gibbs(q_ins)), degrees=False)
            dz_temp.append(np.array([delta_head]))
            var_z_temp.append(self._var_compass)
            dhdx_temp.append(dhdx_[-1:])

        dz = np.concatenate(dz_temp, axis=0)
        dhdx = np.concatenate(dhdx_temp, axis=0)
        R = np.diag(np.concatenate(var_z_temp, axis=0))

        # Discretize system
        phi = I15 + self._dt * dfdx  # state transition matrix
        Q = self._dt * dfdw @ W @ dfdw.T  # process noise covariance matrix

        # Compute Kalman gain
        K = P_prior @ dhdx.T @ inv(dhdx @ P_prior @ dhdx.T + R)

        # Update error-state estimate with measurement
        dx = K @ dz

        # Compute error covariance for updated estimate
        P = (I15 - K @ dhdx) @ P_prior @ (I15 - K @ dhdx).T + K @ R @ K.T
        self._P = P

        # Error quaternion from 2x Gibbs vector
        da = dx[6:9]
        dq = (1.0 / np.sqrt(4.0 + da.T @ da)) * np.r_[2.0, da]

        # Reset
        p_ins = p_ins + dx[0:3]
        v_ins = v_ins + dx[3:6]
        q_ins = _quaternion_product(q_ins, dq)
        q_ins = _normalize(q_ins)
        b_acc_ins = b_acc_ins + dx[9:12]
        b_gyro_ins = b_gyro_ins + dx[12:15]
        x_ins = np.r_[p_ins, v_ins, q_ins, b_acc_ins, b_gyro_ins]
        self._ins.reset(x_ins)

        # Project ahead
        self._ins.update(self._dt, f_imu, w_imu, degrees=False)
        self._P_prior = phi @ P @ phi.T + Q

        return self
