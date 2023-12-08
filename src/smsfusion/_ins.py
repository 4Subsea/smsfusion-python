from __future__ import annotations

import numpy as np
from numpy.linalg import inv
from numpy.typing import ArrayLike, NDArray

from ._ahrs import AHRS
from ._transforms import (
    _angular_matrix_from_euler,
    _angular_matrix_from_quaternion,
    _euler_from_quaternion,
    _quaternion_from_euler,
    _rot_matrix_from_euler,
    _rot_matrix_from_quaternion,
)
from ._vectorops import _normalize


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


class StrapdownINS:
    """
    Strapdown inertial navigation system (INS).

    This class provides an interface for estimating position, velocity and attitude
    of a moving body by integrating the *strapdown navigation equations*.

    Parameters
    ----------
    x0 : numpy.ndarray, shape (10,)
        Initial state vector, containing the following elements in order:

        * Position in x-, y-, and z-direction (3 elements).
        * Velocity in x-, y-, and z-direction (3 elements).
        * Attitude as unit quaternion (4 elements). Should be given as
          [q1, q2, q3, q4], where q1 is the real part and q1, q2 and q3 are
          the three imaginary parts.
    lat : float, optional
        Latitude used to calculate the gravitational acceleration. If none
        provided, the 'standard gravity' is assumed.

    Notes
    -----
    The quaternion provided as part of the initial state will be normalized to
    ensure unity.
    """

    def __init__(self, x0: ArrayLike, lat: float | None = None) -> None:
        self._x0 = np.asarray_chkfinite(x0).reshape(10, 1).copy()
        self._x = self._x0.copy()
        self._x[6:] = _normalize(self._x[6:])
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
    def _q(self) -> NDArray[np.float64]:
        return self._x[6:10]

    @_q.setter
    def _q(self, q: ArrayLike) -> None:
        self._x[6:10] = q

    @property
    def x(self) -> NDArray[np.float64]:
        """
        Get current state vector estimate.

        Returns
        -------
        numpy.ndarray, shape (10,)
            State vector, containing the following elements in order:

            * Position in x-, y-, and z-direction (3 elements).
            * Velocity in x-, y-, and z-direction (3 elements).
            * Attitude as unit quaternion (4 elements). Should be given as
              [q1, q2, q3, q4], where q1 is the real part and q1, q2 and q3
              are the three imaginary parts.
        """
        return self._x.flatten()

    def position(self) -> NDArray[np.float64]:
        """
        Get current position estimate.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Position state vector, containing position in x-, y-, and z-direction
            (in that order).
        """
        return self._p.flatten()

    def velocity(self) -> NDArray[np.float64]:
        """
        Get current velocity estimate.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Velocity state vector, containing (linear) velocity in x-, y-, and z-direction
            (in that order).
        """
        return self._v.flatten()

    def quaternion(self) -> NDArray[np.float64]:
        """
        Get current attitude estimate as unit quaternion (from-body-to-NED).

        Returns
        -------
        numpy.ndarray, shape (4,)
            Attitude as unit quaternion. Given as [q1, q2, q3, q4], where
            q1 is the real part and q1, q2 and q3 are the three
            imaginary parts.
        """
        return self._q.flatten()

    def euler(self, degrees: bool = False) -> NDArray[np.float64]:
        """
        Get current attitude estimate as Euler angles (see Notes).

        Parameters
        ----------
        degrees : bool, default False
            Specify whether to return the Euler angles in degrees or radians.

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
        self._x = np.asarray_chkfinite(x_new).reshape(10, 1).copy()
        self._x[6:] = _normalize(self._x[6:])

    def update(
        self,
        dt: float,
        f_ins: ArrayLike,
        w_ins: ArrayLike,
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

        where,::

            a[k] = R(q[k]) * f_ins[k] + g

            g = [0, 0, 9.81]^T

        Parameters
        ----------
        dt : float
            Sampling period in seconds.
        f_imu : array_like, shape (3,)
            Specific force (bias compansated) measurements (i.e., accelerations
            + gravity), given as [f_x, f_y, f_z]^T where f_x, f_y and f_z are
            acceleration measurements in x-, y-, and z-direction, respectively.
        w_imu : array_like, shape (3,)
            Angular rate (bias compansated) measurements, given as
            [w_x, w_y, w_z]^T where w_x, w_y and w_z are angular rates about
            the x-, y-, and z-axis, respectively.
        degrees : bool, default False
            Specify whether the angular rates are given in degrees or radians.

        Returns
        -------
        AidedINS
            A reference to the instance itself after the update.
        """
        f_ins = np.asarray_chkfinite(f_ins, dtype=float).reshape(3, 1)
        w_ins = np.asarray_chkfinite(w_ins, dtype=float).reshape(3, 1)

        R_bn = _rot_matrix_from_quaternion(self.quaternion())  # body-to-ned
        T = _angular_matrix_from_quaternion(self.quaternion())

        if degrees:
            w_ins = (np.pi / 180.0) * w_ins

        # State propagation (assuming constant linear acceleration and angular velocity)
        a = R_bn @ f_ins + self._g
        self._p = self._p + dt * self._v + 0.5 * dt**2 * a
        self._v = self._v + dt * a
        self._q = self._q + dt * T @ w_ins
        self._q = _normalize(self._q)

        return self


class _LegacyStrapdownINS:
    """
    Strapdown inertial navigation system (INS).

    This class provides an interface for estimating position, velocity and attitude
    of a moving body by integrating the *strapdown navigation equations*.

    Parameters
    ----------
    x0 : numpy.ndarray, shape (9,)
        Initial state vector, containing the following elements in order:

        * Position in x-, y-, and z-direction (3 elements).
        * Velocity in x-, y-, and z-direction (3 elements).
        * Euler angles in radians (3 elements), specifically: alpha (roll),
          beta (pitch) and gamma (yaw) in that order.
    lat : float, optional
        Latitude used to calculate the gravitational acceleration. If none
        provided, the 'standard gravity' is assumed.
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
        Get current state vector estimate.

        Returns
        -------
        numpy.ndarray, shape (9,)
            State vector, containing the following elements in order:

            * Position in x-, y-, and z-direction (3 elements).
            * Velocity in x-, y-, and z-direction (3 elements).
            * Euler angles in radians (3 elements), specifically: alpha (roll),
              beta (pitch) and gamma (yaw) in that order.
        """
        return self._x.flatten()

    def position(self) -> NDArray[np.float64]:
        """
        Get current position estimate.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Position state vector, containing position in x-, y-, and z-direction
            (in that order).
        """
        return self._p.flatten()

    def velocity(self) -> NDArray[np.float64]:
        """
        Get current velocity estimate.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Velocity state vector, containing (linear) velocity in x-, y-, and z-direction
            (in that order).
        """
        return self._v.flatten()

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
        theta = self._theta.flatten()

        if degrees:
            theta = (180.0 / np.pi) * theta

        return theta

    def quaternion(self) -> NDArray[np.float64]:
        """
        Get current attitude estimate as unit quaternion (from-body-to-NED).

        Returns
        -------
        numpy.ndarray, shape (4,)
            Attitude as unit quaternion. Given as [q1, q2, q3, q4], where
            q1 is the real part and q1, q2 and q3 are the three
            imaginary parts.
        """
        return _quaternion_from_euler(self._theta.flatten())  # type: ignore[no-any-return]

    def reset(self, x_new: ArrayLike) -> None:
        """
        Reset current state with a new one.

        Parameters
        ----------
        x_new : numpy.ndarray, shape (10,)
            New state vector, containing the following elements in order:

            * Position in x-, y-, and z-direction (3 elements).
            * Velocity in x-, y-, and z-direction (3 elements).
            * Euler angles in radians (3 elements), specifically: alpha (roll),
              beta (pitch) and gamma (yaw) in that order.
        """
        self._x = np.asarray_chkfinite(x_new).reshape(9, 1).copy()

    def update(
        self,
        dt: float,
        f_imu: ArrayLike,
        w_imu: ArrayLike,
        degrees: bool = False,
        theta_ext: ArrayLike | None = None,
    ) -> (
        "_LegacyStrapdownINS"
    ):  # TODO: Replace with ``typing.Self`` when Python > 3.11:
        """
        Update the INS states by integrating the *strapdown navigation equations*.

        Assuming constant inputs (i.e., accelerations and angular velocities) over
        the sampling period.

        The states are updated according to::

            p[k+1] = p[k] + h * v[k] + 0.5 * dt * a[k]

            v[k+1] = v[k] + dt * a[k]

            theta[k+1] = theta[k] + dt * T[k] * w[k]

        where,::

            a[k] = R(q[k]) * f_ins[k] + g

            g = [0, 0, 9.81]^T

        Parameters
        ----------
        dt : float
            Sampling period in seconds.
        f_imu : array_like, shape (3,)
            Specific force (bias compansated) measurements (i.e., accelerations
            + gravity), given as [f_x, f_y, f_z]^T where f_x, f_y and f_z are
            acceleration measurements in x-, y-, and z-direction, respectively.
        w_imu : array_like, shape (3,)
            Angular rate (bias compansated) measurements, given as
            [w_x, w_y, w_z]^T where w_x, w_y and w_z are angular rates about
            the x-, y-, and z-axis, respectively.
        degrees : bool, default False
            Whether the angular rates are given in degrees or radians.
        theta_ext : array_like, shape (3,) optional
            Externally provided IMU orientation as Euler angles (in radians)
            according to the ned-to-body `z-y-x` convention, which is used to
            calculate the R and T matrices. If ``None``, the most recent
            orientation state is used instead.

        Returns
        -------
        AidedINS
            A reference to the instance itself after the update.
        """
        theta = (
            self._theta.flatten()
            if theta_ext is None
            else np.asarray_chkfinite(theta_ext, dtype=float)
        )

        R_bn = _rot_matrix_from_euler(theta)  # body-to-ned rotation matrix
        T = _angular_matrix_from_euler(theta)

        f_imu = np.asarray_chkfinite(f_imu, dtype=float).reshape(3, 1)
        w_imu = np.asarray_chkfinite(w_imu, dtype=float).reshape(3, 1)

        if degrees:
            w_imu = (np.pi / 180.0) * w_imu

        # State propagation (assuming constant linear acceleration and angular velocity)
        a = R_bn @ f_imu + self._g
        self._p = self._p + dt * self._v + 0.5 * dt**2 * a
        self._v = self._v + dt * a
        self._theta = self._theta + dt * T @ w_imu

        return self


class AidedINS:
    """
    Aided inertial navigation system (AINS).

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    x0 : array-like, shape (15,)
        Initial state vector containing the following elements in order:

        * Position in x, y, z directions (3 elements).
        * Velocity in x, y, z directions (3 elements).
        * Euler angles in radians (3 elements), specifically: alpha (roll),
          beta (pitch) and gamma (yaw) in that order.
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
        Variance of position measurement noise in m^2
    var_ahrs : array-like, shape (3,)
        Variance of attitude measurements in rad^2. Specifically, it refers to
        the variance of the AHRS error.
    ahrs : AHRS
        A configured instance of :class:`AHRS`.

    Attributes
    ----------
    ahrs : AHRS
        A configured instance of :class:`AHRS`.

    Notes
    -----
    This AINS model has the following limitations:

    * Supports only position and AHRS aiding.
    * Operates at a constant sampling rate.
    * Initial error covariance matrix, P, is fixed to the identity matrix, and
      cannot be set during initialization.
    * Update method is not properly tested.
    * Does not correct for sensor installation offsets.
    * Estimates the system states at the 'sensor location'.
    * IMU error models are the same for all axes.
    """

    _I15 = np.eye(15)

    def __init__(
        self,
        fs: float,
        x0: ArrayLike,
        err_acc: dict[str, float],
        err_gyro: dict[str, float],
        var_pos: ArrayLike,
        var_ahrs: ArrayLike,
        ahrs: AHRS,
    ) -> None:
        self._fs = fs
        self._dt = 1.0 / fs
        self._err_acc = err_acc
        self._err_gyro = err_gyro
        self._x0 = np.asarray_chkfinite(x0).reshape(15, 1).copy()
        var_pos = np.asarray_chkfinite(var_pos).reshape(3).copy()
        var_ahrs = np.asarray_chkfinite(var_ahrs).reshape(3).copy()

        # Attitude Heading Reference System (AHRS)
        if not isinstance(ahrs, AHRS):
            raise TypeError("`ahrs` must be an instance of `AHRS`")
        if ahrs._fs != fs:
            raise ValueError("`AidedINS` and `AHRS` sampling frequencies must be equal")
        self.ahrs = ahrs

        # Strapdown algorithm
        self._x_ins = self._x0
        self._ins = _LegacyStrapdownINS(self._x_ins[0:9])

        # Initial Kalman filter error covariance
        self._P_prior = np.eye(15)

        # Prepare system matrices
        self._F = self._prep_F_matrix(err_acc, err_gyro, self._theta.flatten())
        self._G = self._prep_G_matrix(self._theta.flatten())
        self._W = self._prep_W_matrix(err_acc, err_gyro)
        self._H = self._prep_H_matrix()
        self._R = np.diag(np.r_[var_pos, var_ahrs])

    @property
    def _x(self) -> NDArray[np.float64]:
        """Full state (i.e., INS state + error state)"""
        return self._x_ins  # error state is zero due to reset

    @property
    def _p(self) -> NDArray[np.float64]:
        return self._x[0:3]

    @property
    def _v(self) -> NDArray[np.float64]:
        return self._x[3:6]

    @property
    def _theta(self) -> NDArray[np.float64]:
        return self._x[6:9]

    @property
    def x(self) -> NDArray[np.float64]:
        """
        Get current state vector estimate.

        Returns
        -------
        numpy.ndarray, shape (15,)
            State vector, containing the following elements in order:

            * Position in x-, y-, and z-direction (3 elements).
            * Velocity in x-, y-, and z-direction (3 elements).
            * Euler angles in radians (3 elements), specifically: alpha (roll),
              beta (pitch) and gamma (yaw) in that order.
            * Accelerometer bias in x, y, z directions (3 elements).
            * Gyroscope bias in x, y, z directions (3 elements).
        """
        return self._x.flatten()

    def position(self) -> NDArray[np.float64]:
        """
        Get current position estimate.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Position state vector, containing position in x-, y-, and z-direction
            (in that order).
        """
        return self._p.flatten()

    def velocity(self) -> NDArray[np.float64]:
        """
        Get current velocity estimate.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Velocity state vector, containing (linear) velocity in x-, y-, and z-direction
            (in that order).
        """
        return self._v.flatten()

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
        theta = self._theta.flatten()

        if degrees:
            theta = (180.0 / np.pi) * theta

        return theta

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
        return _quaternion_from_euler(self._theta.flatten())  # type: ignore[no-any-return]

    @staticmethod
    def _prep_F_matrix(
        err_acc: dict[str, float],
        err_gyro: dict[str, float],
        theta_rad: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Prepare F (state) matrix.
        """

        beta_acc = 1.0 / err_acc["tau_cb"]
        beta_gyro = 1.0 / err_gyro["tau_cb"]

        R_bn = _rot_matrix_from_euler(theta_rad)
        T = _angular_matrix_from_euler(theta_rad)

        # State matrix
        F = np.zeros((15, 15))
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 9:12] = -R_bn  # NB! update each time step
        F[6:9, 12:15] = -T  # NB! update each time step
        F[9:12, 9:12] = -beta_acc * np.eye(3)
        F[12:15, 12:15] = -beta_gyro * np.eye(3)

        return F

    @staticmethod
    def _prep_G_matrix(theta_rad: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Prepare G (white noise) input matrix.
        """

        R_bn = _rot_matrix_from_euler(theta_rad)
        T = _angular_matrix_from_euler(theta_rad)

        # Input (white noise) matrix
        G = np.zeros((15, 12))
        G[3:6, 0:3] = -R_bn  # NB! update each time step
        G[6:9, 3:6] = -T  # NB! update each time step
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)

        return G

    @staticmethod
    def _prep_W_matrix(
        err_acc: dict[str, float], err_gyro: dict[str, float]
    ) -> NDArray[np.float64]:
        """
        Prepare W (white noise power spectral density) matrix.
        """
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

    @staticmethod
    def _prep_H_matrix() -> NDArray[np.float64]:
        """
        Prepare H (measurement) matrix.
        """
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)  # position
        H[3:6, 6:9] = np.eye(3)  # attitude
        return H

    def _update_system_matrices(
        self, R_bn: NDArray[np.float64], T: NDArray[np.float64]
    ) -> None:
        """
        Update F and G (system) matrices with new rotation and angular transformation matrices.
        """
        self._F[3:6, 9:12] = -R_bn
        self._F[6:9, 12:15] = -T
        self._G[3:6, 0:3] = -R_bn
        self._G[6:9, 3:6] = -T

    def update(
        self,
        f_imu: ArrayLike,
        w_imu: ArrayLike,
        head: float,
        pos: ArrayLike | None = None,
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
        head : float
            Heading measurement, i.e., yaw angle.
        pos : array-like (3,), default=None
            Position aiding measurement. If `None`, no position aiding is used.
        degrees : bool, default=False
            Specifies whether the unit of the ``w_imu`` are in degrees or radians.
        head_degrees : bool, default=True
            Specifies whether the unit ``head`` are in degrees or radians.

        Returns
        -------
        AidedINS
            A reference to the instance itself after the update.
        """
        f_imu = np.asarray_chkfinite(f_imu, dtype=float).reshape(3, 1).copy()
        w_imu = np.asarray_chkfinite(w_imu, dtype=float).reshape(3, 1).copy()
        theta_ext = self.ahrs.euler(degrees=False)

        if pos is not None:
            pos = np.asarray_chkfinite(pos, dtype=float).reshape(3, 1).copy()
            z = np.vstack([pos, theta_ext.reshape(3, 1)])  # measurement vector
            H = self._H  # measurement matrix
            R = self._R  # measurement noise covariance matrix
        else:
            z = theta_ext.reshape(3, 1)  # measurement vector
            H = self._H[3:6, :]  # measurement matrix
            R = self._R[3:6, 3:6]  # measurement noise covariance matrix

        if degrees:
            w_imu = np.radians(w_imu)
        if head_degrees:
            head = np.radians(head)

        # Update INS state
        self._x_ins[0:9] = self._ins.x.reshape(9, 1)

        # Setup transformation matrices based on AHRS 'measurement'
        R_bn = _rot_matrix_from_euler(theta_ext)
        T = _angular_matrix_from_euler(theta_ext)  # rotation rates to Euler rates

        # Update system matrices with AHRS attitude 'measurements'
        self._update_system_matrices(R_bn, T)

        F = self._F  # state matrix
        G = self._G  # (white noise) input matrix
        W = self._W  # white noise power spectral density matrix
        P_prior = self._P_prior  # error covariance matrix
        I15 = self._I15  # 15x15 identity matrix

        # Discretize system
        phi = I15 + self._dt * F  # state transition matrix
        Q = self._dt * G @ W @ G.T  # process noise covariance matrix

        # Compute Kalman gain
        K = P_prior @ H.T @ inv(H @ P_prior @ H.T + R)

        # Update error-state estimate with measurement
        dz = z - H @ self._x_ins
        dx = K @ dz

        # Compute error covariance for updated estimate
        P = (I15 - K @ H) @ P_prior @ (I15 - K @ H).T + K @ R @ K.T

        # Reset
        self._x_ins = self._x_ins + dx
        self._ins.reset(self._x_ins[0:9])

        # Project ahead
        f_ins = f_imu - self._x_ins[9:12]
        w_ins = w_imu - self._x_ins[12:15]
        self._ins.update(self._dt, f_ins, w_ins, theta_ext=theta_ext, degrees=False)
        self.ahrs.update(f_imu, w_imu, head, degrees=False, head_degrees=False)
        self._P_prior = phi @ P @ phi.T + Q

        return self
