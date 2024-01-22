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
