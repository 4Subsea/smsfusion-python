from __future__ import annotations

import numpy as np
from numba import njit
from numpy.linalg import inv
from numpy.typing import ArrayLike, NDArray

from ._transforms import (
    _angular_matrix_from_quaternion,
    _euler_from_quaternion,
    _rot_matrix_from_quaternion,
)
from ._vectorops import _normalize, _quaternion_product, _skew_symmetric


class FixedNED:
    """
    Convert position coordinates between a fixed NED frame (x, y, z) and ECEF frame
    (lattitude, longitude, height).

    The fixed NED frame is a tangential plane on the WGS-84 ellipsoid with its origin
    fixed at the provided reference coordinates. It is assumed that the tangential
    plane is close to the ellipsoid surface.

    Parameters
    ----------
    lat_ref: float
        Reference latitude coordinate in decimal degrees.
    lon_ref: float
        Reference longitude coordinate in decimal degrees.
    height_ref: ref
        Reference height coordinate in decimal degrees.
    """

    def __init__(self, lat_ref: float, lon_ref: float, height_ref: float) -> None:
        self._lat_ref = lat_ref
        self._lon_ref = lon_ref
        self._height_ref = height_ref

        radius_eq = 6_378_137  # equatorial radius (WGS-84)
        radius_polar = 6_356_752.314245  # polar radius (WGS-84)

        radius_ratio_squared = (radius_polar / radius_eq) ** 2
        denom = np.cos(
            self._lat_ref * (np.pi / 180.0)
        ) ** 2 + radius_ratio_squared * np.sin(self._lat_ref * (np.pi / 180.0))

        self._Rn = radius_eq / np.sqrt(denom)  # radius prime vertical
        self._Rm = self._Rn * radius_ratio_squared / denom  # radius meridian

        self._Rm_h = self._Rm + height_ref
        self._Rn_h_cos = (self._Rn + height_ref) * np.cos(
            self._lat_ref * (np.pi / 180.0)
        )

    def to_llh(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """
        Compute longitude, latitude, and height coordinates (WGS-84) from local
        Cartesian coordinates in the fixed NED frame.

        Parameters
        ----------
        x: float
            Local x-coordinate in meters in the fixed NED frame.
        y: float
            Local y-coordinate in meters in the fixed NED frame.
        z: float
            Local z-coordinate in meters in the fixed NED frame.

        Returns
        -------
        lat: float
            Latitude coordinate in decimal degrees.
        lon: float
            Longitude coordinate in decimal degrees.
        height: float
            Height coordinate in meters.
        """
        dlat = (180.0 / np.pi) * x / self._Rm_h
        dlon = (180.0 / np.pi) * y / self._Rn_h_cos

        lat = _signed_smallest_angle(self._lat_ref + dlat, degrees=True)
        lon = _signed_smallest_angle(self._lon_ref + dlon, degrees=True)
        h = self._height_ref - z
        return lat, lon, h

    def to_xyz(
        self, lat: float, lon: float, height: float
    ) -> tuple[float, float, float]:
        """
        Compute local Cartesian coordinates in the fixed NED frame from longitude,
        latitude, and height coordinates (WGS-84).

        Parameters
        ----------
        lat: float
            Latitude coordinate in decimal degrees.
        lon: float
            Longitude coordinate in decimal degrees.
        height: float
            Height coordinate in meters.

        Returns
        -------
        x: float
            Local x-coordinate in meters in the fixed NED frame.
        y: float
            Local y-coordinate in meters in the fixed NED frame.
        z: float
            Local z-coordinate in meters in the fixed NED frame.
        """
        dlat = lat - self._lat_ref
        dlon = lon - self._lon_ref

        x = (np.pi / 180.0) * dlat * self._Rm_h
        y = (np.pi / 180.0) * dlon * self._Rn_h_cos
        z = self._height_ref - height
        return x, y, z


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


class INSMixin:
    """
    Mixin class for inertial navigation systems (INS).

    Requires that the inheriting class has an `_x` attribute which is a 1D numpy array
    of length 16 containing the following elements in order:

        * Position in x, y, z directions (3 elements).
        * Velocity in x, y, z directions (3 elements).
        * Attitude as unit quaternion (4 elements).
        * Accelerometer bias in x, y, z directions (3 elements).
        * Gyroscope bias in x, y, z directions (3 elements).
    """

    _x: NDArray[np.float64]  # state array of length 16

    @property
    def _pos(self) -> NDArray[np.float64]:
        return self._x[0:3]

    @_pos.setter
    def _pos(self, p: ArrayLike) -> None:
        self._x[0:3] = p

    @property
    def _vel(self) -> NDArray[np.float64]:
        return self._x[3:6]

    @_vel.setter
    def _vel(self, v: ArrayLike) -> None:
        self._x[3:6] = v

    @property
    def _q_nm(self) -> NDArray[np.float64]:
        return self._x[6:10]

    @_q_nm.setter
    def _q_nm(self, q_nm: ArrayLike) -> None:
        self._x[6:10] = q_nm

    @property
    def _bias_acc(self) -> NDArray[np.float64]:
        return self._x[10:13]

    @_bias_acc.setter
    def _bias_acc(self, b_acc: ArrayLike) -> None:
        self._x[10:13] = b_acc

    @property
    def _bias_gyro(self) -> NDArray[np.float64]:
        return self._x[13:16]

    @_bias_gyro.setter
    def _bias_gyro(self, b_gyro: ArrayLike) -> None:
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
        return self._pos.copy()

    def velocity(self) -> NDArray[np.float64]:
        """
        Get current velocity estimate.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Velocity state vector, containing (linear) velocity in x-, y-, and z-direction
            (in that order).
        """
        return self._vel.copy()

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
            ``q1`` is the real part and ``q2``, ``q3`` and ``q4`` are the three
            imaginary parts.
        """
        return self._q_nm.copy()

    def bias_acc(self) -> NDArray[np.float64]:
        """
        Get current accelerometer bias estimate.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Accelerometer bias vector, containing biases in x-, y-, and z-direction
            (in that order).
        """
        return self._bias_acc.copy()

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
        b_gyro = self._bias_gyro.copy()
        if degrees:
            b_gyro = (180.0 / np.pi) * b_gyro
        return b_gyro


class StrapdownINS(INSMixin):
    """
    Strapdown inertial navigation system (INS).

    This class provides an interface for estimating position, velocity and attitude
    of a moving body by integrating the *strapdown navigation equations*.

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
    lat : float, optional
        Latitude used to calculate the gravitational acceleration. If none
        provided, the 'standard gravity' is assumed.

    Notes
    -----
    The quaternion provided as part of the initial state will be normalized to
    ensure unity.
    """

    def __init__(self, fs: float, x0: ArrayLike, lat: float | None = None) -> None:
        self._fs = fs
        self._dt = 1.0 / fs

        self._x0 = np.asarray_chkfinite(x0).reshape(16).copy()
        self._x0[6:10] = _normalize(self._x0[6:10])
        self._x = self._x0.copy()
        self._g = np.array([0, 0, gravity(lat)])  # gravity vector in NED

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
        f_imu = np.asarray(f_imu, dtype=float)
        w_imu = np.asarray(w_imu, dtype=float)

        if degrees:
            w_imu = (np.pi / 180.0) * w_imu

        # Bias compensated IMU measurements
        f_ins = f_imu - self._bias_acc
        w_ins = w_imu - self._bias_gyro

        q_nm = self._q_nm
        R_nm = _rot_matrix_from_quaternion(q_nm)  # body-to-ned
        T = _angular_matrix_from_quaternion(q_nm)

        # State propagation (assuming constant linear acceleration and angular velocity)
        acc = R_nm @ f_ins + self._g
        self._pos = self._pos + self._dt * self._vel
        self._vel = self._vel + self._dt * acc
        q_nm = q_nm + self._dt * T @ w_ins
        self._q_nm = _normalize(q_nm)

        return self


def _h_head(q: NDArray[np.float64]) -> float:
    """
    Compute yaw angle from unit quaternion.

    Defined in terms of scaled Gibbs vector in ref [1]_, but implemented in terms of
    unit quaternion here to avoid singularities.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Unit quaternion.

    Returns
    -------
    float
        Yaw angle in the NED reference frame.

    References
    ----------
    .. [1] Fossen, T.I., "Handbook of Marine Craft Hydrodynamics and Motion Control",
    2nd Edition, equation 14.251, John Wiley & Sons, 2021.
    """
    q_w, q_x, q_y, q_z = q
    u_y = 2.0 * (q_x * q_y + q_z * q_w)
    u_x = 1.0 - 2.0 * (q_y**2 + q_z**2)
    return np.arctan2(u_y, u_x)  # type: ignore[no-any-return]


@njit  # type: ignore[misc]
def _dhda_head(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute yaw angle gradient wrt to the unit quaternion.

    Defined in terms of scaled Gibbs vector in ref [1]_, but implemented in terms of
    unit quaternion here to avoid singularities.

    Parameters
    ----------
    q : numpy.ndarray, shape (3,)
        Unit quaternion.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Yaw angle gradient vector.

    References
    ----------
    .. [1] Fossen, T.I., "Handbook of Marine Craft Hydrodynamics and Motion Control",
    2nd Edition, equation 14.254, John Wiley & Sons, 2021.
    """
    q_w, q_x, q_y, q_z = q
    u_y = 2.0 * (q_x * q_y + q_z * q_w)
    u_x = 1.0 - 2.0 * (q_y**2 + q_z**2)
    u = u_y / u_x

    duda_scale = 1.0 / u_x**2
    duda_x = -(q_w * q_y) * (1.0 - 2.0 * q_w**2) - (2.0 * q_w**2 * q_x * q_z)
    duda_y = (q_w * q_x) * (1.0 - 2.0 * q_z**2) + (2.0 * q_w**2 * q_y * q_z)
    duda_z = q_w**2 * (1.0 - 2.0 * q_y**2) + (2.0 * q_w * q_x * q_y * q_z)
    duda = duda_scale * np.array([duda_x, duda_y, duda_z])

    dhda = 1.0 / (1.0 + u**2) * duda

    return dhda  # type: ignore[no-any-return]


class AidedINS(INSMixin):
    """
    Aided inertial navigation system (AINS) using a multiplicative extended
    Kalman filter (MEKF).

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    x0_prior : array-like, shape (16,)
        Initial (a priori) INS state estimate, containing the following elements in order:

        * Position in x, y, z directions (3 elements).
        * Velocity in x, y, z directions (3 elements).
        * Attitude as unit quaternion (4 elements).
        * Accelerometer bias in x, y, z directions (3 elements).
        * Gyroscope bias in x, y, z directions (3 elements).
    P0_prior : array-like, shape (15, 15) or (12, 12)
        Initial (a priori) estimate of the error covariance matrix, **P**. If uncertain, a
        small diagonal matrix (e.g., ``1e-6 * numpy.eye(15)``) can be used. If the accelerometer
        bias is excluded from the error estimate (see ``ignore_bias_acc``), the covariance
        matrix should be of shape (12, 12) instead of (15, 15) to reflect the reduced state
        dimensionality.
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
    lever_arm : array-like, shape (3,), default numpy.zeros(3)
        Lever-arm vector describing the location of position aiding (in meters) relative
        to the IMU expressed in the IMU's measurement frame. For instance, the location
        of the GNSS antenna relative to the IMU. By default it is assumed that the
        aiding position coincides with the IMU's origin.
    lat : float, optional
        Latitude used to calculate the gravitational acceleration. If ``None`` provided,
        the 'standard gravity' of 9.80665 is assumed.
    ignore_bias_acc : bool, default True
        Determines whether the accelerometer bias should be included in the error estimate.
        If set to ``True``, the accelerometer bias provided in ``x0`` during initialization
        will remain fixed and not updated. This option is useful in situations where the
        accelerometer bias is unobservable, such as when there is insufficient aiding
        information or minimal dynamic motion, making bias estimation unreliable. Note
        that this will reduce the error-state dimension from 15 to 12, and hence also the
        error covariance matrix, **P**, from dimension (15, 15) to (12, 12).
    """

    _vg_ref_n = np.array([0.0, 0.0, 1.0])  # gravity reference vector in NED frame

    # Permutation matrix for reordering error-state bias terms, such that:
    # [pos, vel, quat, b_gyro, b_acc]^T = T_dx @ [pos, vel, quat, b_acc, b_gyro]^T
    _T_dx = np.zeros((15, 15))
    _T_dx[:9, :9] = np.eye(9)
    _T_dx[9:12, 12:15] = np.eye(3)
    _T_dx[12:15, 9:12] = np.eye(3)

    # Permutation matrix for reordering white noise bias terms, such that:
    # [acc, gyro, b_gyro, b_acc]^T = T_wn @ [acc, gyro, b_acc, b_gyro]^T
    _T_wn = np.zeros((12, 12))
    _T_wn[:6, :6] = np.eye(6)
    _T_wn[6:9, 9:12] = np.eye(3)
    _T_wn[9:12, 6:9] = np.eye(3)

    def __init__(
        self,
        fs: float,
        x0_prior: ArrayLike,
        P0_prior: ArrayLike,
        err_acc: dict[str, float],
        err_gyro: dict[str, float],
        lever_arm: ArrayLike = np.zeros(3),
        lat: float | None = None,
        ignore_bias_acc: bool = True,
    ) -> None:
        self._fs = fs
        self._dt = 1.0 / fs
        self._err_acc = err_acc
        self._err_gyro = err_gyro
        self._lat = lat
        self._lever_arm = np.asarray_chkfinite(lever_arm).reshape(3).copy()
        self._ignore_bias_acc = ignore_bias_acc
        self._dq_prealloc = np.array([2.0, 0.0, 0.0, 0.0])  # Preallocation

        # Strapdown algorithm / INS state
        self._ins = StrapdownINS(self._fs, x0_prior, lat=self._lat)

        # Total state
        self._x = self._ins.x

        # Error state
        self._dx = np.zeros(15)  # always zero, but used in sequential update

        # Initialize Kalman filter
        self._P_prior = np.asarray_chkfinite(P0_prior).copy(order="C")
        self._P = self._P_prior.copy(order="C")

        # Prepare system matrices
        q0 = self._ins._q_nm
        self._F = self._prep_F(err_acc, err_gyro, q0)
        self._G = self._prep_G(q0)
        self._H = self._prep_H()
        self._W = self._prep_W(err_acc, err_gyro)
        self._I = np.eye(15, order="C")

        # Filter out the accelerometer bias terms from the system matrices (if ignored)
        if self._ignore_bias_acc:
            dx_dim = 12
            wn_dim = 9
            self._F = (self._T_dx @ self._F @ self._T_dx.T)[:dx_dim, :dx_dim]
            self._G = (self._T_dx @ self._G @ self._T_wn)[:dx_dim, :wn_dim]
            self._H = (self._H @ self._T_dx)[:, :dx_dim]
            self._W = (self._T_wn @ self._W @ self._T_wn.T)[:wn_dim, :wn_dim]
            self._I = self._I[:dx_dim, :dx_dim]
            self._dx = self._dx[:dx_dim]

    @property
    def x_prior(self) -> NDArray[np.float64]:
        """
        Next a priori state vector estimate.

        Returns
        -------
        numpy.ndarray, shape (16,)
            A priori state vector estimate, containing the following elements in order:

            * Position in x, y, z directions (3 elements).
            * Velocity in x, y, z directions (3 elements).
            * Attitude as unit quaternion (4 elements).
            * Accelerometer bias in x, y, z directions (3 elements).
            * Gyroscope bias in x, y, z directions (3 elements).
        """
        return self._ins.x

    def dump(
        self,
    ) -> dict[str, np.float64 | list[np.float64] | dict[str, np.float64] | bool]:
        """
        Dump the configuration and current state of the AINS to a dictionary. The dumped
        parameters can be used to restore the AINS to its current state.

        Returns
        -------
        dict
            A dictionary containing the configuration and current state of the AINS.
        """
        params = {
            "fs": self._fs,
            "x0_prior": self.x_prior.tolist(),
            "P0_prior": self.P_prior.tolist(),
            "err_acc": self._err_acc,
            "err_gyro": self._err_gyro,
            "lever_arm": self._lever_arm.tolist(),
            "lat": self._lat,
            "ignore_bias_acc": self._ignore_bias_acc,
        }
        return params

    @property
    def P(self) -> NDArray[np.float64]:
        """
        Error covariance matrix, **P**. I.e., the error covariance matrix associated with
        the Kalman filter's updated (a posteriori) error-state estimate.
        """
        P = self._P.copy()
        return P

    @property
    def P_prior(self) -> NDArray[np.float64]:
        """
        Next (a priori) estimate of the error covariance matrix, **P**. I.e., the error
        covariance matrix associated with the Kalman filter's projected (a priori)
        error-state estimate.
        """
        P_prior = self._P_prior.copy()
        return P_prior

    @staticmethod
    def _prep_F(
        err_acc: dict[str, float],
        err_gyro: dict[str, float],
        q_nm: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Prepare linearized state matrix, F.
        """

        beta_acc = 1.0 / err_acc["tau_cb"]
        beta_gyro = 1.0 / err_gyro["tau_cb"]

        # Temporary placeholder vectors (to be replaced each timestep)
        f_ins = np.array([0.0, 0.0, 0.0])
        w_ins = np.array([0.0, 0.0, 0.0])

        S = _skew_symmetric  # alias skew symmetric matrix
        R_nm = _rot_matrix_from_quaternion(q_nm)  # body-to-ned rotation matrix

        # State transition matrix
        F = np.zeros((15, 15))
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -R_nm @ S(f_ins)  # NB! update each time step
        F[3:6, 9:12] = -R_nm  # NB! update each time step
        F[6:9, 6:9] = -S(w_ins)  # NB! update each time step
        F[6:9, 12:15] = -np.eye(3)
        F[9:12, 9:12] = -beta_acc * np.eye(3)
        F[12:15, 12:15] = -beta_gyro * np.eye(3)

        return F

    def _update_F(
        self,
        R_nm: NDArray[np.float64],
        f_ins: NDArray[np.float64],
        w_ins: NDArray[np.float64],
    ) -> None:
        """Update linearized state transition matrix, F."""
        S = _skew_symmetric  # alias skew symmetric matrix

        # Update matrix
        self._F[3:6, 6:9] = -R_nm @ S(f_ins)  # NB! update each time step
        self._F[6:9, 6:9] = -S(w_ins)  # NB! update each time step
        if not self._ignore_bias_acc:
            self._F[3:6, 9:12] = -R_nm  # NB! update each time step

    @staticmethod
    def _prep_G(q_nm: NDArray[np.float64]) -> NDArray[np.float64]:
        """Prepare (white noise) input matrix, G."""
        R_nm = _rot_matrix_from_quaternion(q_nm)  # body-to-ned rotation matrix

        # Input (white noise) matrix
        G = np.zeros((15, 12))
        G[3:6, 0:3] = -R_nm  # NB! update each time step
        G[6:9, 3:6] = -np.eye(3)
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)
        return G

    def _update_G(self, R_nm: NDArray[np.float64]) -> None:
        """Update (white noise) input matrix, G."""

        # Update matrix
        self._G[3:6, 0:3] = -R_nm

    @staticmethod
    def _prep_H() -> NDArray[np.float64]:
        """Prepare linearized measurement matrix, H. Values are placeholders only"""
        H = np.zeros((10, 15))
        H[0:3, 0:3] = np.eye(3)  # position
        H[3:6, 3:6] = np.eye(3)  # velocity
        return H

    def _update_H_pos(
        self, R_nm: NDArray[np.float64], lever_arm: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Update and return part of H matrix relevant for position aiding."""
        S = _skew_symmetric
        self._H[0:3, 6:9] = -R_nm @ S(lever_arm)
        return self._H[0:3]

    def _update_H_vel(self) -> NDArray[np.float64]:
        """Update and return part of H matrix relevant for velocity aiding."""
        return self._H[3:6]

    def _update_H_g_ref(self, R_nm: NDArray[np.float64]) -> NDArray[np.float64]:
        """Update and return part of H matrix relevant for g_ref aiding."""
        S = _skew_symmetric
        self._H[6:9, 6:9] = S(R_nm.T @ self._vg_ref_n)
        return self._H[6:9]

    def _update_H_head(self, q_nm: NDArray[np.float64]) -> NDArray[np.float64]:
        """Update and return part of H matrix relevant for heading aiding."""
        self._H[9:10, 6:9] = _dhda_head(q_nm)
        return self._H[9:10]

    @staticmethod
    def _prep_W(
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

    def _reset_ins(self, dx: NDArray[np.float64]) -> None:
        """Combine states and reset INS"""
        da = dx[6:9]
        self._dq_prealloc[1:4] = da
        dq = (1.0 / np.sqrt(4.0 + da.T @ da)) * self._dq_prealloc
        self._ins._x[:3] = self._ins._x[:3] + dx[:3]
        self._ins._x[3:6] = self._ins._x[3:6] + dx[3:6]
        self._ins._x[6:10] = _quaternion_product(self._ins._x[6:10], dq)
        self._ins._x[6:10] = _normalize(self._ins._x[6:10])
        self._ins._x[-3:] = self._ins._x[-3:] + dx[-3:]
        if not self._ignore_bias_acc:
            self._ins._x[10:13] = self._ins._x[10:13] + dx[9:12]
        self._dx[:] = np.zeros(dx.size)

    @staticmethod
    @njit  # type: ignore[misc]
    def _update_dx_P(
        dx: NDArray[np.float64],
        P: NDArray[np.float64],
        dz: NDArray[np.float64],
        var: NDArray[np.float64],
        H: NDArray[np.float64],
        I_: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        for i, (dz_i, var_i) in enumerate(zip(dz, var)):
            H_i = np.ascontiguousarray(H[i, :])
            K_i = P @ H_i.T / (H_i @ P @ H_i.T + var_i)
            dx += K_i * (dz_i - H_i @ dx)
            K_i = np.ascontiguousarray(K_i[:, np.newaxis])  # as 2D array
            H_i = np.ascontiguousarray(H_i[np.newaxis, :])  # as 2D array
            P = (I_ - K_i @ H_i) @ P @ (I_ - K_i @ H_i).T + var_i * K_i @ K_i.T
        return dx, P

    def update(
        self,
        f_imu: ArrayLike,
        w_imu: ArrayLike,
        degrees: bool = False,
        pos: ArrayLike | None = None,
        pos_var: ArrayLike | None = None,
        vel: ArrayLike | None = None,
        vel_var: ArrayLike | None = None,
        head: float | None = None,
        head_var: float | None = None,
        head_degrees: bool = True,
        g_ref: bool = False,
        g_var: ArrayLike | None = None,
    ) -> "AidedINS":  # TODO: Replace with ``typing.Self`` when Python > 3.11
        """
        Update/correct the AINS' state estimate with aiding measurements, and project
        ahead using IMU measurements.

        If no aiding measurements are provided, the AINS is simply propagated ahead
        using dead reckoning with the IMU measurements.

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
        degrees : bool, default False
            Specifies whether the unit of ``w_imu`` are in degrees or radians.
        pos : array-like, shape (3,), optional
            Position aiding measurement in m. If ``None``, position aiding is not used.
        pos_var : array-like, shape (3,), optional
            Variance of position measurement noise in m^2. Required for ``pos``.
        vel : array-like, shape (3,), optional
            Velocity aiding measurement in m/s. If ``None``, velocity aiding is not used.
        vel_var : array-like, shape (3,), optional
            Variance of velocity measurement noise in (m/s)^2. Required for ``vel``.
        head : float, optional
            Heading measurement, i.e., yaw angle. If ``None``, compass aiding is not used.
            See ``head_degrees`` for units.
        head_var : float, optional
            Variance of heading measurement noise. Units must be compatible with ``head``.
             See ``head_degrees`` for units. Required for ``head``.
        head_degrees : bool, default False
            Specifies whether the unit of ``head`` and ``head_var`` are in degrees and degrees^2,
            or radians and radians^2. Default is in radians and radians^2.
        g_ref : bool, optional, default False
            Specifies whether the gravity reference vector is used as an aiding measurement.
        g_var : array-like, shape (3,), optional
            Variance of gravitational reference vector measurement noise. Required for
            ``g_ref``.

        Returns
        -------
        AidedINS
            A reference to the instance itself after the update.
        """
        f_imu = np.asarray(f_imu, dtype=float)
        w_imu = np.asarray(w_imu, dtype=float)

        if degrees:
            w_imu = (np.pi / 180.0) * w_imu

        # Current INS state estimates
        pos_ins = self._ins._pos
        vel_ins = self._ins._vel
        q_ins_nm = self._ins._q_nm
        bias_acc_ins = self._ins._bias_acc
        bias_gyro_ins = self._ins._bias_gyro
        R_ins_nm = _rot_matrix_from_quaternion(q_ins_nm)  # body-to-ned rotation matrix

        # Aliases
        dx = self._dx  # zeros
        dt = self._dt
        F = self._F
        G = self._G
        W = self._W
        P = self._P_prior
        I_ = self._I

        # Bias compensated IMU measurements
        f_ins = f_imu - bias_acc_ins
        w_ins = w_imu - bias_gyro_ins

        # Lever arm vector - IMU-to-aiding
        lever_arm = self._lever_arm

        # Update system matrices
        self._update_F(R_ins_nm, f_ins, w_ins)
        self._update_G(R_ins_nm)

        # Update with available aiding measurements
        if pos is not None:
            if pos_var is None:
                raise ValueError("'pos_var' not provided.")

            pos = np.asarray(pos, dtype=float, order="C")
            pos_var = np.asarray(pos_var, dtype=float, order="C")
            dz_pos = pos - pos_ins - R_ins_nm @ lever_arm
            H_pos = self._update_H_pos(R_ins_nm, lever_arm)
            dx, P = self._update_dx_P(dx, P, dz_pos, pos_var, H_pos, I_)

        if vel is not None:
            if vel_var is None:
                raise ValueError("'vel_var' not provided.")

            vel = np.asarray(vel, dtype=float, order="C")
            vel_var = np.asarray(vel_var, dtype=float, order="C")
            dz_vel = vel - vel_ins
            H_vel = self._update_H_vel()
            dx, P = self._update_dx_P(dx, P, dz_vel, vel_var, H_vel, I_)

        if g_ref:
            if g_var is None:
                raise ValueError("'g_var' not provided.")
            vg_meas_m = -_normalize(f_ins)
            g_var = np.asarray(g_var, dtype=float, order="C")
            dz_g = vg_meas_m - R_ins_nm.T @ self._vg_ref_n
            H_g = self._update_H_g_ref(R_ins_nm)
            dx, P = self._update_dx_P(dx, P, dz_g, g_var, H_g, I_)

        if head is not None:
            if head_var is None:
                raise ValueError("'head_var' not provided.")

            if head_degrees:
                head = (np.pi / 180.0) * head
                head_var = (np.pi / 180.0) ** 2 * head_var

            head_var_ = np.asarray([head_var], dtype=float, order="C")
            dz_head = np.asarray(
                [_signed_smallest_angle(head - _h_head(q_ins_nm), degrees=False)],
                dtype=float,
                order="C",
            )

            H_head = self._update_H_head(q_ins_nm)
            dx, P = self._update_dx_P(dx, P, dz_head, head_var_, H_head, I_)

        if dx.any():
            # Reset INS state
            self._reset_ins(dx.ravel())

        # Discretize system
        phi = I_ + dt * F  # state transition matrix
        Q = dt * G @ W @ G.T  # process noise covariance matrix

        # Update current state
        self._x[:] = self._ins._x
        self._P[:] = P

        # Project ahead
        self._ins.update(f_imu, w_imu, degrees=False)
        self._P_prior[:] = phi @ P @ phi.T + Q

        return self
