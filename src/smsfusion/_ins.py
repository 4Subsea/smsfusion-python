from __future__ import annotations

import numpy as np
from numpy.linalg import inv
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm

from ._ahrs import AHRS
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


def van_loan(dt, F, G, W):
    """
    Calculate the state transition matrix, ``phi``, and the process noise covariance
    matrix, ``Q``, using the 'Van Loan method'.
    """
    F = np.asarray_chkfinite(F)
    G = np.asarray_chkfinite(G)
    W = np.asarray_chkfinite(W)

    n_states = F.shape[0]
    A = np.zeros((2 * n_states, 2 * n_states))
    A[:n_states, :n_states] = -F
    A[:n_states, n_states:] = G @ W @ G.T
    A[n_states:, n_states:] = F.T
    A = dt * A

    B = expm(A)

    phi = B[n_states:, n_states:].T
    Q = phi @ B[:n_states, n_states:]

    return phi, Q


class StrapdownINS:
    """
    Inertial navigation system (INS) strapdown algorithm.

    This class provides an interface for estimating position, velocity and attitude
    of a moving body by integrating the 'strapdown navigation equations'.

    Parameters
    ----------
    x0 : array_like
        Initial state vector as 1D array of length 9 (see Notes).
    lat : float, optional
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

        Parameters
        ----------
        degrees : bool, default False
            Whether the rotation rates are given in `degrees` (``True``) or `radians`
            (``False``).

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
        x_new : array_like
            New state as 1D array of length 9 (see Notes).

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
        f_imu : array_like
            IMU specific force measurements (i.e., accelerations + gravity). Given as
            ``[f_x, f_y, f_z]^T`` where ``f_x``, ``f_y`` and ``f_z`` are
            acceleration measurements in x-, y-, and z-direction, respectively.
        w_imu : array_like
            IMU rotation rate measurements. Given as ``[w_x, w_y, w_z]^T`` where
            ``w_x``, ``w_y`` and ``w_z`` are rotation rates about the x-, y-,
            and z-axis, respectively. Unit determined with ``degrees`` keyword argument.
        degrees : bool, default False
            Whether the rotation rates are given in `degrees` (``True``) or `radians`
            (``False``).
        theta_ext : array_like, optional
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


class AidedINS:
    """
    Aided inertial navigation system (AINS).

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    x0 : array-like (15,)
        Initial state vector containing the following elements in order:
            - Position in x, y, z directions (3 elements).
            - Velocity in x, y, z directions (3 elements).
            - Euler angles: alpha (roll), beta (pitch), and gamma (yaw) (3 elements).
            - Accelerometer bias in x, y, z directions (3 elements).
            - Gyroscope bias in x, y, z directions (3 elements).
    err_acc : dict
        Dictionary containing accelerometer noise parameters:
            - N: White noise power spectral density in (m/s^2)/sqrt(Hz).
            - B: Bias stability in m/s^2.
            - tau_cb: Bias correlation time in seconds.
    err_gyro : dict
        Dictionary containing gyroscope noise parameters:
            - N: White noise power spectral density in (rad/s)/sqrt(Hz).
            - B: Bias stability in 'rad/s'.
            - tau_cb: Bias correlation time in 's'.
    var_pos : array-like (3,)
        Variance of position measurement noise in m/s^2
    var_ahrs : array-like (3,)
        Variance of attitude measurements in rad^2. Specifically, it refers to the
        variance of the AHRS error.

    Notes
    -----
    This AINS model has the following limitations:
        - Assumes aiding measurements are available at every time step.
        - Supports only position and AHRS aiding.
        - Operates at a constant sampling rate.
        - AHRS gain factor parameters are not configurable.
        - Initial error covariance matrix, P, is fixed to the identity matrix, and
          cannot be set during initialization.
    """

    _I15 = np.eye(15)

    _Kp = 0.05
    _Ki = 0.035

    def __init__(self, fs, x0, err_acc, err_gyro, var_pos, var_ahrs):
        self._fs = fs
        self._dt = 1.0 / fs
        self._err_acc = err_acc
        self._err_gyro = err_gyro
        self._x0 = np.asarray_chkfinite(x0).reshape(15, 1).copy()
        var_pos = np.asarray_chkfinite(var_pos).reshape(3).copy()
        var_ahrs = np.asarray_chkfinite(var_ahrs).reshape(3).copy()

        # Attitude Heading Reference System (AHRS)
        self._ahrs = AHRS(fs, self._Kp, self._Ki)  # TODO: use initial attitude from x0

        # Strapdown algorithm
        self._ins = StrapdownINS(self._x0[0:9])
        self._bias_ins = self._x0[9:15]

        # Initial Kalman filter error covariance
        self._P_prior = np.eye(15)

        # Prepare system matrices
        self._F = self._prep_F_matrix(err_acc, err_gyro, self._theta.flatten())
        self._G = self._prep_G_matrix(self._theta.flatten())
        self._W = self._prep_W_matrix(err_acc, err_gyro)
        self._H = self._prep_H_matrix()
        self._R = np.diag(np.r_[var_pos, var_ahrs])

    @property
    def _x_ins(self):
        """INS state"""
        x_ins = np.r_[self._ins.x, self._bias_ins]
        return x_ins

    @property
    def _x(self):
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
        Current AINS state estimate.

        Returns
        -------
        x : ndarray (15, 1)
            The current state vector, containing the following elements in order:
                - Position in x, y, z directions (3 elements).
                - Velocity in x, y, z directions (3 elements).
                - Euler angles: alpha (roll), beta (pitch), and gamma (yaw) (3 elements).
                - Accelerometer bias in x, y, z directions (3 elements).
                - Gyroscope bias in x, y, z directions (3 elements).
        """
        return self._x.copy()

    def position(self) -> NDArray[np.float64]:
        """
        Current AINS position estimate.

        Returns
        -------
        position : ndarray (3, 1)
            The current position vector, containing the following elements:
                - Position in x direction.
                - Position in y direction.
                - Position in z direction.
        """
        return self._p.copy()

    def velocity(self) -> NDArray[np.float64]:
        """
        Current AINS velocity estimate.

        Returns
        -------
        position : ndarray (3, 1)
            The current velocity vector, containing the following elements:
                - Velocity in x direction.
                - Velocity in y direction.
                - Velocity in z direction.
        """
        return self._v.copy()

    def attitude(self, degrees: bool = False) -> NDArray[np.float64]:
        """
        Current AINS attitude estimate as Euler angles (i.e., roll, pitch, yaw).

        Parameters
        ----------
        degrees : bool, default=False
            If `True`, the Euler angles are returned in degrees. If `False`, they
            are returned in radians.

        Returns
        -------
        attitude : ndarray (3, 1)
            The current attitude estimate, represented by Euler angles:
                - alpha (roll).
                - beta (pitch).
                - gamma (yaw).
        """
        theta = self._theta.copy()

        if degrees:
            theta = (180.0 / np.pi) * theta

        return theta

    @staticmethod
    def _prep_F_matrix(err_acc, err_gyro, theta_rad):
        """Prepare state matrix"""

        beta_acc = 1.0 / err_acc["tau_cb"]
        beta_gyro = 1.0 / err_gyro["tau_cb"]

        R_bn = _rot_matrix_from_euler(theta_rad).T  # body-to-NED
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
    def _prep_G_matrix(theta_rad):
        """Prepare (white noise) input matrix"""

        R_bn = _rot_matrix_from_euler(theta_rad).T  # body-to-NED
        T = _angular_matrix_from_euler(theta_rad)

        # Input (white noise) matrix
        G = np.zeros((15, 12))
        G[3:6, 0:3] = -R_bn  # NB! update each time step
        G[6:9, 3:6] = -T  # NB! update each time step
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)

        return G

    @staticmethod
    def _prep_W_matrix(err_acc, err_gyro):
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

    @staticmethod
    def _prep_H_matrix():
        """Prepare measurement matrix"""
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)  # position
        H[3:6, 6:9] = np.eye(3)  # attitude
        return H

    def update(self, f_imu, w_imu, head, pos, degrees=False, head_degrees=True):
        """
        Update the AINS with measurements, and project ahead.

        Parameters
        ----------
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
        head_degrees : bool
            If ``True`` (default), the heading is assumed to be in
            degrees. Otherwise in radians.
        """
        f_imu = np.asarray_chkfinite(f_imu, dtype=float).reshape(3, 1).copy()
        w_imu = np.asarray_chkfinite(w_imu, dtype=float).reshape(3, 1).copy()
        pos = np.asarray_chkfinite(pos, dtype=float).reshape(3, 1).copy()
        head = np.asarray_chkfinite(head, dtype=float).reshape(1, 1).copy()
        theta_ext = self._ahrs.attitude(degrees=False)

        if degrees:
            w_imu = np.radians(w_imu)
        if head_degrees:
            head = np.radians(head)

        # Setup transformation matrices based on AHRS 'measurement'
        R_bn = _rot_matrix_from_euler(theta_ext).T  # body-to-NED rotation matrix
        T = _angular_matrix_from_euler(theta_ext)  # rotation rates to Euler rates

        # Update system matrices with AHRS attitude 'measurements'
        self._F[3:6, 9:12] = -R_bn
        self._F[6:9, 12:15] = -T
        self._G[3:6, 0:3] = -R_bn
        self._G[6:9, 3:6] = -T

        F = self._F  # state matrix
        G = self._G  # (white noise) input matrix
        H = self._H  # measurement matrix
        W = self._W  # white niser powerr spectral density matrix
        R = self._R  # measurement noise covariance matrix
        P_prior = self._P_prior  # error covariance matrix
        x_ins = self._x_ins  # INS state
        I15 = self._I15  # 15x15 identity matrix

        # Discretize
        phi = I15 + self._dt * F  # state transition matrix
        Q = self._dt * G @ W @ G.T  # process noise covariance matrix

        # Measurement
        z = np.r_[pos, theta_ext.reshape(3, 1)]

        # Compute Kalman gain
        K = P_prior @ H.T @ inv(H @ P_prior @ H.T + R)

        # Update error-state estimate with measurement
        dz = z - H @ x_ins
        dx = K @ dz

        # Compute error covariance for updated estimate
        P = (I15 - K @ H) @ P_prior @ (I15 - K @ H).T + K @ R @ K.T

        # Reset
        self._ins.reset(self._ins.x + dx[0:9])
        self._bias_ins = dx[9:15]

        # Project ahead
        f_ins = f_imu - self._bias_ins[0:3]
        w_ins = w_imu - self._bias_ins[3:6]
        self._ins.update(self._dt, f_ins, w_ins, theta_ext=theta_ext, degrees=False)
        self._ahrs.update(f_imu, w_imu, head, degrees=False, head_degrees=False)
        self._P_prior = phi @ P @ phi.T + Q
