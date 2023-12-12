import numpy as np
from numpy.linalg import inv
from numpy.typing import ArrayLike, NDArray

from ._ins import StrapdownINS, _signed_smallest_angle
from ._transforms import _euler_from_quaternion, _rot_matrix_from_quaternion
from ._vectorops import _normalize, _quaternion_product, _skew_symmetric


def _gibbs(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Gibbs vector"""
    q_w, q_xyz = np.split(q, [1])
    a_g = (1.0 / q_w) * q_xyz  # Gibbs vector (Eq. 14.228 in Fossen)
    return a_g


def _gibbs_scaled(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """2 x Gibbs vector"""
    return 2.0 * _gibbs(q)


def _h(a: NDArray[np.float64]) -> float:
    """See Eq. 14.251 in Fossen"""
    a_x, a_y, a_z = a
    u_y = 2.0 * (a_x * a_y + 2.0 * a_z)
    u_x = 4.0 + a_x**2 - a_y**2 - a_z**2
    return np.arctan2(u_y, u_x)


def _dhda(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """See Eq. 14.254 in Fossen"""
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

    return dhda


class MEKF:
    """
    Aided inertial navigation system (AINS) using a multiplicative extended
    Kalman filter (MEKF).

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    x0 : array-like (16,)
        Initial state vector containing the following elements in order:
            - Position in x, y, z directions (3 elements).
            - Velocity in x, y, z directions (3 elements).
            - Attitude as unit quaternion (4 elements).
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
            - B: Bias stability in rad/s.
            - tau_cb: Bias correlation time in seconds.
    var_pos : array-like (3,)
        Variance of position measurement noise in m^2.
    var_compass : float
        Variance of compass measurement noise in deg^2.
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
    ) -> None:
        self._fs = fs
        self._dt = 1.0 / fs
        self._err_acc = err_acc
        self._err_gyro = err_gyro
        self._x0 = np.asarray_chkfinite(x0).reshape(16).copy()
        self._var_pos = np.asarray_chkfinite(var_pos).reshape(3).copy()
        self._var_vel = np.asarray_chkfinite(var_vel).reshape(3).copy()
        self._var_g = np.asarray_chkfinite(var_g).reshape(3).copy()
        self._var_compass = np.asarray_chkfinite(var_compass).reshape(1).copy()

        # Strapdown algorithm
        self._x_ins = self._x0.copy()
        self._ins = StrapdownINS(self._x_ins[0:10])

        # Initial Kalman filter error covariance
        self._P_prior = np.eye(15)

        # Prepare system matrices
        q0 = self._x0[6:10]
        self._dfdx = self._prep_dfdx_matrix(err_acc, err_gyro, q0)
        self._dfdw = self._prep_dfdw_matrix(q0)
        self._dhdx = self._prep_dhdx_matrix(q0)
        self._W = self._prep_W_matrix(err_acc, err_gyro)

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
    def _q(self) -> NDArray[np.float64]:
        return self._x[6:10]

    @property
    def _b_acc(self) -> NDArray[np.float64]:
        return self._x[10:13]

    @property
    def _b_gyro(self) -> NDArray[np.float64]:
        return self._x[13:16]

    @property
    def x(self) -> NDArray[np.float64]:
        """
        Current AINS state estimate.

        Returns
        -------
        x : numpy.ndarray (16,)
            The current state vector, containing the following elements in order:
                - Position in x, y, z directions (3 elements).
                - Velocity in x, y, z directions (3 elements).
                - Attitude as unit quaternion (4 elements).
                - Accelerometer bias in x, y, z directions (3 elements).
                - Gyroscope bias in x, y, z directions (3 elements).
        """
        return self._x.copy()

    def position(self) -> NDArray[np.float64]:
        """
        Current AINS position estimate.

        Returns
        -------
        position : numpy.ndarray (3,)
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
        position : numpy.ndarray (3,)
            The current velocity vector, containing the following elements:
                - Velocity in x direction.
                - Velocity in y direction.
                - Velocity in z direction.
        """
        return self._v.copy()

    def quaternion(self) -> NDArray[np.float64]:
        """
        Current attitude estimate as unit quaternion (from-body-to-NED).

        Returns
        -------
        q : numpy.ndarray (4,)
            Attitude as unit quaternion. Given as [q1, q2, q3, q4], where q1 is
            the real part and q1, q2 and q3 are the three imaginary parts.
        """
        return self._q.copy()

    def euler(self, degrees: bool = False) -> NDArray[np.float64]:
        """
        Current attitude estimate as Euler angles (see Notes).

        Parameters
        ----------
        degrees : bool
            Whether to return the Euler angles in degrees (`True`) or radians (`False`).

        Returns
        -------
        euler : numpy.ndarray (3,)
            Euler angles, specifically: alpha (roll), beta (pitch) and gamma (yaw)
            in that order.

        Notes
        -----
        The Euler angles describe how to transition from the 'NED' frame to the 'body'
        frame through three consecutive intrinsic and passive rotations in the ZYX order:
            1. A rotation by an angle gamma (often called yaw) about the z-axis.
            2. A subsequent rotation by an angle beta (often called pitch) about the y-axis.
            3. A final rotation by an angle alpha (often called roll) about the x-axis.

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

        return theta

    def bias_acc(self) -> NDArray[np.float64]:
        """
        Current accelerometer bias estimate.

        Returns
        -------
        b_acc : numpy.ndarray (3,)
            The current accelerometer bias vector, containing the following elements:
                - x-axis acceleration bias.
                - y-axis acceleration bias.
                - z-axis acceleration bias.
        """
        return self._b_acc.copy()

    def bias_gyro(self) -> NDArray[np.float64]:
        """
        Current gyroscope bias estimate.

        Returns
        -------
        b_gyro : numpy.ndarray (3,)
            The current gyroscope bias vector, containing the following elements:
                - x-axis rotation rate bias.
                - y-axis rotation rate bias.
                - z-axis rotation rate bias.
        """
        return self._b_acc.copy()

    @staticmethod
    def _prep_dfdx_matrix(
        err_acc: dict[str, float],
        err_gyro: dict[str, float],
        q: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Prepare linearized state matrix"""

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

    def _prep_dhdx_matrix(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
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
        dhdx[9:10, 6:9] = _dhda(_gibbs_scaled(q))  # compass
        return dhdx

    def _update_dhdx_matrix(self, q: NDArray[np.float64]) -> None:
        """Update linearized measurement matrix"""

        # Reference vector
        v01_ned = np.array([0.0, 0.0, 1.0])

        # Aliases for transformation matrices
        R = _rot_matrix_from_quaternion  # body-to-ned rotation matrix
        S = _skew_symmetric  # skew symmetric matrix

        self._dhdx[6:9, 6:9] = S(R(q).T @ v01_ned)  # gravity reference vector
        self._dhdx[9:10, 6:9] = _dhda(_gibbs_scaled(q))  # compass

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
    ) -> "MEKF":  # TODO: Replace with ``typing.Self`` when Python > 3.11
        """
        Update the AINS state estimates based on measurements, and project ahead.

        Parameters
        ----------
        f_imu : array-like (3,)
            Acceleration / specific force measurements.
        w_imu : array-like (3,)
            Rotation rate measurements.
        pos : array-like (3,), default=None
            Position aiding measurement. If `None`, no position aiding is used.
        vel : array-like (3,), default=None
            Velocity aiding measurement. If `None`, no velocity aiding is used.
        head : float
            Heading (i.e., yaw angle) aiding measurement. If `head_degrees` is
            `True`, the heading is assumed to be in degrees; otherwise, in radians.
        degrees : bool, default=False
            Specifies the units of the `w_imu` parameter. If `True`, the rotation
            rates are assumed to be in degrees; otherwise, in radians.
        head_degrees : bool, default=True
            Specifies the unit of the `head` parameter. If `True`, the heading is
            in degrees; otherwise, in radians.
        """
        f_imu = np.asarray_chkfinite(f_imu, dtype=float).reshape(3).copy()
        w_imu = np.asarray_chkfinite(w_imu, dtype=float).reshape(3).copy()

        if degrees:
            w_imu = np.radians(w_imu)
        if head_degrees:
            head = np.radians(head)

        # Update INS state
        self._x_ins[0:10] = self._ins.x

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
        dz, var_z, dhdx = [], [], []
        if pos is not None:
            pos = np.asarray_chkfinite(pos, dtype=float).reshape(3).copy()
            delta_pos = pos - p_ins
            dz.append(delta_pos)
            var_z.append(self._var_pos)
            dhdx.append(dhdx_[0:3])

        # Velocity aiding
        if vel is not None:
            vel = np.asarray_chkfinite(vel, dtype=float).reshape(3).copy()
            delta_vel = vel - v_ins
            dz.append(delta_vel)
            var_z.append(self._var_vel)
            dhdx.append(dhdx_[3:6])

        # Gravity reference vector aiding
        delta_g = v1 - R_bn.T @ v01
        dz.append(delta_g)
        var_z.append(self._var_g)
        dhdx.append(dhdx_[6:9])

        # Compass aiding
        if head is not None:
            head = np.asarray_chkfinite(head, dtype=float).reshape(1).copy()
            delta_head = head - _h(_gibbs_scaled(q_ins))
            dz.append(_signed_smallest_angle(delta_head, degrees=False))
            var_z.append(self._var_compass)
            dhdx.append(dhdx_[-1:])

        dz = np.concatenate(dz)
        dhdx = np.vstack(dhdx)
        R = np.diag(np.concatenate(var_z))

        # Discretize system
        phi = I15 + self._dt * dfdx  # state transition matrix
        Q = self._dt * dfdw @ W @ dfdw.T  # process noise covariance matrix

        # Compute Kalman gain
        K = P_prior @ dhdx.T @ inv(dhdx @ P_prior @ dhdx.T + R)

        # Update error-state estimate with measurement
        dx = K @ dz

        # Compute error covariance for updated estimate
        P = (I15 - K @ dhdx) @ P_prior @ (I15 - K @ dhdx).T + K @ R @ K.T

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
        self._x_ins = x_ins
        self._ins.reset(self._x_ins[:10])

        # Project ahead
        self._ins.update(self._dt, f_ins, w_ins, degrees=False)
        self._P_prior = phi @ P @ phi.T + Q
