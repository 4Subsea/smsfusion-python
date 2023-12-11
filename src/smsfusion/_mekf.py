import numpy as np
from numpy.linalg import inv
from numpy.typing import ArrayLike, NDArray

from ._ins import StrapdownINS, gravity, _signed_smallest_angle
from ._transforms import _euler_from_quaternion, _rot_matrix_from_quaternion
from ._vectorops import _normalize, _quaternion_product, _skew_symmetric


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
        self._x0 = np.asarray_chkfinite(x0).reshape(16, 1).copy()
        self._var_pos = np.asarray_chkfinite(var_pos).reshape(3).copy()
        self._var_vel = np.asarray_chkfinite(var_vel).reshape(3).copy()
        self._var_g = np.asarray_chkfinite(var_g).reshape(3).copy()
        self._var_compass = np.asarray_chkfinite(var_compass).reshape(1).copy()

        # Strapdown algorithm
        self._x_ins = self._x0
        self._ins = StrapdownINS(self._x_ins[0:10])

        # Initial Kalman filter error covariance
        self._P_prior = np.eye(15)

        # Prepare system matrices
        q0 = self._x0[6:10].flatten()
        self._dfdx = self._prep_dfdx_matrix(err_acc, err_gyro, q0)
        self._dfdw = self._prep_dfdw_matrix(q0)
        self._dhdx = self._prep_dhdx_matrix(q0)
        self._W = self._prep_W_matrix(err_acc, err_gyro)
        # self._R = np.diag(np.r_[var_pos, var_vel, var_g, var_compass])

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
        return self._x.flatten()

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
        return self._p.flatten()

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
        return self._v.flatten()

    def quaternion(self) -> NDArray[np.float64]:
        """
        Current attitude estimate as unit quaternion (from-body-to-NED).

        Returns
        -------
        q : numpy.ndarray (4,)
            Attitude as unit quaternion. Given as [q1, q2, q3, q4], where q1 is
            the real part and q1, q2 and q3 are the three imaginary parts.
        """
        return self._q.flatten()

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

    @staticmethod
    def _h_gamma(q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Linearization of compass measurement"""

        q_w, q_xyz = np.split(q, [1])

        a = (2.0 / q_w) * q_xyz  # 2 x Gibbs vector (Eq. 14.257 in Fossen)

        a_x, a_y, a_z = a
        u_n = 2.0 * (a_x * a_y + 2.0 * a_z)
        u_d = 4.0 + a_x**2 - a_y**2 - a_z**2
        u = u_n / u_d  # (Eq. 14.255 in Fossen)

        duda_scale = 1.0 / (4.0 + a_x**2 - a_y**2 - a_z**2) ** 2
        duda_x = -2.0 * ((a_x**2 + a_z**2 - 4.0) * a_y + a_y**3 + 4.0 * a_y * a_z)
        duda_y = 2.0 * ((a_y**2 - a_z**2 + 4.0) * a_x + a_x**3 + 4.0 * a_y * a_z)
        duda_z = 4.0 * (a_z**2 + a_x * a_y * a_z + a_x**2 - a_y**2 + 4.0)
        duda = duda_scale * np.array([duda_x, duda_y, duda_z])  # (Eq. 14.256 in Fossen)

        dhda = 1.0 / (1.0 + np.sum(u**2)) * duda  # (Eq. 14.254 in Fossen)
        return dhda

    def _prep_dhdx_matrix(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Prepare linearized measurement matrix"""

        # Reference vector
        v01_ned = np.array([0.0, 0.0, 1.0]).reshape(3, 1)

        # Aliases for transformation matrices
        R = _rot_matrix_from_quaternion  # body-to-ned rotation matrix
        S = _skew_symmetric  # skew symmetric matrix

        # Linearization of compass measurement
        h_gamma = self._h_gamma(q)

        dhdx = np.zeros((10, 15))
        dhdx[0:3, 0:3] = np.eye(3)  # position
        dhdx[3:6, 3:6] = np.eye(3)  # velocity
        dhdx[6:9, 6:9] = S((R(q).T @ v01_ned).flatten())  # gravity reference vector
        dhdx[9:10, 6:9] = h_gamma  # compass
        return dhdx

    def _update_dhdx_matrix(self, q: NDArray[np.float64]) -> None:
        """Update linearized measurement matrix"""

        # Reference vector
        v01_ned = np.array([0.0, 0.0, 1.0]).reshape(3, 1)

        # Aliases for transformation matrices
        R = _rot_matrix_from_quaternion  # body-to-ned rotation matrix
        S = _skew_symmetric  # skew symmetric matrix

        self._dhdx[6:9, 6:9] = S(
            (R(q).T @ v01_ned).flatten()
        )  # gravity reference vector
        self._dhdx[9:10, 6:9] = self._h_gamma(q)  # compass

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
        head: float | None = None,
        pos: ArrayLike | None = None,
        vel: ArrayLike | None = None,
        degrees: bool = False,
        head_degrees: bool = True,
    ) -> "MEKF":  # TODO: Replace with ``typing.Self`` when Python > 3.11
        """
        Update the AINS state estimates based on measurements, and project ahead.

        Parameters
        ----------
        f_imu : array-like (3,)
            IMU specific force measurements (i.e., accelerations + gravity). Given
            as `[f_x, f_y, f_z]^T` where `f_x`, `f_y`, and `f_z` are acceleration
            measurements in the x-, y-, and z-directions, respectively.
        w_imu : array-like (3,)
            IMU rotation rate measurements. Given as `[w_x, w_y, w_z]^T` where `w_x`,
            `w_y`, and `w_z` are rotation rates about the x-, y-, and z-axes, respectively.
            The unit is determined by the `degrees` keyword argument.
        head : float
            Heading measurement, i.e., yaw angle. If `head_degrees` is `True`, the
            heading is assumed to be in degrees; otherwise, in radians.
        pos : array-like (3,), default=None
            Position aiding measurement. If `None`, no position aiding is used.
        degrees : bool, default=False
            Specifies the units of the `w_imu` parameter. If `True`, the rotation
            rates are assumed to be in degrees; otherwise, in radians.
        head_degrees : bool, default=True
            Specifies the unit of the `head` parameter. If `True`, the heading is
            in degrees; otherwise, in radians.
        """
        f_imu = np.asarray_chkfinite(f_imu, dtype=float).reshape(3, 1).copy()
        w_imu = np.asarray_chkfinite(w_imu, dtype=float).reshape(3, 1).copy()

        if degrees:
            w_imu = np.radians(w_imu)
        if head_degrees:
            head = np.radians(head)

        # Update INS state
        self._x_ins[0:10] = self._ins.x.reshape(10, 1)

        # Bias compensated IMU measurements
        b_acc_ins = self._x_ins[10:13]
        b_gyro_ins = self._x_ins[13:16]
        f_ins = f_imu - b_acc_ins
        w_ins = w_imu - b_gyro_ins

        # Update system matrices
        q = self._q.reshape(4)
        self._update_dfdx_matrix(q, f_ins.reshape(3), w_ins.reshape(3))
        self._update_dfdw_matrix(q)
        self._update_dhdx_matrix(q)

        dfdx = self._dfdx  # state matrix
        dfdw = self._dfdw  # (white noise) input matrix
        dhdx_ = self._dhdx  # measurement matrix
        W = self._W  # white noise power spectral density matrix
        P_prior = self._P_prior  # error covariance matrix
        I15 = self._I15  # 15x15 identity matrix

        # Gravity vector measured
        v01 = np.array([0.0, 0.0, 1.0]).reshape(3, 1)
        v1 = -f_ins / gravity()  # gravity vector measured
        v1 = _normalize(v1)

        R_bn = _rot_matrix_from_quaternion(self._x_ins[6:10].reshape(4))

        dz = []
        var_z = []
        dhdx = []
        if pos is not None:
            pos = np.asarray_chkfinite(pos, dtype=float).reshape(3, 1).copy()
            dz.append(pos - self._x_ins[0:3])
            var_z.append(self._var_pos)
            dhdx.append(dhdx_[0:3])
        if vel is not None:
            vel = np.asarray_chkfinite(vel, dtype=float).reshape(3, 1).copy()
            dz.append(vel - self._x_ins[3:6])
            var_z.append(self._var_vel)
            dhdx.append(dhdx_[3:6])
        dz.append(v1 - R_bn.T @ v01)
        var_z.append(self._var_g)
        dhdx.append(dhdx_[6:9])
        if head is not None:
            q_w, q_xyz = np.split(q, [1])
            a = (2.0 / q_w) * q_xyz  # 2 x Gibbs vector (Eq. 14.257 in Fossen)
            a_x, a_y, a_z = a
            u_n = 2.0 * (a_x * a_y + 2.0 * a_z)
            u_d = 4.0 + a_x**2 - a_y**2 - a_z**2
            head = np.asarray_chkfinite(head, dtype=float).reshape(1, 1).copy()
            dz.append(_signed_smallest_angle(head - np.arctan2(u_n, u_d)))
            var_z.append(self._var_compass)
            dhdx.append(dhdx_[-1:])
        dz = np.vstack(dz)
        dhdx = np.vstack(dhdx)
        R = np.diag(np.concatenate(var_z))

        # Discretize system
        phi = I15 + self._dt * dfdx  # state transition matrix
        Q = self._dt * dfdw @ W @ dfdw.T  # process noise covariance matrix

        # Compute Kalman gain
        K = P_prior @ dhdx.T @ inv(dhdx @ P_prior @ dhdx.T + R)

        # Update error-state estimate with measurement
        # dz = z - H @ self._x_ins  # TODO
        dx = K @ dz

        # Compute error covariance for updated estimate
        P = (I15 - K @ dhdx) @ P_prior @ (I15 - K @ dhdx).T + K @ R @ K.T

        # Error quaternion from 2x Gibbs vector
        da = dx[6:9]
        dq = (1.0 / np.sqrt(4.0 + da.T @ da)) * np.vstack([2.0, da])

        # Reset
        self._x_ins[0:6] = self._x_ins[0:6] + dx[0:6]
        self._x_ins[6:10] = _quaternion_product(
            self._x_ins[6:10].flatten(), dq.flatten()
        ).reshape(4, 1)
        self._x_ins[10:] = self._x_ins[10:] + dx[9:]
        self._ins.reset(self._x_ins[0:10])

        # Project ahead
        f_ins = f_imu - self._x_ins[9:12]
        w_ins = w_imu - self._x_ins[12:15]
        self._ins.update(self._dt, f_ins, w_ins, degrees=False)
        self._P_prior = phi @ P @ phi.T + Q
