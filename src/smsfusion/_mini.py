from typing import Self

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray
from ._transforms import _euler_from_quaternion, _angular_matrix_from_quaternion, _rot_matrix_from_quaternion, _quaternion_from_euler
from ._vectorops import _normalize, _quaternion_product, _skew_symmetric
from ._ins import _h_head, _signed_smallest_angle, _roll_pitch_from_acc, _dhda_head
from smsfusion.constants import ERR_GYRO_MOTION2


class AHRSMixin:
    """
    Mixin class for inertial navigation systems (INS).

    Requires that the inheriting class has an `_x` attribute which is a 1D numpy array
    of length 7 containing the following elements in order:

        * Attitude as unit quaternion (4 elements).
        * Gyroscope bias in x, y, z directions (3 elements).
    """

    _x: NDArray[np.float64]  # state array of length 7

    @property
    def _q_nm(self) -> NDArray[np.float64]:
        return self._x[:4]

    @_q_nm.setter
    def _q_nm(self, q_nm: ArrayLike) -> None:
        self._x[:4] = q_nm

    @property
    def _bias_gyro(self) -> NDArray[np.float64]:
        return self._x[4:7]

    @_bias_gyro.setter
    def _bias_gyro(self, b_gyro: ArrayLike) -> None:
        self._x[4:7] = b_gyro

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
        The Euler angles describe how to transition from the 'navigation' frame
        ('NED' or 'ENU) to the 'body' frame through three consecutive intrinsic
        and passive rotations in the ZYX order:

        #. A rotation by an angle gamma (often called yaw) about the z-axis.
        #. A subsequent rotation by an angle beta (often called pitch) about the y-axis.
        #. A final rotation by an angle alpha (often called roll) about the x-axis.

        This sequence of rotations is used to describe the orientation of the 'body' frame
        relative to the 'navigation' frame ('NED' or 'ENU) in 3D space.

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
        Get current attitude estimate as unit quaternion (from-body-to-navigation-frame).

        Returns
        -------
        numpy.ndarray, shape (4,)
            Attitude as unit quaternion. Given as ``[q1, q2, q3, q4]``, where
            ``q1`` is the real part and ``q2``, ``q3`` and ``q4`` are the three
            imaginary parts.
        """
        return self._q_nm.copy()

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


class StrapdownAHRS(AHRSMixin):
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
    g : float, default 9.80665
        The gravitational acceleration. Default is 'standard gravity' of 9.80665.
    nav_frame : {'NED', 'ENU'}, default 'NED'
        Specifies the assumed inertial-like 'navigation' frame. Should be 'NED' (North-East-Down)
        (default) or 'ENU' (East-North-Up). The body's (or IMU sensor's) degrees of freedom
        will be expressed relative to this frame.

    Notes
    -----
    The quaternion provided as part of the initial state will be normalized to
    ensure unity.
    """

    def __init__(self, fs: float, x0: ArrayLike) -> None:
        self._fs = fs
        self._dt = 1.0 / fs

        self._x0 = np.asarray_chkfinite(x0).reshape(7).copy()
        self._x0[:4] = _normalize(self._x0[:4])
        self._x = self._x0.copy()

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
        self._x = np.asarray_chkfinite(x_new).reshape(7).copy()
        self._x[:4] = _normalize(self._x[:4])

    def update(
        self,
        w_imu: ArrayLike,
        degrees: bool = False,
    ) -> Self:
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
        w_imu = np.asarray(w_imu, dtype=float)

        if degrees:
            w_imu = (np.pi / 180.0) * w_imu

        # Bias compensated IMU measurements
        w_ins = w_imu - self._bias_gyro

        q_nm = self._q_nm
        T = _angular_matrix_from_quaternion(q_nm)

        # State propagation (assuming constant linear acceleration and angular velocity)
        q_nm = q_nm + self._dt * T @ w_ins
        self._q_nm = _normalize(q_nm)

        return self




class MiniAHRS:
    """
    Aided inertial navigation system (AINS) using a multiplicative extended
    Kalman filter (MEKF).

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    x0_prior : array-like, shape (16,), default :const:`smsfusion.constants.X0`
        Initial (a priori) 16-element INS state estimate:

        * Position (x, y, z) - 3 elements
        * Velocity (x, y, z) - 3 elements
        * Attitude (unit quaternion) - 4 elements
        * Accelerometer bias (x, y, z) - 3 elements
        * Gyroscope bias (x, y, z) - 3 elements

        Defaults to a zero vector, but with the attitude part as a unit quaternion
        (i.e., no rotation).
    P0_prior : array-like (shape (12, 12) or (15, 15)), default np.eye(12) * 1e-6 (:const:`smsfusion.constants.P0`)
        Initial (a priori) estimate of the error covariance matrix, **P**. If not given, a
        small diagonal matrix will be used. If the accelerometer bias is excluded from the
        error estimate (see ``ignore_bias_acc``), the covariance matrix should be of shape
        (12, 12), otherwise (15, 15).
    err_acc : dict of {str: float}, default :const:`smsfusion.constants.ERR_ACC_MOTION2`
        Dictionary containing accelerometer noise parameters with keys:

        * ``N``: White noise power spectral density in (m/s^2)/sqrt(Hz).
        * ``B``: Bias stability in m/s^2.
        * ``tau_cb``: Bias correlation time in seconds.

        Defaults to error characteristics of SMS Motion gen. 2.
    err_gyro : dict of {str: float}, default :const:`smsfusion.constants.ERR_GYRO_MOTION2`
        Dictionary containing gyroscope noise parameters with keys:

        * ``N``: White noise power spectral density in (rad/s)/sqrt(Hz).
        * ``B``: Bias stability in rad/s.
        * ``tau_cb``: Bias correlation time in seconds.

        Defaults to error characteristics of SMS Motion gen. 2.
    g : float, default 9.80665
        The gravitational acceleration. Default is 'standard gravity' of 9.80665.
    nav_frame : {'NED', 'ENU'}, default 'NED'
        Specifies the assumed inertial-like 'navigation' frame. Should be 'NED' (North-East-Down)
        (default) or 'ENU' (East-North-Up). The body's (or IMU sensor's) degrees of freedom
        will be expressed relative to this frame. Furthermore, the aiding heading angle is
        also interpreted relative to this frame according to the right-hand rule.
    lever_arm : array-like, shape (3,), default numpy.zeros(3)
        Lever-arm vector describing the location of position aiding (in meters) relative
        to the IMU expressed in the IMU's measurement frame. For instance, the location
        of the GNSS antenna relative to the IMU. By default it is assumed that the
        aiding position coincides with the IMU's origin.
    ignore_bias_acc : bool, default True
        Determines whether the accelerometer bias should be included in the error estimate.
        If set to ``True``, the accelerometer bias provided in ``x0`` during initialization
        will remain fixed and not updated. This option is useful in situations where the
        accelerometer bias is unobservable, such as when there is insufficient aiding
        information or minimal dynamic motion, making bias estimation unreliable. Note
        that this will reduce the error-state dimension from 15 to 12, and hence also the
        error covariance matrix, **P**, from dimension (15, 15) to (12, 12). When set to
        ``False``, the P0_prior argument must have shape (15, 15).
    cold_start : bool, default True
        Whether to start the AINS filter in a 'cold' (default) or 'warm' state.
        A cold state indicates that the provided initial conditions are uncertain,
        and possibly far from the true state. Thus, to reduce the risk of divergence,
        an initial vertical alignment (i.e., roll and pitch calibration) is performed
        using accelerometer measurements and the known direction of gravity during
        the first measurement update. The IMU should remain stationary with negligible
        linear acceleration during a cold start; otherwise, divergence may occur.
        A warm start, on the other hand, assumes accurate initial conditions, and
        initializes the Kalman filter immediately without any initial roll and pitch
        calibration.
    """

    def __init__(
        self,
        fs: float,
        x0_prior: ArrayLike = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        P0_prior: ArrayLike = 1e-6*np.eye(6),
        err_gyro: dict[str, float] = ERR_GYRO_MOTION2,
        nav_frame: str = "NED",
        cold_start: bool = True,
    ) -> None:
        self._fs = fs
        self._dt = 1.0 / fs
        self._err_gyro = err_gyro
        self._cold = cold_start
        self._dq_prealloc = np.array([2.0, 0.0, 0.0, 0.0])  # Preallocation
        self._nav_frame = nav_frame.lower()

        # Strapdown algorithm / INS state
        self._ins = StrapdownAHRS(self._fs, x0_prior)

        # Gravity reference vector
        if self._nav_frame == "ned":
            self._vg_ref_n = np.array([0.0, 0.0, 1.0])
        elif self._nav_frame == "enu":
            self._vg_ref_n = np.array([0.0, 0.0, -1.0])
        else:
            raise ValueError(f"Unknown navigation frame: {self._nav_frame}")

        # Total state estimate
        self._x = self._ins.x

        # Error state estimate (after reset)
        self._dx_prealloc = np.zeros(6)  # always zero, but used in sequential update

        # Initialize Kalman filter
        self._P_prior = np.asarray_chkfinite(P0_prior).copy(order="C")
        self._P = self._P_prior.copy(order="C")

        # Prepare system matrices
        self._F = self._prep_F(err_gyro)
        self._G = self._prep_G()
        self._H = self._prep_H()
        self._W = self._prep_W(err_gyro)
        self._I = np.eye(6, order="C")

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
    def _prep_F(err_gyro: dict[str, float]) -> NDArray[np.float64]:
        """
        Prepare linearized state matrix, F.
        """

        beta_gyro = 1.0 / err_gyro["tau_cb"]

        # Temporary placeholder vectors (to be replaced each timestep)
        w_ins = np.array([0.0, 0.0, 0.0])

        S = _skew_symmetric  # alias skew symmetric matrix

        # State transition matrix
        F = np.zeros((6, 6))
        F[0:3, 0:3] = -S(w_ins)  # NB! update each time step
        F[0:3, 3:6] = -np.eye(3)
        F[3:6, 3:6] = -beta_gyro * np.eye(3)

        return F

    def _update_F(self, w_ins: NDArray[np.float64]) -> None:
        """Update linearized state transition matrix, F."""
        S = _skew_symmetric  # alias skew symmetric matrix

        # Update matrix
        self._F[0:3, 0:3] = -S(w_ins)  # NB! update each time step

    @staticmethod
    def _prep_G() -> NDArray[np.float64]:
        """Prepare (white noise) input matrix, G."""

        # Input (white noise) matrix
        G = np.zeros((6, 6))
        G[0:3, 0:3] = -np.eye(3)
        G[3:6, 3:6] = np.eye(3)
        return G

    @staticmethod
    def _prep_H() -> NDArray[np.float64]:
        """Prepare linearized measurement matrix, H. Values are placeholders only"""
        H = np.zeros((4, 6))
        return H

    def _update_H_g_ref(self, R_nm: NDArray[np.float64]) -> NDArray[np.float64]:
        """Update and return part of H matrix relevant for g_ref aiding."""
        S = _skew_symmetric
        self._H[0:3, 0:3] = S(R_nm.T @ self._vg_ref_n)
        return self._H[0:3]

    def _update_H_head(self, q_nm: NDArray[np.float64]) -> NDArray[np.float64]:
        """Update and return part of H matrix relevant for heading aiding."""
        self._H[3:4, 0:3] = _dhda_head(q_nm)
        return self._H[3:4]

    @staticmethod
    def _prep_W(err_gyro: dict[str, float]) -> NDArray[np.float64]:
        """Prepare white noise power spectral density matrix"""
        N_gyro = err_gyro["N"]
        sigma_gyro = err_gyro["B"]
        beta_gyro = 1.0 / err_gyro["tau_cb"]

        # White noise power spectral density matrix
        W = np.eye(6)
        W[0:3, 0:3] *= N_gyro**2
        W[3:6, 3:6] *= 2.0 * sigma_gyro**2 * beta_gyro
        return W

    def _reset_ins(self, dx: NDArray[np.float64]) -> None:
        """Combine states and reset INS"""
        da = dx[0:3]
        self._dq_prealloc[1:4] = da
        dq = (1.0 / np.sqrt(4.0 + da.T @ da)) * self._dq_prealloc
        self._ins._x[0:4] = _quaternion_product(self._ins._x[0:4], dq)
        self._ins._x[0:4] = _normalize(self._ins._x[0:4])
        self._ins._x[4:7] = self._ins._x[4:7] + dx[3:6]
        self._dx_prealloc[:] = np.zeros(dx.size)

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

    def _align_vertical(self, f_ins, head, head_degrees):
        """
        Vertical alignment.

        Estimate the attitude (roll and pitch) of the IMU sensor relative to the
        navigation frame using accelerometer measurements and the known direction
        of gravity. Assumes a static sensor; i.e., negligible linear acceleration.

        Parameters
        ----------
        f_ins : array-like, shape (3,)
            Bias-compensated specific force measurements (fx, fy, fz).
        head : float, optional
            Heading of measurement frame relative to navigation frame.
        head_degrees : bool, default False
            Specifies whether the heading is given in degrees or radians.
        """
        if head is None:
            head = _h_head(self.quaternion())
        else:
            if head_degrees:
                head = (np.pi / 180.0) * head

        roll, pitch = _roll_pitch_from_acc(f_ins, self._ins._nav_frame)
        self._ins._x[6:10] = _quaternion_from_euler(np.array([roll, pitch, head]))
        self._x[:] = self._ins._x

    def update(
        self,
        f_imu: ArrayLike,
        w_imu: ArrayLike,
        degrees: bool = False,
        head: float | None = None,
        head_var: float | None = None,
        head_degrees: bool = True,
        g_ref: bool = False,
        g_var: ArrayLike | None = None,
    ) -> Self:
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
            Heading measurement. I.e., the yaw angle of the 'body' frame relative to the
            assumed 'navigation' frame ('NED' or 'ENU') specified during initialization.
            If ``None``, compass aiding is not used. See ``head_degrees`` for units.
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

        # Bias compensated IMU measurements
        w_ins = w_imu - self._ins._bias_gyro

        # Initial vertical alignment (i.e., roll and pitch calibration)
        if self._cold:
            self._align_vertical(f_imu, head, head_degrees)
            self._cold = False

        # Current INS state estimates
        q_ins_nm = self._ins._q_nm
        R_ins_nm = _rot_matrix_from_quaternion(q_ins_nm)  # body-to-inertial rot matrix

        # Aliases
        dx = self._dx_prealloc  # zeros
        dt = self._dt
        F = self._F
        G = self._G
        W = self._W
        P = self._P_prior
        I_ = self._I

        # Update system matrices
        self._update_F(w_ins)

        if g_ref:
            if g_var is None:
                raise ValueError("'g_var' not provided.")
            vg_meas_m = -_normalize(f_imu)
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

        self._dx[:] = dx.ravel().copy()

        # Reset INS state
        if dx.any():
            self._reset_ins(dx.ravel())

        # Discretize system
        self._phi[:] = I_ + dt * F  # state transition matrix
        Q = dt * G @ W @ G.T  # process noise covariance matrix

        # Update current state
        self._x[:] = self._ins._x
        self._P[:] = P

        # Project ahead
        self._ins.update(w_imu, degrees=False)
        self._P_prior[:] = self._phi @ P @ self._phi.T + Q

        return self

