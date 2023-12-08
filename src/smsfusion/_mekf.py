import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._ins import StrapdownINS
from ._transforms import _euler_from_quaternion


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

    def __init__(
        self,
        fs: float,
        x0: ArrayLike,
        err_acc: dict[str, float],
        err_gyro: dict[str, float],
        var_pos: ArrayLike,
        var_compass: float,
    ) -> None:
        self._fs = fs
        self._dt = 1.0 / fs
        self._err_acc = err_acc
        self._err_gyro = err_gyro
        self._x0 = np.asarray_chkfinite(x0).reshape(16, 1).copy()
        var_pos = np.asarray_chkfinite(var_pos).reshape(3).copy()

        # Strapdown algorithm
        self._x_ins = self._x0
        self._ins = StrapdownINS(self._x_ins[0:10])

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
