import numpy as np
from numpy.typing import ArrayLike

from ._ins import StrapdownINS


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
