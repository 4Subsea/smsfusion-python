import numpy as np
from numpy.typing import ArrayLike

from ._vectorops import _cross


class ConingScullingAlgorithm:
    """
    Coning and sculling algorithm.
    """

    def __init__(self, fs: float, bias_gyro=None, bias_acc=None):
        self._fs = fs
        self._dt = 1.0 / fs
        self._bg = bias_gyro or np.zeros(3)
        self._ba = bias_acc or np.zeros(3)

        # Coning params
        self._theta = np.zeros(3)
        self._dtheta = np.zeros(3)
        self._dtheta_prev = np.zeros(3)
        self._w_prev = np.zeros(3)

        # Sculling params
        self._gamma1 = np.zeros(3, dtype=float)
        self._u = np.zeros(3, dtype=float)
        self._f_prev = np.zeros(3, dtype=float)

    def _coning_update(self, w):
        """
        Update the coning integrals using new gyroscope measurements, w[l+1].

        Coning algorithm:

            phi := beta + dbeta

        where,

            dtheta[l] = dt * (w[l+1] - w[l])
            dbeta[l+1] = dbeta[l] + 0.5 * cross((beta[l] + (1/6) * dtheta[l-1]), dtheta[l])
            beta[l+1] = beta[l] + dtheta[l]

        with initial conditions,
            beta[0] = [0, 0, 0]
            dbeta[0] = [0, 0, 0]

        Parameters
        ----------
        w : numpy.ndarray
            Bias corrected angular rate measurements.
        """
        dtheta = (w - self._w_prev) * self._dt
        self._dbeta += 0.5 * _cross(self._dbeta + (1.0 / 6.0) * self._dtheta_prev, dtheta)
        self._beta += dtheta

        self._w_prev = w
        self._dtheta_prev = dtheta

    def _sculling_update(self, f):
        """
        Update the sculling integrals using new accelerometer measurements, f[l+1].
        """
        dvel = (f - self._f_prev) * self._dt
        self._gamma1 += np.cross(self._beta + 0.5 * self._dtheta, dvel)
        self._u += dvel


    def update(self, f_imu: ArrayLike, w_imu: ArrayLike, degrees: bool = False):
        """
        Update.

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
        """
        f_imu = np.asarray(f_imu, dtype=float)
        w_imu = np.asarray(w_imu, dtype=float)

        if degrees:
            w_imu = (np.pi / 180.0) * w_imu

        f_ins = f_imu - self._ba
        w_ins = w_imu - self._bg

        self._sculling_update(f_ins)
        self._coning_update(w_ins)

    def dtheta(self):
        """
        Coning integral.
        """
        return self._beta + self._dbeta
    
    def dvel(self):
        """
        Sculling integral.
        """
        return self._u + self._gamma1
    
    def reset(self):
        self._beta = np.zeros(3, dtype=float)
        self._dbeta = np.zeros(3, dtype=float)
        self._gamma1 = np.zeros(3, dtype=float)
        self._u = np.zeros(3, dtype=float)