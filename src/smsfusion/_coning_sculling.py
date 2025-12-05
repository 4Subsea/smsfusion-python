import numpy as np
from numpy.typing import ArrayLike

from smsfusion._vectorops import _cross


class ConingScullingAlg:
    """
    Coning and sculling algorithm.

    Integrates an IMU's specific force and angular rate measurements to coning and
    sculling corrected velocity (dvel) and attitude (dtheta) changes.

    For use in a strapdown algorithm as:

        vel[m+1] = vel[m] + R(q[m]) @ dvel[m] + dvel_corr
        q[m+1] = q[m] ⊗ h(dtheta[m])

    where,

        dvel_corr = [0, 0, g] (if NED)
        dvel_corr = [0, 0, -g] (if ENU)

    and,

    - dvel[m] is the sculling integral, i.e., the velocity vector change (no gravity
      correction) from time step m to m+1.
    - dtheta[m] is the coning integral, i.e., the rotation vector change from time
      step m to m+1.
    - h(dtheta[m]) is the unit quaternion representation of the rotation increment
      over the interval [m, m+1].

    The coning and sculling integrals are computed according to ref [1]_.

    References
    ----------
    .. [1] https://apps.dtic.mil/sti/tr/pdf/ADP003621.pdf
    """

    def __init__(self, fs: float, bias_gyro=None, bias_acc=None):
        self._fs = fs
        self._dt = 1.0 / fs
        self._bg = bias_gyro or np.zeros(3)
        self._ba = bias_acc or np.zeros(3)

        # Coning params
        self._beta = np.zeros(3)
        self._dbeta = np.zeros(3)
        self._dtheta_prev = np.zeros(3)
        self._w_prev = None

        # Sculling params
        self._gamma1 = np.zeros(3, dtype=float)
        self._u = np.zeros(3, dtype=float)
        self._f_prev = None

    def update(self, f_imu: ArrayLike, w_imu: ArrayLike, degrees: bool = False):
        """
        Update the coning (dtheta) and sculling (dvel) integrals using new IMU measurements.

        Parameters
        ----------
        f_imu : array-like, shape (3,)
            Specific force (acceleration + gravity) measurements [f_x, f_y, f_z],
            where f_x, f_y and f_z are specific forces along the x-, y-, and z-axis,
            respectively.
        w_imu : array-like, shape (3,)
            Angular rate measurements [w_x, w_y, w_z], where w_x, w_y and w_z are
            angular rates about the x-, y-, and z-axis, respectively.
        degrees : bool, default False
            Specify whether the angular rates are given in degrees or radians.
        """
        f_imu = np.asarray(f_imu, dtype=float)
        w_imu = np.asarray(w_imu, dtype=float)

        if degrees:
            w_imu = (np.pi / 180.0) * w_imu

        f = f_imu - self._ba
        w = w_imu - self._bg

        # Accelerometer and gyro pulse vector counts from l to l+1
        if self._f_prev is not None or self._w_prev is not None:  # first sample
            dvel = f * self._dt  # backward Euler
            dtheta = w * self._dt  # backward Euler
        else:
            dvel = 0.5 * (f + self._f_prev) * self._dt  # trapezoidal
            dtheta = 0.5 * (w + self._w_prev) * self._dt  # trapezoidal

        # Sculling update
        self._gamma1 += np.cross(self._beta + 0.5 * dtheta, dvel)
        self._u += dvel

        # Coning update
        self._dbeta += 0.5 * _cross(
            self._beta + (1.0 / 6.0) * self._dtheta_prev, dtheta
        )
        self._beta += dtheta

        self._f_prev = f
        self._w_prev = w
        self._dtheta_prev = dtheta

    def dtheta(self):
        """
        The accumulated 'body attitude change' vector. I.e., the rotation vector
        describing the total rotation over all samples since initialization (or
        last reset).
        """
        return self._beta + self._dbeta

    def dvel(self):
        """
        The accumulated specific force velocity vector change. I.e., 
        the total change in velocity (no gravity correction) over all samples since
        initialization (or last reset).
        """
        return self._u + self._gamma1

    def reset(self):
        """
        Reset the coning (dtheta) and sculling (dvel) integrals to zero.
        """
        self._beta = np.zeros(3, dtype=float)
        self._dbeta = np.zeros(3, dtype=float)
        self._gamma1 = np.zeros(3, dtype=float)
        self._u = np.zeros(3, dtype=float)
