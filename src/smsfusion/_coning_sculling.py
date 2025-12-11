import numpy as np
from numpy.typing import ArrayLike

from smsfusion._vectorops import _cross


class ConingScullingAlg:
    """
    Coning and sculling algorithm.

    Integrates an IMU's specific force and angular rate measurements to coning and
    sculling corrected velocity (dvel) and attitude (dtheta) changes.

    Can be used in a strapdown algorithm as:

        vel[m+1] = vel[m] + R(q[m]) @ dvel[m] + dvel_corr
        q[m+1] = q[m] ⊗ h(dtheta[m])

    where,

        dvel_corr = [0, 0, g] (if 'NED')
        dvel_corr = [0, 0, -g] (if 'ENU')

    and,

    - dvel[m] is the sculling integral, i.e., the velocity vector change (no gravity
      correction) from time step m to m+1.
    - dtheta[m] is the coning integral, i.e., the rotation vector change from time
      step m to m+1.
    - h(dtheta[m]) is the unit quaternion representation of the rotation increment
      over the interval [m, m+1].

    The coning and sculling integrals are computed according to Eq. (26) and Eq. (55)
    in ref [1]_.

    References
    ----------
    .. [1] https://apps.dtic.mil/sti/tr/pdf/ADP003621.pdf
    """

    def __init__(self, fs: float):
        self._fs = fs
        self._dt = 1.0 / fs

        # Coning params
        self._beta = np.zeros(3, dtype=float)
        self._dbeta = np.zeros(3, dtype=float)
        self._dtheta_prev = np.zeros(3, dtype=float)
        # self._w_prev = None

        # Sculling params
        self._gamma2 = np.zeros(3, dtype=float)
        # self._gamma1 = np.zeros(3, dtype=float)
        self._u = np.zeros(3, dtype=float)
        self._dvel_prev = np.zeros(3, dtype=float)
        # self._f_prev = None

    def update(self, f: ArrayLike, w: ArrayLike, degrees: bool = False):
        """
        Update the coning (dtheta) and sculling (dvel) integrals using new IMU measurements.

        Parameters
        ----------
        f : array-like, shape (3,)
            Specific force (acceleration + gravity) measurements [f_x, f_y, f_z],
            where f_x, f_y and f_z are specific forces along the x-, y-, and z-axis,
            respectively.
        w : array-like, shape (3,)
            Angular rate measurements [w_x, w_y, w_z], where w_x, w_y and w_z are
            angular rates about the x-, y-, and z-axis, respectively.
        degrees : bool, default False
            Specify whether the angular rates are given in degrees or radians.
        """
        f = np.asarray(f, dtype=float)
        w = np.asarray(w, dtype=float)

        if degrees:
            w = (np.pi / 180.0) * w

        # Accelerometer and gyro pulse vector counts from l to l+1
        # if self._f_prev is None or self._w_prev is None:  # first sample
        #     dvel = f * self._dt  # backward Euler
        #     dtheta = w * self._dt  # backward Euler
        # else:
        #     dvel = 0.5 * (f + self._f_prev) * self._dt  # trapezoidal
        #     dtheta = 0.5 * (w + self._w_prev) * self._dt  # trapezoidal

        dvel = f * self._dt  # backward Euler
        dtheta = w * self._dt  # backward Euler

        # Sculling update 1st order
        # self._gamma1 += np.cross(self._beta + 0.5 * dtheta, dvel)
        # self._u += dvel

        # # Sculling update 1st order
        # self._gamma2 += 0.5 * (np.cross(self._beta, dvel) + np.cross(self._u, dtheta))
        # self._u += dvel

        # Sculling update 2nd order
        self._gamma2 += 0.5 * (np.cross(self._beta + (1.0 / 6.0) * self._dtheta_prev, dvel) + np.cross(self._u + (1.0 / 6.0) * self._dvel_prev, dtheta))
        self._u += dvel

        # # Sculling update 2nd order
        # self._gamma1 += np.cross(self._beta + 0.5 * dtheta, dvel) + (1.0 / 12.0) * (
        #     np.cross(self._dtheta_prev, dvel) + np.cross(self._dvel_prev, dtheta)
        # )
        # self._u += dvel

        # Coning update
        self._dbeta += 0.5 * _cross(
            self._beta + (1.0 / 6.0) * self._dtheta_prev, dtheta
        )
        self._beta += dtheta

        # self._f_prev = f.copy()
        # self._w_prev = w.copy()
        self._dvel_prev = dvel.copy()
        self._dtheta_prev = dtheta.copy()

    def dtheta(self, degrees=False):
        """
        The accumulated 'body attitude change' vector. I.e., the rotation vector
        describing the total rotation over all samples since initialization (or
        last reset).

        Parameters
        ----------
        degrees : bool, default False
            Specifies whether the returned rotation vector should be in degrees
            or radians (default).
        """
        dtheta = self._beta + self._dbeta
        if degrees:
            dtheta *= 180.0 / np.pi
        return dtheta

    def dvel(self):
        """
        The accumulated specific force velocity vector change. I.e.,
        the total change in velocity (no gravity correction) over all samples since
        initialization (or last reset).
        """
        return self._u + 0.5 * np.cross(self._beta, self._u) + self._gamma2

    def reset(self):
        """
        Reset the coning (dtheta) and sculling (dvel) integrals to zero.
        """
        self._beta = np.zeros(3, dtype=float)
        self._dbeta = np.zeros(3, dtype=float)
        self._gamma1 = np.zeros(3, dtype=float)
        self._gamma2 = np.zeros(3, dtype=float)
        self._u = np.zeros(3, dtype=float)
