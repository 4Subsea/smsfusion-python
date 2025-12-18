import numpy as np
from numpy.typing import ArrayLike

from smsfusion._vectorops import _cross


class ConingScullingAlg:
    """
    Coning and sculling algorithm.

    Integrates the specific force and angular rate measurements to coning and
    sculling corrected velocity (dvel) and attitude (dtheta) changes.

    Can be used in a strapdown algorithm as:

        vel[m+1] = vel[m] + R(q[m]) @ dvel[m] + dvel_corr
        q[m+1] = q[m] ⊗ dq(dtheta[m])

    where,

        dvel_corr = [0, 0, g * dt] (if 'NED')
        dvel_corr = [0, 0, -g * dt] (if 'ENU')

    and,

    - dvel[m] is the sculling integral, i.e., the velocity vector change (no gravity
      correction) from time step m to m+1.
    - dtheta[m] is the coning integral, i.e., the rotation vector change from time
      step m to m+1.
    - dq(dtheta[m]) is the unit quaternion representation of the rotation increment
      over the interval [m, m+1].
    - R(q[m]) is the rotation matrix (body-to-nav) corresponding to the attitude
      quaternion q[m].

    Here, ⊗ denotes quaternion multiplication (Hamilton product).

    The coning and sculling integrals are computed using a 2nd order algorithm as
    described in [1]_ and [2]_.

    References
    ----------
    .. [1] Savage, Paul G., Strapdown System Algorithms, AD-P003 621, https://apps.dtic.mil/sti/tr/pdf/ADP003621.pdf
    .. [2] Savage, Paul G., Strapdown Analytics, 2nd Edition, Part 1, 2007, https://strapdownassociates.com/Strapdown%20Analytics%20II%20Part%201.pdf
    """

    def __init__(self, fs: float):
        self._fs = fs
        self._dt = 1.0 / fs

        # Coning params
        self._theta = np.zeros(3, dtype=float)
        self._dtheta_con = np.zeros(3, dtype=float)
        self._dtheta_prev = np.zeros(3, dtype=float)

        # Sculling params
        self._vel = np.zeros(3, dtype=float)
        self._dvel_scul = np.zeros(3, dtype=float)
        self._dv_prev = np.zeros(3, dtype=float)

    def update(self, f: ArrayLike, w: ArrayLike, degrees: bool = False):
        """
        Update the coning (dtheta) and sculling (dvel) integrals using new measurements.

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
            Specifies whether the angular rates are given in degrees or radians (default).
        """
        f = np.asarray(f, dtype=float)
        w = np.asarray(w, dtype=float)

        if degrees:
            w = (np.pi / 180.0) * w

        # View for readability
        theta = self._theta
        dtheta_con = self._dtheta_con
        dtheta_prev = self._dtheta_prev
        vel = self._vel
        dvel_scul = self._dvel_scul
        dv_prev = self._dv_prev

        dv = f * self._dt  # backward Euler
        dtheta = w * self._dt  # backward Euler

        # Sculling update (2nd order)
        # See Eq. (7.2.2.2.2-15) in ref [2]_ and Eq. (56) in ref [1]_
        dvel_scul += 0.5 * (
            _cross(theta + (1.0 / 6.0) * dtheta_prev, dv)
            + _cross(vel + (1.0 / 6.0) * dv_prev, dtheta)
        )
        vel += dv

        # Coning update
        # See Eq. (26) in ref [1]_
        dtheta_con += 0.5 * _cross(theta + (1.0 / 6.0) * dtheta_prev, dtheta)
        theta += dtheta

        dv_prev[:] = dv
        dtheta_prev[:] = dtheta

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
        dtheta = self._theta + self._dtheta_con
        return np.degrees(dtheta) if degrees else dtheta

    @property
    def _dvel_rot(self):
        return 0.5 * _cross(self._theta, self._vel)

    def dvel(self):
        """
        The accumulated specific force velocity change vector. I.e.,
        the total change in velocity (no gravity correction) over all samples since
        initialization (or last reset).
        """
        return self._vel + self._dvel_rot + self._dvel_scul

    def flush(self, degrees=False):
        """
        Return dtheta (the accumulated 'body attitude change' vector) and
        dvel (the accumulated specific force velocity change vector), and reset
        the coning (dtheta) and sculling (dvel) integrals to zero.

        Parameters
        ----------
        degrees : bool, default False
            Specifies whether the returned rotation vector should be in degrees
            or radians (default).

        Returns
        -------
        dtheta : ndarray, shape (3,)
            The accumulated 'body attitude change' vector, see :meth:`dtheta`.
        dvel : ndarray, shape (3,)
            The accumulated specific force velocity change vector, see :meth:`dvel`.
        """
        dtheta = self.dtheta(degrees=degrees)
        dvel = self.dvel()

        self._theta[:] = np.zeros(3, dtype=float)
        self._dtheta_con[:] = np.zeros(3, dtype=float)
        self._dvel_scul[:] = np.zeros(3, dtype=float)
        self._vel[:] = np.zeros(3, dtype=float)
        return dtheta, dvel