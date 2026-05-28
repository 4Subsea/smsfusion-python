import numpy as np
from numpy.typing import ArrayLike

from smsfusion._vectorops import _cross


class ConingScullingAlg:
    """
    Coning and sculling algorithm.

    Integrates the specific force and angular rate measurements to coning and
    sculling corrected velocity (dvel) and attitude (dtheta) changes.

    Can be used in a strapdown algorithm as:

        vel[m+1] = vel[m] + R(q[m]) @ dvel[m+1] + dvel_corr
        q[m+1] = q[m] ⊗ dq(dtheta[m+1])

    where,

        dvel_corr = [0, 0, g * dt] (if 'NED')
        dvel_corr = [0, 0, -g * dt] (if 'ENU')

    and,

    - dvel[m+1] is the sculling integral, i.e., the velocity vector change (no gravity
      correction) from time step m to m+1.
    - dtheta[m+1] is the coning integral, i.e., the rotation vector change from time
      step m to m+1.
    - dq(dtheta[m+1]) is the unit quaternion representation of the rotation increment
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

    @property
    def _dvel_rot(self):
        return 0.5 * _cross(self._theta, self._vel)

    def _calc_dtheta_dvel(self, degrees=False):
        """
        Calculate the coning and sculling corrected dtheta and dvel.
        """
        dtheta = self._theta + self._dtheta_con
        dtheta = np.degrees(dtheta) if degrees else dtheta
        # Equation (7.2.2.2-23) in ref [2]_
        dvel = self._vel + self._dvel_rot + self._dvel_scul

        return dtheta, dvel

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
            The accumulated 'body attitude change' vector. I.e., the rotation vector
            describing the total rotation over all samples since initialization (or
            last reset).
        dvel : ndarray, shape (3,)
            The accumulated specific force velocity change vector. I.e., the total change
            in velocity (no gravity correction) over all samples since initialization
            (or last reset).
        """
        dtheta, dvel = self._calc_dtheta_dvel(degrees)

        self._theta[:] = np.zeros(3, dtype=float)
        self._dtheta_con[:] = np.zeros(3, dtype=float)
        self._dvel_scul[:] = np.zeros(3, dtype=float)
        self._vel[:] = np.zeros(3, dtype=float)
        return dtheta, dvel


class ConingScullingAlgCalibrated(ConingScullingAlg):
    """Extension of :class:`ConingScullingAlg` that applies a calibration matrix and
    bias correction to the measurements while minimizing the number of operations. See
    :class:`ConingScullingAlg` for full API and more algorithm details.

    Parameters
    ----------
    fs : float
        Sampling frequency of the measurements (Hz).
    W_w : array-like, shape (3, 3), optional
        Gyroscope calibration matrix (default: identity).
    W_f : array-like, shape (3, 3), optional
        Accelerometer calibration matrix (default: identity).
    b_w : array-like, shape (3,), optional
        Gyroscope bias vector (default: zero).
    b_f : array-like, shape (3,), optional
        Accelerometer bias vector (default: zero).
    bias_alt : bool, default False
        If set to ``True``, the bias definition of the alternative calibration model
        is returned. See Notes.

    Notes
    -----
    The calibration model is defined as::

        xyz_ref = W @ xyz + bias

    The alternative calibration model where biases are added first is defined as::

        xyz_ref = W @ (xyz + bias)

    The alternative model is enabled by setting ``bias_alt=True``.
    """

    def __init__(
        self,
        fs,
        W_w: np.ndarray = np.eye(3),
        W_f: np.ndarray = np.eye(3),
        b_w: np.ndarray = np.zeros(3),
        b_f: np.ndarray = np.zeros(3),
        bias_alt: bool = False,
    ):
        W_w_det = _determinant_3_by_3(W_w)
        W_w_inv = _inverse_3_by_3(W_w, determinant=W_w_det)
        self.cof_W = W_w_inv.T * W_w_det
        self.W_star = W_w_inv @ W_f
        if bias_alt:
            self.b_f_star = b_f
            self.b_w_star = b_w
        else:
            self.b_f_star = _inverse_3_by_3(W_f) @ b_f
            self.b_w_star = W_w_inv @ b_w
        self.W_w = W_w
        super().__init__(fs)

    def update(self, f, w, degrees=False):
        f_adjusted = self.W_star @ (f + self.b_f_star)
        w_adjusted = w + self.b_w_star
        super().update(f_adjusted, w_adjusted, degrees)

    def _calc_dtheta_dvel(self, degrees=False):
        dtheta = self.W_w @ self._theta + self.cof_W @ self._dtheta_con
        dtheta = np.degrees(dtheta) if degrees else dtheta

        dvel = self.W_w @ self._vel + self.cof_W @ (self._dvel_rot + self._dvel_scul)
        return dtheta, dvel


def _determinant_3_by_3(m: np.ndarray) -> float:
    """
    Calculates and returns the determinant of the matrix
    ```python
    m = [[a, b, c],
         [d, e, f],
         [g, h, i]]
    ```

    Parameters
    ----------
    m : array-like, shape (3, 3)
        The matrix for which to calculate the determinant.

    Returns
    -------
    det : float
        The determinant of the input matrix m.
    """
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    A = e * i - f * h
    B = -(d * i - f * g)
    C = d * h - e * g

    return a * A + b * B + c * C


def _inverse_3_by_3(m: np.ndarray, determinant: float | None = None):
    """
    Calculates and returns the inverse of the matrix
    ```python
    m = [[a, b, c],
         [d, e, f],
         [g, h, i]]
    ```

    Parameters
    ----------
    m : array-like, shape (3, 3)
        The matrix to be inverted.
    determinant : float, optional
        The determinant of the input matrix m. If not provided, it will be calculated
        internally.

    Returns
    -------
    inv_m : ndarray, shape (3, 3)
        The inverse of the input matrix m.
    """
    if m.shape != (3, 3):
        raise ValueError("Input matrix must be 3x3.")
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    A = e * i - f * h
    B = -(d * i - f * g)
    C = d * h - e * g
    D = -(b * i - c * h)
    E = a * i - c * g
    F = -(a * h - b * g)
    G = b * f - c * e
    H = -(a * f - c * d)
    I = a * e - b * d
    if determinant is None:
        determinant = _determinant_3_by_3(m)
    if determinant == 0:
        raise ValueError("Input matrix is singular and cannot be inverted.")
    return np.array([[A, D, G], [B, E, H], [C, F, I]]) / determinant
