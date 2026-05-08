import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from smsfusion._vectorops import _normalize, _skew_symmetric

from ._common import (
    _dhda_head,
    _h_head,
    _kalman_update_scalar,
    _kalman_update_sequential,
    _signed_smallest_angle,
)


@njit  # type: ignore[misc]
def _aiding_update_vel(
    dx: NDArray[np.float64],
    P: NDArray[np.float64],
    H: NDArray[np.float64],
    vel_n: NDArray[np.float64],
    vel_meas: NDArray[np.float64],
    vel_var: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Update with velocity aiding measurement.
    """

    if vel_var is None:
        raise ValueError("'vel_var' not provided.")

    dz = vel_meas - vel_n
    dx, P = _kalman_update_sequential(dx, P, dz, vel_var, H)
    return dx, P


@njit  # type: ignore[misc]
def _aiding_update_head(
    dx: NDArray[np.float64],
    P: NDArray[np.float64],
    H: NDArray[np.float64],
    q_nb: NDArray[np.float64],
    head_meas: float,
    head_var: float,
    head_degrees: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Update with heading aiding measurement.
    """

    if head_var is None:
        raise ValueError("'head_var' not provided.")

    if head_degrees:
        head_meas = (np.pi / 180.0) * head_meas
        head_var = (np.pi / 180.0) ** 2 * head_var

    dz = _signed_smallest_angle(head_meas - _h_head(q_nb))
    dx, P = _kalman_update_scalar(dx, P, dz, head_var, H)
    return dx, P


def _aiding_update_gref(
    dx: NDArray[np.float64],
    P: NDArray[np.float64],
    H: NDArray[np.float64],
    vg_b: NDArray[np.float64],
    dvel: NDArray[np.float64],
    gref_var: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Update state and covariance with gravity reference vector aiding measurement.
    """

    if gref_var is None:
        raise ValueError("gref_var is not provided; required for gref aiding.")

    dz = -_normalize(dvel) - vg_b
    dx, P = _kalman_update_sequential(dx, P, dz, gref_var, H)
    return dx, P
