import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from smsfusion import quaternion_from_euler
from smsfusion._ins import _common

YAW_GRADIENT_EULER_ANGLES = [  # (roll, pitch, yaw) in degrees
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 90.0),
    (0.0, 0.0, -90.0),
    (0.0, 0.0, 180.0),
    (30.0, 0.0, 0.0),
    (0.0, 30.0, 0.0),
    (10.0, -20.0, 135.0),
    (-45.0, 60.0, -150.0),
    (170.0, -80.0, 45.0),
]


@pytest.mark.parametrize("roll, pitch, yaw", YAW_GRADIENT_EULER_ANGLES)
def test__yaw_gradient(roll, pitch, yaw):
    """
    The yaw gradient wrt a body-frame attitude error is given by the yaw row of the
    ZYX Euler angle kinematics: [0, sin(roll), cos(roll)] / cos(pitch).
    """
    q = quaternion_from_euler([roll, pitch, yaw], degrees=True)
    dhda_out = _common._yaw_gradient(q)

    roll, pitch = np.radians((roll, pitch))
    dhda_expect = np.array([0.0, np.sin(roll), np.cos(roll)]) / np.cos(pitch)
    np.testing.assert_allclose(dhda_out, dhda_expect, atol=1e-12)


@pytest.mark.parametrize("roll, pitch, yaw", YAW_GRADIENT_EULER_ANGLES)
def test__yaw_gradient_finite_difference(roll, pitch, yaw):
    """
    The yaw gradient must be consistent with how the attitude error is applied in
    the MEKF reset step (i.e., ``_update_quaternion_with_gibbs2``).
    """
    q = quaternion_from_euler([roll, pitch, yaw], degrees=True)
    dhda_out = _common._yaw_gradient(q)

    h = 1e-6
    dhda_expect = np.zeros(3)
    for i in range(3):
        da = np.zeros(3)
        da[i] = h
        q_plus, q_minus = q.copy(), q.copy()
        _common._update_quaternion_with_gibbs2(q_plus, da)
        _common._update_quaternion_with_gibbs2(q_minus, -da)
        dyaw = _common._yaw_from_quaternion(q_plus) - _common._yaw_from_quaternion(
            q_minus
        )
        dhda_expect[i] = _common._signed_smallest_angle(dyaw) / (2.0 * h)

    np.testing.assert_allclose(dhda_out, dhda_expect, atol=1e-6)


@pytest.mark.parametrize(
    "angles",
    [
        np.radians([0.0, 0.0, 35.0]),
        np.radians([25.0, 180.0, -125.0]),
        np.radians([10.0, 95.0, 1.0]),
    ],
)
def test__yaw_from_quaternion(angles):
    alpha, beta, gamma = np.radians((0.0, 0.0, 15.0))

    quaternion = Rotation.from_euler(
        "ZYX", (gamma, beta, alpha), degrees=False
    ).as_quat()
    quaternion = np.r_[quaternion[3], quaternion[:3]]

    gamma_expect = _common._yaw_from_quaternion(quaternion)
    assert gamma_expect == pytest.approx(gamma)


@pytest.mark.parametrize(
    "angle, degrees, angle_expect",
    [
        (0.0, True, 0.0),
        (-180.0, True, -180.0),
        (180.0, True, -180.0),
        (-np.pi, False, -np.pi),
        (np.pi, False, -np.pi),
        (90.0, True, 90.0),
        (-90.0, True, -90.0),
        (181, True, -179.0),
        (-181, True, 179.0),
    ],
)
def test__signed_smallest_angle(angle, degrees, angle_expect):
    assert _common._signed_smallest_angle(angle, degrees=degrees) == pytest.approx(
        angle_expect
    )


@pytest.mark.parametrize(
    "quaternion, dtheta, quaternion_update_expected",
    [
        # Identity quaternion, zero rotation → unchanged
        (
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        ),
        # Identity quaternion, 90° rotation around X-axis
        (
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([np.pi / 2, 0.0, 0.0]),
            np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]),
        ),
        # Identity quaternion, 90° rotation around Y-axis
        (
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, np.pi / 2, 0.0]),
            np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]),
        ),
        # Identity quaternion, 90° rotation around Z-axis
        (
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, np.pi / 2]),
            np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]),
        ),
        # Identity quaternion, 180° rotation around X-axis
        (
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([np.pi, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
        ),
        # Non-identity quaternion (90° around Z), zero rotation → unchanged
        (
            np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]),
            np.array([0.0, 0.0, 0.0]),
            np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]),
        ),
    ],
)
def test__update_quaternion_with_rotvec(quaternion, dtheta, quaternion_update_expected):
    _common._update_quaternion_with_rotvec(quaternion, dtheta)

    assert np.isclose(
        np.linalg.norm(quaternion), 1.0
    ), f"Output quaternion is not unit norm: {quaternion}"

    np.testing.assert_allclose(quaternion, quaternion_update_expected, atol=1e-16)


@pytest.mark.parametrize(
    ("quaternion", "da", "quaternion_update_expected"),
    [
        # Identity quaternion, no correction
        (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        ),
        # Small x-axis correction
        (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.1, 0.0, 0.0], dtype=np.float64),
            np.array(
                [
                    0.9987523388778446,
                    0.04993761694389223,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            ),
        ),
        # Small y-axis correction
        (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.1, 0.0], dtype=np.float64),
            np.array(
                [
                    0.9987523388778446,
                    0.0,
                    0.04993761694389223,
                    0.0,
                ],
                dtype=np.float64,
            ),
        ),
        # Small z-axis correction
        (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.1], dtype=np.float64),
            np.array(
                [
                    0.9987523388778446,
                    0.0,
                    0.0,
                    0.04993761694389223,
                ],
                dtype=np.float64,
            ),
        ),
    ],
)
def test__update_quaternion_with_gibbs2(quaternion, da, quaternion_update_expected):
    _common._update_quaternion_with_gibbs2(quaternion, da)

    # Always assert unit norm
    assert np.isclose(
        np.linalg.norm(quaternion), 1.0
    ), f"Output quaternion is not unit norm: {quaternion}"

    np.testing.assert_allclose(quaternion, quaternion_update_expected, atol=1e-10)


@pytest.mark.parametrize(
    "q_nb,nav_frame_factor",
    [
        (np.array([1.0, 0.0, 0.0, 0.0]), 1.0),
        (np.array([1.0, 0.0, 0.0, 0.0]), -1.0),
        (np.array([np.cos(0.1 / 2), np.sin(0.1 / 2), 0.0, 0.0]), -1.0),
        (np.array([np.cos(0.1 / 2), np.sin(0.1 / 2), 0.0, 0.0]), 1.0),
        (np.array([np.cos(0.1 / 2), 0.0, np.sin(0.1 / 2), 0.0]), -1.0),
        (np.array([np.cos(0.1 / 2), 0.0, np.sin(0.1 / 2), 0.0]), 1.0),
        (np.array([np.cos(0.1 / 2), 0.0, 0.0, np.sin(0.1 / 2)]), -1.0),
        (np.array([np.cos(0.1 / 2), 0.0, 0.0, np.sin(0.1 / 2)]), 1.0),
    ],
)
def test__gref_b_from_quat(q_nb, nav_frame_factor):
    out = _common._gref_b_from_quat(q_nb, nav_frame_factor=nav_frame_factor)

    expect = Rotation.from_quat(q_nb, scalar_first=True).apply(
        nav_frame_factor * np.array([0.0, 0.0, 1.0]), inverse=True
    )
    np.testing.assert_allclose(out, expect)


def test__nz2vg():
    assert _common._nz2vg("NED") == 1.0
    assert _common._nz2vg("ENU") == -1.0


def test__gravity_nav():
    g = 9.81
    assert np.allclose(_common._gravity_nav(g, "NED"), np.array([0.0, 0.0, g]))
    assert np.allclose(_common._gravity_nav(g, "ENU"), np.array([0.0, 0.0, -g]))


def test_kalman_update_sequential():

    rng = np.random.default_rng(42)

    m = 10  # number of measurements
    n = 12  # state dimension

    x = rng.random(n)
    A = rng.random((n, n))
    P = A @ A.T + np.eye(n)  # positive semi-definite
    H = rng.random((m, n))
    var = rng.random(m)
    z = rng.random(m)

    x_upd = x.copy()
    P_upd = P.copy()
    _common._kalman_update_sequential(x_upd, P_upd, z, var, H)

    R = np.diag(var)
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x_expect = x + K @ (z - H @ x)
    P_expect = (np.eye(n) - K @ H) @ P @ (np.eye(n) - K @ H).T + K @ R @ K.T

    np.testing.assert_allclose(x_upd, x_expect)
    np.testing.assert_allclose(P_upd, P_expect)


def test_kalman_update_scalar():

    rng = np.random.default_rng(42)

    n = 9  # state dimension

    x = rng.random(n)
    A = rng.random((n, n))
    P = A @ A.T + np.eye(n)  # positive semi-definite
    h = rng.random(n)
    r = rng.random()
    z = rng.random()

    x_upd = x.copy()
    P_upd = P.copy()
    _common._kalman_update_scalar(x_upd, P_upd, z, r, h)

    R = np.array([[r]])
    H = h.reshape(1, n)
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x_expect = x + K @ (z - H @ x)
    P_expect = (np.eye(n) - K @ H) @ P @ (np.eye(n) - K @ H).T + K @ R @ K.T

    np.testing.assert_allclose(x_upd, x_expect)
    np.testing.assert_allclose(P_upd, P_expect)


def test_project_covariance_ahead():

    rng = np.random.default_rng(42)

    n = 6  # state dimension

    A = rng.random((n, n))
    P = A @ A.T + np.eye(n)  # positive semi-definite
    phi = rng.random((n, n))
    A = rng.random((n, n))
    Q = A @ A.T + np.eye(n)  # positive semi-definite

    P_proj = P.copy()
    _common._project_covariance_ahead(P_proj, phi, Q)

    P_expect = phi @ P @ phi.T + Q

    np.testing.assert_allclose(P_proj, P_expect)
