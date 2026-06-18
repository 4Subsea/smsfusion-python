import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from smsfusion._ins import _common


@pytest.mark.parametrize(
    "quaternion, dhda_expect",
    [
        (
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ),
        (
            np.array([0.89442719, 0.4472136, 0.0, 0.0]),  # gibbs -> [1.0, 0.0, 0.0]
            np.array([0.0, 10.0, 20.0]) / (4.0 + 1.0) ** 2,
        ),
        (
            np.array([0.89442719, 0.0, 0.4472136, 0.0]),  # gibbs -> [0.0, 1.0, 0.0]
            np.array([6.0, 0.0, 12.0]) / (4.0 - 1.0) ** 2,
        ),
        (
            np.array([0.89442719, 0.0, 0.0, 0.4472136]),  # gibbs -> [0.0, 0.0, 1.0]
            np.array([0.0, 0.0, 20.0]) / ((4.0 - 1.0) ** 2 * (1 + (4.0 / 3.0) ** 2)),
        ),
        (
            np.array(
                [0.92387953, 0.22094238, 0.22094238, 0.22094238]
            ),  # gibbs -> [0.47829262, 0.47829262, 0.47829262]
            np.array([0.06751864, 0.29609696, 0.87452584]),
        ),
    ],
)
def test__dhda(quaternion, dhda_expect):
    dhda_out = _common._dhda_head(quaternion)
    np.testing.assert_allclose(dhda_out, dhda_expect)


@pytest.mark.parametrize(
    "angles",
    [
        np.radians([0.0, 0.0, 35.0]),
        np.radians([25.0, 180.0, -125.0]),
        np.radians([10.0, 95.0, 1.0]),
    ],
)
def test__h_head(angles):
    alpha, beta, gamma = np.radians((0.0, 0.0, 15.0))

    quaternion = Rotation.from_euler(
        "ZYX", (gamma, beta, alpha), degrees=False
    ).as_quat()
    quaternion = np.r_[quaternion[3], quaternion[:3]]

    gamma_expect = _common._h_head(quaternion)
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
    quaternion_update = _common._update_quaternion_with_rotvec(quaternion, dtheta)

    assert np.isclose(
        np.linalg.norm(quaternion_update), 1.0
    ), f"Output quaternion is not unit norm: {quaternion_update}"

    np.testing.assert_allclose(
        quaternion_update, quaternion_update_expected, atol=1e-16
    )


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
    quaternion_update = _common._update_quaternion_with_gibbs2(quaternion.copy(), da)

    # Always assert unit norm
    assert np.isclose(
        np.linalg.norm(quaternion_update), 1.0
    ), f"Output quaternion is not unit norm: {quaternion_update}"

    np.testing.assert_allclose(
        quaternion_update, quaternion_update_expected, atol=1e-10
    )


@pytest.mark.parametrize(
    "q_nb,nav_frame_factor",
    [
        (np.array([1.0, 0.0, 0.0, 0.0]), 1.0),
        (np.array([1.0, 0.0, 0.0, 0.0]), -1.0),
        (np.array([np.cos(0.1/2), np.sin(0.1/2), 0.0, 0.0]), -1.0),
        (np.array([np.cos(0.1/2), np.sin(0.1/2), 0.0, 0.0]), 1.0),
        (np.array([np.cos(0.1/2), 0.0, np.sin(0.1/2), 0.0]), -1.0),
        (np.array([np.cos(0.1/2), 0.0, np.sin(0.1/2), 0.0]), 1.0),
        (np.array([np.cos(0.1/2), 0.0, 0.0, np.sin(0.1/2)]), -1.0),
        (np.array([np.cos(0.1/2), 0.0, 0.0, np.sin(0.1/2)]), 1.0)
    ]
)
def test__nz_b_from_quat(q_nb, nav_frame_factor):
    out = _common._nz_b_from_quat(q_nb, nav_frame_factor=nav_frame_factor)

    expect = Rotation.from_quat(q_nb, scalar_first=True).apply(nav_frame_factor * np.array([0.0, 0.0, 1.0]), inverse=True)
    np.testing.assert_allclose(out, expect)


def test__nz2vg():
    assert _common._nz2vg("NED") == 1.0
    assert _common._nz2vg("ENU") == -1.0