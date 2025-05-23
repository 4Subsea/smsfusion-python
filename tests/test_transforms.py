"""
IMPORTANT
---------

SciPy Rotation implementation is used as reference in tests. However, SciPy
operates with active rotations, whereas passive rotations are considered here. Keep in
mind that passive rotations is simply the inverse active rotations and vice versa.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from smsfusion import _transforms


@pytest.mark.parametrize(
    "euler, expected",
    [
        (
            np.radians((30.0, 0.0, 0.0)),
            np.array([[1.0, 0.0, -0.0], [0.0, 0.8660254, -0.5], [0.0, 0.5, 0.8660254]]),
        ),  # pure roll
        (
            np.radians((0.0, 30.0, 0.0)),
            np.array(
                [[1.0, 0.0, 0.57735027], [0.0, 1.0, -0.0], [0.0, 0.0, 1.15470054]]
            ),
        ),  # pure pitch
        (np.radians((0.0, 0.0, 30.0)), np.eye(3)),  # pure yaw
        (
            np.radians((30.0, 15.0, 20.0)),
            np.array(
                [
                    [1.0, 0.1339746, 0.23205081],
                    [0.0, 0.8660254, -0.5],
                    [0.0, 0.51763809, 0.89657547],
                ]
            ),
        ),  # mixed case 1
        (
            np.radians((15.0, 45.0, 5.0)),
            np.array(
                [
                    [1.0, 0.25881905, 0.96592583],
                    [0.0, 0.96592583, -0.25881905],
                    [0.0, 0.3660254, 1.3660254],
                ]
            ),
        ),  # mixed case 2
    ],
    ids=["roll", "pitch", "yaw", "mix 1", "mix 2"],
)
def test_angular_matrix_from_euler(euler, expected):
    output = _transforms._angular_matrix_from_euler(euler)
    np.testing.assert_array_almost_equal(output, expected)


@pytest.mark.parametrize(
    "euler, expected",
    [
        (
            np.radians((30.0, 0.0, 0.0)),
            np.array([[1.0, 0.0, -0.0], [0.0, 0.8660254, 0.5], [0.0, -0.5, 0.8660254]]),
        ),  # pure roll
        (
            np.radians((0.0, 30.0, 0.0)),
            np.array([[1.0, 0.0, -0.5], [0.0, 1.0, 0.0], [0.0, -0.0, 0.8660254]]),
        ),  # pure pitch
        (np.radians((0.0, 0.0, 30.0)), np.eye(3)),  # pure yaw
        (
            np.radians((30.0, 15.0, 20.0)),
            np.array(
                [
                    [1.0, 0.0, -0.25881905],
                    [0.0, 0.8660254, 0.48296291],
                    [0.0, -0.5, 0.8365163],
                ]
            ),
        ),  # mixed case 1
        (
            np.radians((15.0, 45.0, 5.0)),
            np.array(
                [
                    [1.0, 0.0, -0.70710678],
                    [0.0, 0.96592583, 0.1830127],
                    [0.0, -0.25881905, 0.6830127],
                ]
            ),
        ),  # mixed case 2
    ],
    ids=["roll", "pitch", "yaw", "mix 1", "mix 2"],
)
def test_inv_angular_matrix_from_euler(euler, expected):
    output = _transforms._inv_angular_matrix_from_euler(euler)
    np.testing.assert_array_almost_equal(output, expected)


@pytest.mark.parametrize(
    "q",
    [
        np.array([0.96591925, -0.25882081, 0.0, 0.0], dtype=float),  # about x-axis
        np.array([0.96591925, 0.0, -0.25882081, 0.0], dtype=float),  # about y-axis
        np.array([0.96591925, 0.0, 0.0, -0.25882081], dtype=float),  # about z-axis
    ],
)
def test_rot_matrix_from_quaternion(q):
    rot_matrix = _transforms._rot_matrix_from_quaternion(q)
    rot_matrix_expect = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    np.testing.assert_array_almost_equal(rot_matrix, rot_matrix_expect, decimal=3)


@pytest.mark.parametrize(
    "q",
    [
        np.array([0.96591925, 0.0, 0.0, -0.25882081], dtype=float),  # pure about z-axis
        np.array([0.93511153, 0.0, 0.25056579, -0.25056579], dtype=float),  # mixed
    ],
)
def test__gamma_from_quaternion(q):
    gamma = _transforms._gamma_from_quaternion(q)
    gamma_expect = Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("ZYX", degrees=False)[0]
    np.testing.assert_almost_equal(gamma, gamma_expect, decimal=3)


def test__angular_matrix_from_quaternion():
    q = np.array([1.0, 2.0, 3.0, 4.0]) / np.sqrt(30.0)

    T_expect = (
        np.array(
            [
                [-q[1], -q[2], -q[3]],
                [q[0], -q[3], q[2]],
                [q[3], q[0], -q[1]],
                [-q[2], q[1], q[0]],
            ]
        )
        / 2.0
    )

    T = _transforms._angular_matrix_from_quaternion(q)

    np.testing.assert_almost_equal(T, T_expect)


@pytest.mark.parametrize(
    "angle, axis, euler",
    [
        (
            np.radians(10.0),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, np.radians(10.0)]),
        ),  # pure yaw
        (
            np.radians(10.0),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, np.radians(10.0), 0.0]),
        ),  # pure pitch
        (
            np.radians(10.0),
            np.array([1.0, 0.0, 0.0]),
            np.array([np.radians(10.0), 0.0, 0.0]),
        ),  # pure roll
        (
            np.radians(10.0),
            np.array([1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]),
            np.array([0.1059987325729154, 0.0953360919950474, 0.1059987325729154]),
        ),  # mixed
    ],
)
def test__euler_from_quaternion(angle, axis, euler):
    q = np.array(
        [
            np.cos(angle / 2),
            np.sin(angle / 2) * axis[0],
            np.sin(angle / 2) * axis[1],
            np.sin(angle / 2) * axis[2],
        ]
    )

    alpha_beta_gamma = _transforms._euler_from_quaternion(q)
    np.testing.assert_array_almost_equal(alpha_beta_gamma, euler, decimal=16)


@pytest.mark.parametrize(
    "euler",
    [
        np.array([10.0, 0.0, 0.0]),  # pure roll
        np.array([0.0, 10.0, 0.0]),  # pure pitch
        np.array([0.0, 0.0, 10.0]),  # pure yaw
        np.array([10.0, -10.0, 10.0]),  # mixed
    ],
)
def test__rot_matrix_from_euler(euler):
    """
    The Numba optimized implementaiton uses from-origin-to-body (zyx) convention,
    where also the resulting rotation matrix is from-origin-to-body.
    """
    out = _transforms._rot_matrix_from_euler(euler)
    expected = Rotation.from_euler("ZYX", euler[::-1]).as_matrix()
    np.testing.assert_array_almost_equal(out, expected)


def test__quaternion_from_euler():
    # As rads
    euler = np.random.random(3) * np.pi  # passive, intrinsic rotations

    q_out = _transforms._quaternion_from_euler(euler)

    q_expect = Rotation.from_euler("ZYX", euler[::-1]).as_quat()
    q_expect = np.r_[q_expect[3], q_expect[:3]]

    np.testing.assert_array_almost_equal(q_out, q_expect)


class Test_quaternion_from_euler:
    def test_simple(self):
        # As rads
        euler = np.random.random(3) * np.pi  # passive, intrinsic rotations

        q_out = _transforms.quaternion_from_euler(euler, degrees=False)

        q_expect = Rotation.from_euler("ZYX", euler[::-1]).as_quat()
        q_expect = np.r_[q_expect[3], q_expect[:3]]

        np.testing.assert_array_almost_equal(q_out, q_expect)

        # As degrees
        euler = np.random.random(3) * 180.0  # passive, intrinsic rotations

        q_out = _transforms.quaternion_from_euler(euler, degrees=True)

        q_expect = Rotation.from_euler("ZYX", euler[::-1], degrees=True).as_quat()
        q_expect = np.r_[q_expect[3], q_expect[:3]]

        np.testing.assert_array_almost_equal(q_out, q_expect)

    @pytest.mark.parametrize(
        "euler", [np.array([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0], (0.0, 0.0, 0.0)]
    )
    def test_arraylike(self, euler):
        # As rads
        # euler = [0.0, 1.0, 0.0]  # passive, intrinsic rotations

        q_out = _transforms.quaternion_from_euler(euler, degrees=False)
        q_expect = np.array([1.0, 0.0, 0.0, 0.0])

        np.testing.assert_array_almost_equal(q_out, q_expect)

    @pytest.mark.parametrize(
        "euler",
        [[0, 0, 0], np.array([0.0, 0.0, 0.0]), np.array([0, 0, 0]), (0, 0.0, 0)],
    )
    def test_dtype(self, euler):
        q_out = _transforms.quaternion_from_euler(euler, degrees=False)
        q_expect = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(q_out, q_expect)


def test__quaternion_from_euler():
    # As rads
    euler = np.random.random(3) * np.pi  # passive, intrinsic rotations

    q_out = _transforms._quaternion_from_euler(euler)

    q_expect = Rotation.from_euler("ZYX", euler[::-1]).as_quat()
    q_expect = np.r_[q_expect[3], q_expect[:3]]

    np.testing.assert_array_almost_equal(q_out, q_expect)


def test_euler2quaternion2euler_transform():
    euler_in = np.random.random(3) * np.pi / 2  # passive, intrinsic rotations

    euler_out = _transforms._euler_from_quaternion(
        _transforms.quaternion_from_euler(euler_in)
    )

    np.testing.assert_array_almost_equal(euler_out, euler_in)


def test_euler2quaternion2gamma_transform():
    euler_in = np.random.random(3) * np.pi / 2  # passive, intrinsic rotations

    gamma_out = _transforms._gamma_from_quaternion(
        _transforms.quaternion_from_euler(euler_in)
    )

    np.testing.assert_array_almost_equal(gamma_out, euler_in[2])
