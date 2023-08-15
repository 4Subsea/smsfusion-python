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

from smsfusion.numba_optimized import (
    _angular_matrix_from_quaternion,
    _cross,
    _euler_from_quaternion,
    _gamma_from_quaternion,
    _normalize,
    _rot_matrix_from_euler,
    _rot_matrix_from_quaternion,
)


def test__normalize():
    a = np.array([1.0, 0.0, 0.0, 1.0])
    out = _normalize(a)
    expected = np.array([1.0 / np.sqrt(2.0), 0.0, 0.0, 1.0 / np.sqrt(2.0)])
    np.testing.assert_array_equal(out, expected)


def test__cross():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    out = _cross(a, b)
    expected = np.array([0.0, 0.0, 1.0])
    np.testing.assert_array_equal(out, expected)

    a = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0])
    b = np.array([0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])
    out = _cross(a, b)

    expected = np.cross(a, b)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    "q",
    [
        np.array([0.96591925, -0.25882081, 0.0, 0.0], dtype=float),  # about x-axis
        np.array([0.96591925, 0.0, -0.25882081, 0.0], dtype=float),  # about y-axis
        np.array([0.96591925, 0.0, 0.0, -0.25882081], dtype=float),  # about z-axis
    ],
)
def test_rot_matrix_from_quaternion(q):
    rot_matrix = _rot_matrix_from_quaternion(q)
    rot_matrix_expect = Rotation.from_quat(q[[1, 2, 3, 0]]).inv().as_matrix()
    np.testing.assert_array_almost_equal(rot_matrix, rot_matrix_expect, decimal=3)


@pytest.mark.parametrize(
    "q",
    [
        np.array([0.96591925, 0.0, 0.0, -0.25882081], dtype=float),  # pure about z-axis
        np.array([0.93511153, 0.0, 0.25056579, -0.25056579], dtype=float),  # mixed
    ],
)
def test__gamma_from_quaternion(q):
    gamma = _gamma_from_quaternion(q)
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

    T = _angular_matrix_from_quaternion(q)

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

    alpha_beta_gamma = _euler_from_quaternion(q)
    np.testing.assert_array_almost_equal(alpha_beta_gamma, euler, decimal=16)


@pytest.mark.parametrize("euler", [
    np.array([10., 0.0, 0.0]),  # pure roll
    np.array([0.0, 10.0, 0.0]),  # pure pitch
    np.array([0.0, 0.0, 10.0]),  # pure yaw
    np.array([10.0, -10.0, 10.0]),  # mixed
])
def test__rot_matrix_from_euler(euler):
    """
    The Numba optimized implementaiton uses from-origin-to-body (zyx) convention,
    where also the resulting rotation matrix is from-origin-to-body.
    """
    out = _rot_matrix_from_euler(*euler)
    expected = Rotation.from_euler("ZYX", euler[::-1]).inv().as_matrix()
    np.testing.assert_array_almost_equal(out, expected)
