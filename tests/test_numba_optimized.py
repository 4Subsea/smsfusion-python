import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from smsfusion import _transforms as transform
from smsfusion.numba_optimized import (
    _angular_matrix_from_quaternion,
    _cross,
    _euler_from_quaternion,
    _gamma_from_quaternion,
    _normalize,
#     _quaternion_from_rot_matrix,
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


@pytest.mark.parametrize("q", [
    np.array([0.96591925, -0.25882081, 0.0, 0.0], dtype=float),  # about x-axis
    np.array([0.96591925, 0.0, -0.25882081, 0.0], dtype=float),  # about y-axis
    np.array([0.96591925, 0.0, 0.0, -0.25882081], dtype=float),  # about z-axis
    ]
    )
def test_rot_matrix_from_quaternion(q):
    rot_matrix = _rot_matrix_from_quaternion(q)
    rot_matrix_expect = Rotation.from_quat(q[[1, 2, 3, 0]]).inv().as_matrix()
    np.testing.assert_array_almost_equal(rot_matrix, rot_matrix_expect, decimal=3)


def test__gamma_from_quaternion():
    q = np.array([0.96591925, 0.0, 0.0, -0.25882081], dtype=float)
    gamma = _gamma_from_quaternion(q)
    gamma_expect = -np.radians(30.0)
    np.testing.assert_almost_equal(gamma, gamma_expect, decimal=3)


def test__angular_matrix_from_quaternion():
    q = np.array([1.0, 2.0, 3.0, 4.0]) / np.sqrt(30)

    T_expect = np.array(
        [[-1.0, -1.5, -2.0], [0.5, -2.0, 1.5], [2.0, 0.5, -1.0], [-1.5, 1.0, 0.5]]
    ) / np.sqrt(30)

    T = _angular_matrix_from_quaternion(q)

    np.testing.assert_almost_equal(T, T_expect)



class Test__euler_from_quaternion:
    def test_pure_yaw(self):
        angle = np.radians(10.0)
        axis = np.array([0.0, 0.0, 1.0])

        q = np.array(
            [
                np.cos(angle / 2),
                np.sin(angle / 2) * axis[0],
                np.sin(angle / 2) * axis[1],
                np.sin(angle / 2) * axis[2],
            ]
        )

        alpha, beta, gamma = _euler_from_quaternion(q)
        assert alpha == 0.0
        assert beta == 0.0
        np.testing.assert_array_almost_equal(gamma, angle, decimal=16)

    def test_pure_pitch(self):
        angle = np.radians(10.0)
        axis = np.array([0.0, 1.0, 0.0])

        q = np.array(
            [
                np.cos(angle / 2),
                np.sin(angle / 2) * axis[0],
                np.sin(angle / 2) * axis[1],
                np.sin(angle / 2) * axis[2],
            ]
        )

        alpha, beta, gamma = _euler_from_quaternion(q)
        assert alpha == 0.0
        np.testing.assert_array_almost_equal(beta, angle, decimal=16)
        assert gamma == 0.0

    def test_pure_roll(self):
        angle = np.radians(10.0)
        axis = np.array([1.0, 0.0, 0.0])

        q = np.array(
            [
                np.cos(angle / 2),
                np.sin(angle / 2) * axis[0],
                np.sin(angle / 2) * axis[1],
                np.sin(angle / 2) * axis[2],
            ]
        )

        alpha, beta, gamma = _euler_from_quaternion(q)
        np.testing.assert_array_almost_equal(alpha, angle, decimal=16)
        assert beta == 0.0
        assert gamma == 0.0


class Test__rotation_matrix_from_euler:
    """
    The tests compare to sensor_4s.fusion.transform._rot_matrix_from_euler. However,
    the Numba optimized implementaiton uses from-origin-to-body (zyx) convention.
    """

    def test_pure_alpha(self):
        out = _rot_matrix_from_euler(10.0, 0.0, 0.0)
        expected = transform._rot_matrix_from_euler(np.array([10.0, 0.0, 0.0]))[0].T
        np.testing.assert_array_equal(out, expected)

    def test_pure_beta(self):
        out = _rot_matrix_from_euler(0.0, 10.0, 0.0)
        expected = transform._rot_matrix_from_euler(np.array([0.0, 10.0, 0.0]))[0].T
        np.testing.assert_array_equal(out, expected)

    def test_pure_gamma(self):
        out = _rot_matrix_from_euler(0.0, 0.0, 10.0)
        expected = transform._rot_matrix_from_euler(np.array([0.0, 0.0, 10.0]))[0].T
        np.testing.assert_array_equal(out, expected)

    def test_all(self):
        out = _rot_matrix_from_euler(10.0, -10.0, 10.0)
        expected = transform._rot_matrix_from_euler(np.array([10.0, -10.0, 10.0]))[0].T
        np.testing.assert_array_equal(out, expected)


# class Test__quaternion_from_rot_matrix:
#     def test_pure_yaw(self):
#         angle = np.radians(10.0)
#         axis = np.array([0.0, 0.0, 1.0])

#         q_expected = np.array(
#             [
#                 np.cos(angle / 2),
#                 np.sin(angle / 2) * axis[0],
#                 np.sin(angle / 2) * axis[1],
#                 np.sin(angle / 2) * axis[2],
#             ]
#         )

#         rot_matrix = Rotation.from_euler("ZYX", [10., 0., 0.], degrees=True).inv().as_matrix()
#         q = _quaternion_from_rot_matrix(rot_matrix)

#         np.testing.assert_array_almost_equal(q, q_expected, decimal=12)

#     def test_pure_pitch(self):
#         angle = np.radians(10.0)
#         axis = np.array([0.0, 1.0, 0.0])

#         q_expected = np.array(
#             [
#                 np.cos(angle / 2),
#                 np.sin(angle / 2) * axis[0],
#                 np.sin(angle / 2) * axis[1],
#                 np.sin(angle / 2) * axis[2],
#             ]
#         )

#         rot_matrix = Rotation.from_euler("ZYX", [0., 10., 0.], degrees=True).inv().as_matrix()
#         q = _quaternion_from_rot_matrix(rot_matrix)

#         np.testing.assert_array_almost_equal(q, q_expected, decimal=16)

#     def test_pure_roll(self):
#         angle = np.radians(10.0)
#         axis = np.array([1.0, 0.0, 0.0])

#         q_expected = np.array(
#             [
#                 np.cos(angle / 2),
#                 np.sin(angle / 2) * axis[0],
#                 np.sin(angle / 2) * axis[1],
#                 np.sin(angle / 2) * axis[2],
#             ]
#         )

#         rot_matrix = Rotation.from_euler("ZYX", [0., 0., 10.], degrees=True).inv().as_matrix()
#         q = _quaternion_from_rot_matrix(rot_matrix)

#         np.testing.assert_array_almost_equal(q, q_expected, decimal=16)

