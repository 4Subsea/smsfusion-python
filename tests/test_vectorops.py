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

from smsfusion import _vectorops


def test__normalize():
    a = np.array([1.0, 0.0, 0.0, 1.0])
    out = _vectorops._normalize(a)
    expected = np.array([1.0 / np.sqrt(2.0), 0.0, 0.0, 1.0 / np.sqrt(2.0)])
    np.testing.assert_array_equal(out, expected)


def test__cross():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    out = _vectorops._cross(a, b)
    expected = np.array([0.0, 0.0, 1.0])
    np.testing.assert_array_equal(out, expected)

    a = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0])
    b = np.array([0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])
    out = _vectorops._cross(a, b)

    expected = np.cross(a, b)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    "euler_a, euler_b", np.random.uniform(0., 360., size=(10, 2, 3)).tolist()
)
def test___quaternion_product(euler_a, euler_b):
    rot_a = Rotation.from_euler("ZYX", euler_a, degrees=True).inv()
    rot_b = Rotation.from_euler("ZYX", euler_b, degrees=True).inv()

    rot_ab = rot_a * rot_b

    q_a = rot_a.as_quat()
    q_a = np.r_[q_a[3], q_a[:3]]

    q_b = rot_b.as_quat()
    q_b = np.r_[q_b[3], q_b[:3]]

    q_ab = rot_ab.as_quat()
    q_ab = np.r_[q_ab[3], q_ab[:3]]

    q_ab_out = _vectorops._quaternion_product(q_a, q_b)
    np.testing.assert_array_almost_equal(q_ab, q_ab_out)

