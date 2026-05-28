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
    "euler_a, euler_b", np.random.uniform(0.0, 360.0, size=(10, 2, 3)).tolist()
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


def test__skew_symmetric():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    out = _vectorops._skew_symmetric(a) @ b
    expected = np.array([0.0, 0.0, 1.0])
    np.testing.assert_array_equal(out, expected)

    a = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0])
    b = np.array([0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])
    out = _vectorops._skew_symmetric(a) @ b
    expected = np.cross(a, b)
    np.testing.assert_array_equal(out, expected)


def test__inverse_and_determinant_3_by_3():
    """Test matrix inversion and determinant against numpy"""
    rng = np.random.default_rng(42)
    for _ in range(5):
        m = rng.random((3, 3))
        det = _vectorops._determinant_3_by_3(m)
        inv = _vectorops._inverse_3_by_3(m)
        inv_with_det = _vectorops._inverse_3_by_3(m, determinant=det)
        det_expected = np.linalg.det(m)
        inv_expected = np.linalg.inv(m)
        np.testing.assert_allclose(det, det_expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(inv, inv_expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(inv_with_det, inv_expected, rtol=1e-12, atol=1e-12)


def test__inverse_3_by_3_singular():
    """Test that the inverse function raises an error for singular matrices"""
    singular_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    with pytest.raises(ValueError):
        _vectorops._inverse_3_by_3(singular_matrix)
