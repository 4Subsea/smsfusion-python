import numpy as np

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
