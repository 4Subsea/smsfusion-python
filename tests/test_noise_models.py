import numpy as np
import pytest

from smsfusion._noise_models import _standard_normal


def test__standard_normal():
    x = _standard_normal(100)

    assert len(x) == 100
    assert np.mean(x) == pytest.approx(0.0, abs=0.2)  # mean value is zero
    assert np.std(x) == pytest.approx(1.0, abs=1e-1)  # std is 1


def test__standard_normal_seeds():
    x0 = _standard_normal(100, seed=0)
    x1 = _standard_normal(100, seed=0)
    x2 = _standard_normal(100, seed=1)

    np.testing.assert_array_almost_equal(x0, x1)
    assert not np.array_equal(x0, x2)
