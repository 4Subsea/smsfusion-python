import pytest

import numpy as np

from smsfusion._noise_models import _standard_normal


def test__standard_normal():
    x = _standard_normal(100)

    assert len(x) == 100
    assert np.mean(x) == pytest.approx(0.0, abs=0.2)  # mean value is zero
    assert np.std(x) == pytest.approx(1.0, abs=1e-1)  # std is 1
