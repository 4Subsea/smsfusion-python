import pytest

import numpy as np
import smsfusion as sf
from smsfusion import FixedIntervalSmoother


class Test_FixedIntervalSmoother:
    @pytest.fixture
    def ains(self):
        x0 = np.zeros(16)
        x0[6:10] = np.array([1.0, 0.0, 0.0, 0.0])
        ains = sf.AidedINS(10.24, x0)
        return ains
    
    def test__init__(self, ains):
        smoother = FixedIntervalSmoother(ains)
        assert smoother._ains is ains
        assert smoother._cov_smoothing is True
        assert smoother.x.size == 0
        assert smoother.P.size == 0
        assert smoother.position().size == 0
        assert smoother.velocity().size == 0
        assert smoother.quaternion().size == 0
        assert smoother.bias_acc().size == 0
        assert smoother.bias_gyro().size == 0
