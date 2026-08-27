import pytest

from smsfusion import AMEKF, PVAMEKF, VAMEKF
from smsfusion._ins._smoothing import FixedIntervalSmoother


class Test_FixedIntervalSmoother:

    @pytest.mark.parametrize("mekf_class", [AMEKF, PVAMEKF, VAMEKF])
    def test_init_with_different_mekf_types(self, mekf_class):
        mekf = mekf_class(10.24)
        smoother = FixedIntervalSmoother(mekf)
        assert smoother._mekf is mekf
