import numpy as np
import pytest

from smsfusion.simulate import ConstantDOF, SineDOF, LinearRampUp
from smsfusion.simulate._simulate import DOF


@pytest.fixture
def t():
    return np.linspace(0, 10, 100)


class Test_DOF:
    @pytest.fixture
    def some_dof(self):
        class SomeDOF(DOF):

            def _y(self, t):
                return np.ones_like(t)

            def _dydt(self, t):
                return 2 * np.ones_like(t)

            def _d2ydt2(self, t):
                return 3 * np.ones_like(t)

        return SomeDOF()

    def test_y(self, some_dof, t):
        y = some_dof.y(t)
        np.testing.assert_allclose(y, np.ones(100))

    def test_dydt(self, some_dof, t):
        dydt = some_dof.dydt(t)
        np.testing.assert_allclose(dydt, 2 * np.ones(100))

    def test_d2ydt2(self, some_dof, t):
        d2ydt2 = some_dof.d2ydt2(t)
        np.testing.assert_allclose(d2ydt2, 3 * np.ones(100))

    def test__call__(self, some_dof, t):
        y, dydt, dy2dt2 = some_dof(t)
        np.testing.assert_allclose(y, np.ones(100))
        np.testing.assert_allclose(dydt, 2 * np.ones(100))
        np.testing.assert_allclose(dy2dt2, 3 * np.ones(100))


class Test_ConstantDOF:
    @pytest.fixture
    def constant_dof(self):
        return ConstantDOF(value=5.0)

    def test__init__(self):
        constant_dof = ConstantDOF(value=123.0)
        assert isinstance(constant_dof, DOF)
        assert constant_dof._value == 123.0

    def test_y(self, constant_dof, t):
        y = constant_dof.y(t)
        np.testing.assert_allclose(y, 5.0 * np.ones(100))

    def test_dydt(self, constant_dof, t):
        dydt = constant_dof.dydt(t)
        np.testing.assert_allclose(dydt, np.zeros(100))

    def test_d2ydt2(self, constant_dof, t):
        d2ydt2 = constant_dof.d2ydt2(t)
        np.testing.assert_allclose(d2ydt2, np.zeros(100))

    def test__call__(self, constant_dof, t):
        y, dydt, dy2dt2 = constant_dof(t)
        np.testing.assert_allclose(y, 5.0 * np.ones(100))
        np.testing.assert_allclose(dydt, np.zeros(100))
        np.testing.assert_allclose(dy2dt2, np.zeros(100))


class Test_SineDOF:
    @pytest.fixture
    def sine_dof(self):
        return SineDOF(2.0, 1.0)

    def test__init__(self):
        sine_dof = SineDOF(
            amp=2.0, freq=3.0, freq_hz=True, phase=4.0, phase_degrees=True, offset=5.0
        )

        assert isinstance(sine_dof, DOF)
        assert sine_dof._amp == 2.0
        assert sine_dof._w == pytest.approx(2.0 * np.pi * 3.0)
        assert sine_dof._phase == pytest.approx((np.pi / 180.0) * 4.0)
        assert sine_dof._offset == 5.0

    def test_y(self, sine_dof, t):
        y = sine_dof.y(t)
        expected_y = 2.0 * np.sin(1.0 * t + 0.0)
        np.testing.assert_allclose(y, expected_y)

    def test_dydt(self, sine_dof, t):
        dydt = sine_dof.dydt(t)
        expected_dydt = 2.0 * 1.0 * np.cos(1.0 * t + 0.0)
        np.testing.assert_allclose(dydt, expected_dydt)

    def test_d2ydt2(self, sine_dof, t):
        d2ydt2 = sine_dof.d2ydt2(t)
        expected_d2ydt2 = -2.0 * (1.0**2) * np.sin(1.0 * t + 0.0)
        np.testing.assert_allclose(d2ydt2, expected_d2ydt2)

    def test__call__(self, sine_dof, t):
        y, dydt, dy2dt2 = sine_dof(t)
        expected_y = 2.0 * np.sin(1.0 * t + 0.0)
        expected_dydt = 2.0 * 1.0 * np.cos(1.0 * t + 0.0)
        expected_d2ydt2 = -2.0 * (1.0**2) * np.sin(1.0 * t + 0.0)
        np.testing.assert_allclose(y, expected_y)
        np.testing.assert_allclose(dydt, expected_dydt)
        np.testing.assert_allclose(dy2dt2, expected_d2ydt2)

    def test_amp(self):
        sine_dof = SineDOF(amp=3.0)
        t = np.linspace(0, 2 * np.pi, 100)
        y, dydt, dy2dt2 = sine_dof(t)
        y_expect = 3.0 * np.sin(t)
        dydt_expect = 3.0 * np.cos(t)
        dy2dt2_expect = -3.0 * np.sin(t)
        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(dy2dt2, dy2dt2_expect)

    def test_freq_hz(self, t):
        sine_dof = SineDOF(freq=0.5, freq_hz=True)
        y, dydt, dy2dt2 = sine_dof(t)
        y_expect = np.sin(np.pi * t)
        dydt_expect = np.pi * np.cos(np.pi * t)
        dy2dt2_expect = -np.pi**2 * np.sin(np.pi * t)
        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(dy2dt2, dy2dt2_expect)

    def test_freq_rads(self, t):
        sine_dof = SineDOF(freq=np.pi, freq_hz=False)
        y, dydt, dy2dt2 = sine_dof(t)
        y_expect = np.sin(np.pi * t)
        dydt_expect = np.pi * np.cos(np.pi * t)
        dy2dt2_expect = -np.pi**2 * np.sin(np.pi * t)
        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(dy2dt2, dy2dt2_expect)

    def test_phase_degrees(self, t):
        sine_dof = SineDOF(phase=90.0, phase_degrees=True)
        y, dydt, dy2dt2 = sine_dof(t)
        y_expect = np.sin(t + np.pi / 2)
        dydt_expect = np.cos(t + np.pi / 2)
        dy2dt2_expect = -np.sin(t + np.pi / 2)
        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(dy2dt2, dy2dt2_expect)

    def test_phase_radians(self, t):
        sine_dof = SineDOF(phase=np.pi / 2, phase_degrees=False)
        y, dydt, dy2dt2 = sine_dof(t)
        y_expect = np.sin(t + np.pi / 2)
        dydt_expect = np.cos(t + np.pi / 2)
        dy2dt2_expect = -np.sin(t + np.pi / 2)
        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(dy2dt2, dy2dt2_expect)

    def test_offset(self, t):
        sine_dof = SineDOF(offset=2.0)
        y, dydt, dy2dt2 = sine_dof(t)
        y_expect = 2.0 + np.sin(t)
        dydt_expect = np.cos(t)
        dy2dt2_expect = -np.sin(t)
        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(dy2dt2, dy2dt2_expect)


class Test_LinearRampUp:
    @pytest.fixture
    def dof(self):
        return SineDOF(amp=1.0, freq=1.0, freq_hz=True)

    @pytest.fixture
    def dof_with_ramp(self, dof):
        return LinearRampUp(dof, t_start=1.0, ramp_length=2.0)

    def test__init__(self, dof):
        dof_ramp = LinearRampUp(dof, t_start=0.5, ramp_length=3.0)
        assert isinstance(dof_ramp, DOF)
        assert dof_ramp._dof is dof
        assert dof_ramp._t_start == 0.5
        assert dof_ramp._ramp_length == 3.0

    def test_y(self, dof_with_ramp, dof, t):
        y = dof_with_ramp.y(t)

        before_ramp = t < 1.0
        during_ramp = (t >= 1.0) & (t < 3.0)
        after_ramp = t >= 3.0

        y_dof, dydt_dof, d2ydt2_dof = dof(t)
        y_expect = np.where(
            before_ramp, 0.0, np.where(during_ramp, (t - 1.0) / 2.0 * y_dof, y_dof)
        )
        dydt_expect = np.where(
            before_ramp,
            0.0,
            np.where(
                during_ramp,
                ((t - 1.0) / 2.0) * dydt_dof,
                dydt_dof,
            ),
        )
        d2ydt2_expect = np.where(
            before_ramp,
            0.0,
            np.where(
                during_ramp,
                ((t - 1.0) / 2.0) * d2ydt2_dof,
                d2ydt2_dof,
            ),
        )
        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dof_with_ramp.dydt(t), dydt_expect)
        np.testing.assert_allclose(dof_with_ramp.d2ydt2(t), d2ydt2_expect)
