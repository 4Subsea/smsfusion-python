import numpy as np
import pytest

import smsfusion as sf
from smsfusion.simulate import BeatDOF, ConstantDOF, IMUSimulator, SineDOF
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


class Test_IMUSimulator:

    @pytest.fixture
    def sim(self):
        pos_x = SineDOF(1.0, 1.0)
        pos_y = SineDOF(2.0, 0.5)
        pos_z = SineDOF(3.0, 0.1)
        alpha = SineDOF(4.0, 1.0)
        beta = SineDOF(5.0, 0.5)
        gamma = SineDOF(6.0, 0.1)
        sim = IMUSimulator(pos_x, pos_y, pos_z, alpha, beta, gamma, degrees=True)
        return sim

    def test__init__default(self):
        sim = IMUSimulator()
        assert isinstance(sim._pos_x, ConstantDOF)
        assert isinstance(sim._pos_y, ConstantDOF)
        assert isinstance(sim._pos_z, ConstantDOF)
        assert isinstance(sim._alpha, ConstantDOF)
        assert isinstance(sim._beta, ConstantDOF)
        assert isinstance(sim._gamma, ConstantDOF)
        assert sim._pos_x._value == 0.0
        assert sim._pos_y._value == 0.0
        assert sim._pos_z._value == 0.0
        assert sim._alpha._value == 0.0
        assert sim._beta._value == 0.0
        assert sim._gamma._value == 0.0
        assert sim._degrees is False
        assert sim._nav_frame == "ned"
        np.testing.assert_allclose(sim._g_n, np.array([0.0, 0.0, 9.80665]))

    def test__init__float(self):
        sim = IMUSimulator(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        assert isinstance(sim._pos_x, ConstantDOF)
        assert isinstance(sim._pos_y, ConstantDOF)
        assert isinstance(sim._pos_z, ConstantDOF)
        assert isinstance(sim._alpha, ConstantDOF)
        assert isinstance(sim._beta, ConstantDOF)
        assert isinstance(sim._gamma, ConstantDOF)
        assert sim._pos_x._value == 1.0
        assert sim._pos_y._value == 2.0
        assert sim._pos_z._value == 3.0
        assert sim._alpha._value == 4.0
        assert sim._beta._value == 5.0
        assert sim._gamma._value == 6.0

    def test__init__dof(self):
        pos_x = SineDOF(1.0, 1.0)
        pos_y = ConstantDOF(2.0)
        pos_z = SineDOF(0.5, 0.5)
        alpha = ConstantDOF(10.0)
        beta = SineDOF(5.0, 2.0)
        gamma = ConstantDOF(-5.0)

        sim = IMUSimulator(
            pos_x=pos_x,
            pos_y=pos_y,
            pos_z=pos_z,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            degrees=True,
            g=9.84,
            nav_frame="ENU",
        )

        assert sim._pos_x is pos_x
        assert sim._pos_y is pos_y
        assert sim._pos_z is pos_z
        assert sim._alpha is alpha
        assert sim._beta is beta
        assert sim._gamma is gamma
        assert sim._degrees is True
        assert sim._nav_frame == "enu"
        np.testing.assert_allclose(sim._g_n, np.array([0.0, 0.0, -9.84]))

    def test__call__default(self, sim):
        fs = 10.24
        n = 100
        t, pos, vel, euler, f, w = sim(fs, n)

        np.testing.assert_allclose(t, np.arange(n) / fs)

        # Position
        assert pos.shape == (n, 3)
        np.testing.assert_allclose(pos[:, 0], sim._pos_x.y(t))
        np.testing.assert_allclose(pos[:, 1], sim._pos_y.y(t))
        np.testing.assert_allclose(pos[:, 2], sim._pos_z.y(t))

        # Velocity
        assert vel.shape == (n, 3)
        np.testing.assert_allclose(vel[:, 0], sim._pos_x.dydt(t))
        np.testing.assert_allclose(vel[:, 1], sim._pos_y.dydt(t))
        np.testing.assert_allclose(vel[:, 2], sim._pos_z.dydt(t))

        # Euler angles
        assert euler.shape == (n, 3)
        np.testing.assert_allclose(euler[:, 0], sim._alpha.y(t))
        np.testing.assert_allclose(euler[:, 1], sim._beta.y(t))
        np.testing.assert_allclose(euler[:, 2], sim._gamma.y(t))

        # Specific force
        assert f.shape == (n, 3)
        acc_x = sim._pos_x.d2ydt2(t)
        acc_y = sim._pos_y.d2ydt2(t)
        acc_z = sim._pos_z.d2ydt2(t)
        acc_expect = np.column_stack((acc_x, acc_y, acc_z))
        for f_i, euler_i, acc_i in zip(f, euler, acc_expect):
            R_nb_i = sf._transforms._rot_matrix_from_euler(np.radians(euler_i))
            f_i_expect = R_nb_i.T.dot(acc_i - sim._g_n)
            np.testing.assert_allclose(f_i, f_i_expect)

        # Angular rate
        assert w.shape == (n, 3)
        alpha, beta = np.radians(euler[:, 0:2]).T
        alpha_dot = sim._alpha.dydt(t)
        beta_dot = sim._beta.dydt(t)
        gamma_dot = sim._gamma.dydt(t)
        w_x = alpha_dot - np.sin(beta) * gamma_dot
        w_y = np.cos(alpha) * beta_dot + np.sin(alpha) * np.cos(beta) * gamma_dot
        w_z = -np.sin(alpha) * beta_dot + np.cos(alpha) * np.cos(beta) * gamma_dot
        w_b = np.column_stack([w_x, w_y, w_z])
        np.testing.assert_allclose(w, w_b)

    def test__call__degrees(self, sim):
        fs = 10.24
        n = 100
        t, pos, vel, euler, f, w = sim(fs, n, degrees=True)

        np.testing.assert_allclose(t, np.arange(n) / fs)

        # Position
        assert pos.shape == (n, 3)
        np.testing.assert_allclose(pos[:, 0], sim._pos_x.y(t))
        np.testing.assert_allclose(pos[:, 1], sim._pos_y.y(t))
        np.testing.assert_allclose(pos[:, 2], sim._pos_z.y(t))

        # Velocity
        assert vel.shape == (n, 3)
        np.testing.assert_allclose(vel[:, 0], sim._pos_x.dydt(t))
        np.testing.assert_allclose(vel[:, 1], sim._pos_y.dydt(t))
        np.testing.assert_allclose(vel[:, 2], sim._pos_z.dydt(t))

        # Euler angles
        assert euler.shape == (n, 3)
        np.testing.assert_allclose(euler[:, 0], sim._alpha.y(t))
        np.testing.assert_allclose(euler[:, 1], sim._beta.y(t))
        np.testing.assert_allclose(euler[:, 2], sim._gamma.y(t))

        # Specific force
        assert f.shape == (n, 3)
        acc_x = sim._pos_x.d2ydt2(t)
        acc_y = sim._pos_y.d2ydt2(t)
        acc_z = sim._pos_z.d2ydt2(t)
        acc_expect = np.column_stack((acc_x, acc_y, acc_z))
        for f_i, euler_i, acc_i in zip(f, euler, acc_expect):
            R_nb_i = sf._transforms._rot_matrix_from_euler(np.radians(euler_i))
            f_i_expect = R_nb_i.T.dot(acc_i - sim._g_n)
            np.testing.assert_allclose(f_i, f_i_expect)

        # Angular rate
        assert w.shape == (n, 3)
        alpha, beta = np.radians(euler[:, 0:2]).T
        alpha_dot = sim._alpha.dydt(t)
        beta_dot = sim._beta.dydt(t)
        gamma_dot = sim._gamma.dydt(t)
        w_x = alpha_dot - np.sin(beta) * gamma_dot
        w_y = np.cos(alpha) * beta_dot + np.sin(alpha) * np.cos(beta) * gamma_dot
        w_z = -np.sin(alpha) * beta_dot + np.cos(alpha) * np.cos(beta) * gamma_dot
        w_b = np.column_stack([w_x, w_y, w_z])
        np.testing.assert_allclose(w, w_b)

    def test__call__radians(self, sim):
        fs = 10.24
        n = 100
        t, pos, vel, euler, f, w = sim(fs, n, degrees=False)  # radians

        np.testing.assert_allclose(t, np.arange(n) / fs)

        # Position
        assert pos.shape == (n, 3)
        np.testing.assert_allclose(pos[:, 0], sim._pos_x.y(t))
        np.testing.assert_allclose(pos[:, 1], sim._pos_y.y(t))
        np.testing.assert_allclose(pos[:, 2], sim._pos_z.y(t))

        # Velocity
        assert vel.shape == (n, 3)
        np.testing.assert_allclose(vel[:, 0], sim._pos_x.dydt(t))
        np.testing.assert_allclose(vel[:, 1], sim._pos_y.dydt(t))
        np.testing.assert_allclose(vel[:, 2], sim._pos_z.dydt(t))

        # Euler angles
        assert euler.shape == (n, 3)
        np.testing.assert_allclose(euler[:, 0], np.radians(sim._alpha.y(t)))
        np.testing.assert_allclose(euler[:, 1], np.radians(sim._beta.y(t)))
        np.testing.assert_allclose(euler[:, 2], np.radians(sim._gamma.y(t)))

        # Specific force
        assert f.shape == (n, 3)
        acc_x = sim._pos_x.d2ydt2(t)
        acc_y = sim._pos_y.d2ydt2(t)
        acc_z = sim._pos_z.d2ydt2(t)
        acc_expect = np.column_stack((acc_x, acc_y, acc_z))
        for f_i, euler_i, acc_i in zip(f, euler, acc_expect):
            R_nb_i = sf._transforms._rot_matrix_from_euler(euler_i)
            f_i_expect = R_nb_i.T.dot(acc_i - sim._g_n)
            np.testing.assert_allclose(f_i, f_i_expect)

        # Angular rate
        assert w.shape == (n, 3)
        alpha, beta = euler[:, 0:2].T
        alpha_dot = np.radians(sim._alpha.dydt(t))
        beta_dot = np.radians(sim._beta.dydt(t))
        gamma_dot = np.radians(sim._gamma.dydt(t))
        w_x = alpha_dot - np.sin(beta) * gamma_dot
        w_y = np.cos(alpha) * beta_dot + np.sin(alpha) * np.cos(beta) * gamma_dot
        w_z = -np.sin(alpha) * beta_dot + np.cos(alpha) * np.cos(beta) * gamma_dot
        w_b = np.column_stack([w_x, w_y, w_z])
        np.testing.assert_allclose(w, w_b)


class Test_BeatDOF:
    @pytest.fixture
    def beat_dof(self):
        dof = BeatDOF(amp=1.0, freq_main=1.0, freq_beat=0.1, freq_hz=False, offset=1.0)
        return dof

    def test__init__(self):
        beat_dof = BeatDOF(
            amp=3.0,
            freq_main=2.0,
            freq_beat=0.2,
            freq_hz=True,
            phase=4.0,
            phase_degrees=True,
            offset=5.0,
        )

        assert isinstance(beat_dof, DOF)
        assert beat_dof._amp == 3.0
        assert beat_dof._w_main == pytest.approx(2.0 * np.pi * 2.0)
        assert beat_dof._w_beat == pytest.approx(2.0 * np.pi * 0.2)
        assert beat_dof._phase == pytest.approx((np.pi / 180.0) * 4.0)
        assert beat_dof._offset == 5.0

    def test__init__default(self):
        beat_dof = BeatDOF()

        assert isinstance(beat_dof, DOF)
        assert beat_dof._amp == 1.0
        assert beat_dof._w_main == pytest.approx(1.0)
        assert beat_dof._w_beat == pytest.approx(0.1)
        assert beat_dof._phase == pytest.approx(0.0)
        assert beat_dof._offset == 0.0

    def test_y(self, beat_dof, t):
        y = beat_dof.y(t)

        amp = beat_dof._amp
        w_main = beat_dof._w_main
        w_beat = beat_dof._w_beat
        phase = beat_dof._phase
        offset = beat_dof._offset
        main = np.cos(w_main * t + phase)
        beat = np.sin(w_beat / 2.0 * t)
        y_expect = amp * beat * main + offset
    
        np.testing.assert_allclose(y, y_expect)