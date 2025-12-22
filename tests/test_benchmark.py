import numpy as np
import pytest

from smsfusion import benchmark


class Test_BeatSignal:
    def test_init(self):
        beat_signal = benchmark.BeatSignal(0.1, 0.01)
        assert beat_signal._f_main == 0.1 * 2.0 * np.pi
        assert beat_signal._f_beat == 0.01 * 2.0 * np.pi

    def test_init_rads(self):
        beat_signal = benchmark.BeatSignal(0.1, 0.01, freq_hz=False)
        assert beat_signal._f_main == 0.1
        assert beat_signal._f_beat == 0.01

    def test_signal(self):
        beat_signal = benchmark.BeatSignal(0.1, 0.01, freq_hz=True)

        fs = 10.0
        n = 2000
        t = np.linspace(0.0, n / fs, n, endpoint=False)
        y, dydt, d2ydt2 = beat_signal(t)

        assert len(t) == int(n)
        assert len(t) == len(y)
        assert len(t) == len(dydt)
        assert len(t) == len(d2ydt2)

        assert t[0] == 0.0
        assert t[-1] == pytest.approx((n - 1) / fs)

        assert y.max() == pytest.approx(1.0)
        assert y.min() == pytest.approx(-1.0)
        assert y.mean() == pytest.approx(0.0)
        assert y.std() == pytest.approx(0.5)

        np.testing.assert_allclose(
            dydt[1:-1], np.gradient(y, t)[1:-1], atol=0.0025
        )  # Verified tolerance

        np.testing.assert_allclose(
            d2ydt2[1:-1], np.gradient(dydt, t)[1:-1], atol=0.0025
        )  # Verified tolerance

        y_phase, dydt_phase, d2ydt2_phase = beat_signal(t, phase=30.0)
        assert not np.array_equal(y, y_phase)
        assert not np.array_equal(dydt, dydt_phase)
        assert not np.array_equal(d2ydt2, d2ydt2_phase)


class Test_ChirpSignal:
    def test_init(self):
        chirp_signal = benchmark.ChirpSignal(0.1, 0.01)
        assert chirp_signal._f_max == 0.1 * 2.0 * np.pi
        assert chirp_signal._f_os == 0.01 * 2.0 * np.pi

    def test_init_rads(self):
        chirp_signal = benchmark.ChirpSignal(0.1, 0.01, freq_hz=False)
        assert chirp_signal._f_max == 0.1
        assert chirp_signal._f_os == 0.01

    def test_signal(self):
        chirp_signal = benchmark.ChirpSignal(0.1, 0.01, freq_hz=True)

        fs = 10.0
        n = 10_000
        t = np.linspace(0.0, n / fs, n, endpoint=False)
        y, dydt, d2ydt2 = chirp_signal(t)

        assert len(t) == int(n)
        assert len(t) == len(y)
        assert len(t) == len(dydt)

        assert t[0] == 0.0
        assert t[-1] == pytest.approx((n - 1) / fs)

        assert y.max() == pytest.approx(1.0, rel=1e-3)
        assert y.min() == pytest.approx(-1.0, rel=1e-3)
        assert y.mean() == pytest.approx(0.0)
        assert 0.5 < y.std() < 1.0

        np.testing.assert_allclose(
            dydt[1:-1], np.gradient(y, t)[1:-1], atol=0.0025
        )  # Verified tolerance

        np.testing.assert_allclose(
            d2ydt2[1:-1], np.gradient(dydt, t)[1:-1], atol=0.0025
        )  # Verified tolerance

        y_phase, dydt_phase, d2ydt2_phase = chirp_signal(t, phase=30.0)
        assert not np.array_equal(y, y_phase)
        assert not np.array_equal(dydt, dydt_phase)
        assert not np.array_equal(d2ydt2, d2ydt2_phase)


class Test__benchmark_helper:
    def test_signal_family(self):
        f_main = 0.1
        f_beat = 0.01
        signal_family = benchmark.BeatSignal(f_main, f_beat)

        duration = 100.0
        fs = 10.0
        amplitude = (0.5, 0.5, 0.5, *np.radians((5.0, 5.0, 5.0)))
        mean = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        phase = tuple(np.radians((0.0, 22.5, 45.0, 67.5, 90.0, 112.5)))
        t, pos, vel, euler, acc, gyro = benchmark._benchmark_helper(
            duration, amplitude, mean, phase, signal_family, fs
        )

        assert len(t) == int(duration * fs)
        assert pos.shape == (len(t), 3)
        assert vel.shape == (len(t), 3)
        assert euler.shape == (len(t), 3)
        assert acc.shape == (len(t), 3)
        assert gyro.shape == (len(t), 3)

        np.testing.assert_allclose(
            pos[:, 0], amplitude[0] * signal_family(t, phase[0], phase_degrees=False)[0]
        )
        np.testing.assert_allclose(
            pos[:, 1], amplitude[1] * signal_family(t, phase[1], phase_degrees=False)[0]
        )
        np.testing.assert_allclose(
            pos[:, 2], amplitude[2] * signal_family(t, phase[2], phase_degrees=False)[0]
        )
        np.testing.assert_allclose(
            euler[:, 0],
            amplitude[3] * signal_family(t, phase[3], phase_degrees=False)[0],
        )
        np.testing.assert_allclose(
            euler[:, 1],
            amplitude[4] * signal_family(t, phase[4], phase_degrees=False)[0],
        )
        np.testing.assert_allclose(
            euler[:, 2],
            amplitude[5] * signal_family(t, phase[5], phase_degrees=False)[0],
        )

    def test_pure_roll(self):
        f_main = 0.1
        f_beat = 0.01
        signal_family = benchmark.BeatSignal(f_main, f_beat)

        duration = 100.0
        fs = 10.0
        amplitude = (0.0, 0.0, 0.0, 0.1, 0.0, 0.0)
        mean = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        phase = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        t, pos, vel, euler, acc, gyro = benchmark._benchmark_helper(
            duration, amplitude, mean, phase, signal_family, fs
        )

        assert len(t) == int(duration * fs)
        assert pos.shape == (len(t), 3)
        assert vel.shape == (len(t), 3)
        assert euler.shape == (len(t), 3)
        assert acc.shape == (len(t), 3)
        assert gyro.shape == (len(t), 3)

        np.testing.assert_allclose(acc[:, 0], np.zeros_like(t))
        np.testing.assert_allclose(acc[:, 1], -9.80665 * np.sin(euler[:, 0]))
        np.testing.assert_allclose(acc[:, 2], -9.80665 * np.cos(euler[:, 0]))

        np.testing.assert_allclose(gyro[:, 0], np.gradient(euler[:, 0], t), atol=0.003)
        np.testing.assert_allclose(gyro[:, 1], np.zeros_like(t))
        np.testing.assert_allclose(gyro[:, 2], np.zeros_like(t))

    def test_pure_pitch(self):
        f_main = 0.1
        f_beat = 0.01
        signal_family = benchmark.BeatSignal(f_main, f_beat)

        duration = 100.0
        fs = 10.0
        amplitude = (0.0, 0.0, 0.0, 0.0, 0.1, 0.0)
        mean = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        phase = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        t, pos, vel, euler, acc, gyro = benchmark._benchmark_helper(
            duration, amplitude, mean, phase, signal_family, fs
        )

        assert len(t) == int(duration * fs)
        assert pos.shape == (len(t), 3)
        assert vel.shape == (len(t), 3)
        assert euler.shape == (len(t), 3)
        assert acc.shape == (len(t), 3)
        assert gyro.shape == (len(t), 3)

        np.testing.assert_allclose(acc[:, 0], 9.80665 * np.sin(euler[:, 1]))
        np.testing.assert_allclose(acc[:, 1], np.zeros_like(t))
        np.testing.assert_allclose(acc[:, 2], -9.80665 * np.cos(euler[:, 1]))

        np.testing.assert_allclose(gyro[:, 0], np.zeros_like(t))
        np.testing.assert_allclose(gyro[:, 1], np.gradient(euler[:, 1], t), atol=0.003)
        np.testing.assert_allclose(gyro[:, 2], np.zeros_like(t))

    def test_pure_yaw(self):
        f_main = 0.1
        f_beat = 0.01
        signal_family = benchmark.BeatSignal(f_main, f_beat)

        duration = 100.0
        fs = 10.0
        amplitude = (0.0, 0.0, 0.0, 0.0, 0.0, 0.1)
        mean = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        phase = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        t, pos, vel, euler, acc, gyro = benchmark._benchmark_helper(
            duration, amplitude, mean, phase, signal_family, fs
        )

        assert len(t) == int(duration * fs)
        assert pos.shape == (len(t), 3)
        assert vel.shape == (len(t), 3)
        assert euler.shape == (len(t), 3)
        assert acc.shape == (len(t), 3)
        assert gyro.shape == (len(t), 3)

        np.testing.assert_allclose(acc[:, 0], np.zeros_like(t))
        np.testing.assert_allclose(acc[:, 1], np.zeros_like(t))
        np.testing.assert_allclose(acc[:, 2], -9.80665 * np.ones_like(t))

        np.testing.assert_allclose(gyro[:, 0], np.zeros_like(t))
        np.testing.assert_allclose(gyro[:, 1], np.zeros_like(t))
        np.testing.assert_allclose(gyro[:, 2], np.gradient(euler[:, 2], t), atol=0.003)

    def test_attitude_mean(self):
        f_main = 0.1
        f_beat = 0.01
        signal_family = benchmark.BeatSignal(f_main, f_beat)

        duration = 100.0
        fs = 10.0
        amplitude = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        mean = (0.0, 0.0, 0.0, 0.1, 0.2, 0.3)
        phase = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        t, pos, vel, euler, acc, gyro = benchmark._benchmark_helper(
            duration, amplitude, mean, phase, signal_family, fs
        )

        assert len(t) == int(duration * fs)
        assert pos.shape == (len(t), 3)
        assert vel.shape == (len(t), 3)
        assert euler.shape == (len(t), 3)
        assert acc.shape == (len(t), 3)
        assert gyro.shape == (len(t), 3)

        np.testing.assert_allclose(acc[:, 0], -9.80665 * -np.sin(0.2))
        np.testing.assert_allclose(acc[:, 1], -9.80665 * np.sin(0.1) * np.cos(0.2))
        np.testing.assert_allclose(acc[:, 2], -9.80665 * np.cos(0.1) * np.cos(0.2))

        np.testing.assert_allclose(
            np.sqrt(acc[:, 0] ** 2 + acc[:, 1] ** 2 + acc[:, 2] ** 2),
            9.80665 * np.ones_like(t),
        )

        np.testing.assert_allclose(gyro[:, 0], np.zeros_like(t))
        np.testing.assert_allclose(gyro[:, 1], np.zeros_like(t))
        np.testing.assert_allclose(gyro[:, 2], np.zeros_like(t))

    def test_pure_translation(self):
        f_main = 0.1
        f_beat = 0.01
        signal_family = benchmark.BeatSignal(f_main, f_beat)

        duration = 100.0
        fs = 10.0
        amplitude = (1.0, 2.0, 3.0, 0.0, 0.0, 0.0)
        mean = (-1.0, -2.0, -3.0, 0.0, 0.0, 0.0)
        phase = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        t, pos, vel, euler, acc, gyro = benchmark._benchmark_helper(
            duration, amplitude, mean, phase, signal_family, fs
        )

        assert len(t) == int(duration * fs)
        assert pos.shape == (len(t), 3)
        assert vel.shape == (len(t), 3)
        assert euler.shape == (len(t), 3)
        assert acc.shape == (len(t), 3)
        assert gyro.shape == (len(t), 3)

        np.testing.assert_allclose(pos[:, 0].mean(), -1.0, atol=0.1)
        np.testing.assert_allclose(pos[:, 1].mean(), -2.0, atol=0.1)
        np.testing.assert_allclose(pos[:, 2].mean(), -3.0, atol=0.1)

        np.testing.assert_allclose(vel[:, 0], np.gradient(pos[:, 0], t), atol=0.005)
        np.testing.assert_allclose(vel[:, 1], np.gradient(pos[:, 1], t), atol=0.005)
        np.testing.assert_allclose(vel[:, 2], np.gradient(pos[:, 2], t), atol=0.005)

        np.testing.assert_allclose(acc[:, 0], np.gradient(vel[:, 0], t), atol=0.01)
        np.testing.assert_allclose(acc[:, 1], np.gradient(vel[:, 1], t), atol=0.01)
        np.testing.assert_allclose(
            acc[:, 2], -9.80665 + np.gradient(vel[:, 2], t), atol=0.01
        )

        np.testing.assert_allclose(gyro[:, 0], np.zeros_like(t))
        np.testing.assert_allclose(gyro[:, 1], np.zeros_like(t))
        np.testing.assert_allclose(gyro[:, 2], np.zeros_like(t))


def test_benchmark_pure_attitude_beat_202311A():
    signature_signal, _, _ = benchmark.BeatSignal(0.1, 0.01)(
        np.arange(0.0, 1800.0, 1.0 / 10.24)
    )

    t, euler, acc, gyro = benchmark.benchmark_pure_attitude_beat_202311A()

    assert len(t) == int(1800 * 10.24)
    assert euler.shape == (len(t), 3)
    assert acc.shape == (len(t), 3)
    assert gyro.shape == (len(t), 3)

    np.testing.assert_allclose(euler[:, 0], np.radians(5.0) * signature_signal)


def test_benchmark_pure_attitude_chirp_202311A():
    signature_signal, _, _ = benchmark.ChirpSignal(0.25, 0.01)(
        np.arange(0.0, 1800.0, 1.0 / 10.24)
    )

    t, euler, acc, gyro = benchmark.benchmark_pure_attitude_chirp_202311A()

    assert len(t) == int(1800 * 10.24)
    assert euler.shape == (len(t), 3)
    assert acc.shape == (len(t), 3)
    assert gyro.shape == (len(t), 3)

    np.testing.assert_array_equal(euler[:, 0], np.radians(5.0) * signature_signal)


def test_benchmark_full_pva_beat_202311A():
    signature_signal, _, _ = benchmark.BeatSignal(0.1, 0.01)(
        np.arange(0.0, 1800.0, 1.0 / 10.24)
    )

    t, pos, vel, euler, acc, gyro = benchmark.benchmark_full_pva_beat_202311A()

    assert len(t) == int(1800 * 10.24)
    assert pos.shape == (len(t), 3)
    assert vel.shape == (len(t), 3)
    assert euler.shape == (len(t), 3)
    assert acc.shape == (len(t), 3)
    assert gyro.shape == (len(t), 3)

    np.testing.assert_array_equal(pos[:, 0], 0.5 * signature_signal)


def test_benchmark_full_pva_chirp_202311A():
    signature_signal, _, _ = benchmark.ChirpSignal(0.25, 0.01)(
        np.arange(0.0, 1800.0, 1.0 / 10.24)
    )

    t, pos, vel, euler, acc, gyro = benchmark.benchmark_full_pva_chirp_202311A()

    assert len(t) == int(1800 * 10.24)
    assert pos.shape == (len(t), 3)
    assert vel.shape == (len(t), 3)
    assert euler.shape == (len(t), 3)
    assert acc.shape == (len(t), 3)
    assert gyro.shape == (len(t), 3)

    np.testing.assert_array_equal(pos[:, 0], 0.5 * signature_signal)
