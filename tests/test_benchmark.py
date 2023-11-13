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


class Test_benchmark_ahrs:
    def test_signal_family_default(self):
        signal_family = benchmark.BeatSignal()

        fs = 10.0
        n = 10_000
        t, acc, gyro, euler = benchmark.benchmark_ahrs(
            fs,
            n,
            (5.0, 5.0, 5.0),
            (0.0, 0.0, 0.0),
            (0.0, 45.0, 90.0),
            phase_degrees=True,
        )

        assert len(t) == int(n)
        assert acc.shape == (n, 3)
        assert gyro.shape == (n, 3)
        assert euler.shape == (n, 3)

        assert t[0] == 0.0
        assert t[-1] == pytest.approx((n - 1) / fs)

        np.testing.assert_allclose(euler[:, 0], signal_family(fs, n, 5.0, 0.0)[1])
        np.testing.assert_allclose(euler[:, 1], signal_family(fs, n, 5.0, 45.0)[1])
        np.testing.assert_allclose(euler[:, 2], signal_family(fs, n, 5.0, 90.0)[1])

    def test_signal_family_beat(self):
        signal_family = benchmark.BeatSignal(0.09, 0.02)

        fs = 10.0
        n = 10_000
        t, acc, gyro, euler = benchmark.benchmark_ahrs(
            fs,
            n,
            (5.0, 5.0, 5.0),
            (0.0, 0.0, 0.0),
            (0.0, 45.0, 90.0),
            phase_degrees=True,
            signal_family=signal_family,
        )

        assert len(t) == int(n)
        assert acc.shape == (n, 3)
        assert gyro.shape == (n, 3)
        assert euler.shape == (n, 3)

        assert t[0] == 0.0
        assert t[-1] == pytest.approx((n - 1) / fs)

        np.testing.assert_allclose(euler[:, 0], signal_family(fs, n, 5.0, 0.0)[1])
        np.testing.assert_allclose(euler[:, 1], signal_family(fs, n, 5.0, 45.0)[1])
        np.testing.assert_allclose(euler[:, 2], signal_family(fs, n, 5.0, 90.0)[1])

    def test_signal_family_chirp(self):
        signal_family = benchmark.ChirpSignal(0.5, 0.05)

        fs = 10.0
        n = 10_000
        t, acc, gyro, euler = benchmark.benchmark_ahrs(
            fs,
            n,
            (5.0, 5.0, 5.0),
            (0.0, 0.0, 0.0),
            (0.0, 45.0, 90.0),
            phase_degrees=True,
            signal_family=signal_family,
        )

        assert len(t) == int(n)
        assert acc.shape == (n, 3)
        assert gyro.shape == (n, 3)
        assert euler.shape == (n, 3)

        assert t[0] == 0.0
        assert t[-1] == pytest.approx((n - 1) / fs)

        np.testing.assert_allclose(euler[:, 0], signal_family(fs, n, 5.0, 0.0)[1])
        np.testing.assert_allclose(euler[:, 1], signal_family(fs, n, 5.0, 45.0)[1])
        np.testing.assert_allclose(euler[:, 2], signal_family(fs, n, 5.0, 90.0)[1])

    def test_pure_roll(self):
        fs = 10.0
        n = 10_000
        t, acc, gyro, euler = benchmark.benchmark_ahrs(
            fs, n, (5.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), phase_degrees=True
        )

        assert len(t) == int(n)
        assert acc.shape == (n, 3)
        assert gyro.shape == (n, 3)
        assert euler.shape == (n, 3)

        assert t[0] == 0.0
        assert t[-1] == pytest.approx((n - 1) / fs)

        np.testing.assert_allclose(acc[:, 0], np.zeros(n))
        np.testing.assert_allclose(acc[:, 1], 9.80665 * np.sin(np.radians(euler[:, 0])))
        np.testing.assert_allclose(
            acc[:, 2], -9.80665 * np.cos(np.radians(euler[:, 0]))
        )

        np.testing.assert_allclose(gyro[:, 0], np.gradient(euler[:, 0], t), atol=0.003)
        np.testing.assert_allclose(gyro[:, 1], np.zeros(n))
        np.testing.assert_allclose(gyro[:, 2], np.zeros(n))

    def test_pure_pitch(self):
        fs = 10.0
        n = 10_000
        t, acc, gyro, euler = benchmark.benchmark_ahrs(
            fs, n, (0.0, 5.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), phase_degrees=True
        )

        assert len(t) == int(n)
        assert acc.shape == (n, 3)
        assert gyro.shape == (n, 3)
        assert euler.shape == (n, 3)

        assert t[0] == 0.0
        assert t[-1] == pytest.approx((n - 1) / fs)

        np.testing.assert_allclose(
            acc[:, 0], -9.80665 * np.sin(np.radians(euler[:, 1]))
        )
        np.testing.assert_allclose(acc[:, 1], np.zeros(n))
        np.testing.assert_allclose(
            acc[:, 2], -9.80665 * np.cos(np.radians(euler[:, 1]))
        )

        np.testing.assert_allclose(gyro[:, 0], np.zeros(n))
        np.testing.assert_allclose(gyro[:, 1], np.gradient(euler[:, 1], t), atol=0.003)
        np.testing.assert_allclose(gyro[:, 2], np.zeros(n))

    def test_pure_yaw(self):
        fs = 10.0
        n = 10_000
        t, acc, gyro, euler = benchmark.benchmark_ahrs(
            fs, n, (0.0, 0.0, 5.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), phase_degrees=True
        )

        assert len(t) == int(n)
        assert acc.shape == (n, 3)
        assert gyro.shape == (n, 3)
        assert euler.shape == (n, 3)

        assert t[0] == 0.0
        assert t[-1] == pytest.approx((n - 1) / fs)

        np.testing.assert_allclose(acc[:, 0], np.zeros(n))
        np.testing.assert_allclose(acc[:, 1], np.zeros(n))
        np.testing.assert_allclose(acc[:, 2], -9.80665 * np.ones(n))

        np.testing.assert_allclose(gyro[:, 0], np.zeros(n))
        np.testing.assert_allclose(gyro[:, 0], np.zeros(n))
        np.testing.assert_allclose(gyro[:, 2], np.gradient(euler[:, 2], t), atol=0.003)

    def test_pure_roll_mean(self):
        fs = 10.0
        n = 10_000
        t, acc, gyro, euler = benchmark.benchmark_ahrs(
            fs, n, (0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (0.0, 0.0, 0.0), phase_degrees=True
        )

        assert len(t) == int(n)
        assert acc.shape == (n, 3)
        assert gyro.shape == (n, 3)
        assert euler.shape == (n, 3)

        assert t[0] == 0.0
        assert t[-1] == pytest.approx((n - 1) / fs)

        np.testing.assert_allclose(acc[:, 0], np.zeros(n))
        np.testing.assert_allclose(acc[:, 1], 9.80665 * np.sin(np.radians(euler[:, 0])))
        np.testing.assert_allclose(
            acc[:, 2], -9.80665 * np.cos(np.radians(euler[:, 0]))
        )

        np.testing.assert_allclose(gyro[:, 0], np.zeros(n))
        np.testing.assert_allclose(gyro[:, 1], np.zeros(n))
        np.testing.assert_allclose(gyro[:, 2], np.zeros(n))

    def test_pure_pitch_mean(self):
        fs = 10.0
        n = 10_000
        t, acc, gyro, euler = benchmark.benchmark_ahrs(
            fs, n, (0.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 0.0), phase_degrees=True
        )

        assert len(t) == int(n)
        assert acc.shape == (n, 3)
        assert gyro.shape == (n, 3)
        assert euler.shape == (n, 3)

        assert t[0] == 0.0
        assert t[-1] == pytest.approx((n - 1) / fs)

        np.testing.assert_allclose(
            acc[:, 0], -9.80665 * np.sin(np.radians(euler[:, 1]))
        )
        np.testing.assert_allclose(acc[:, 1], np.zeros(n))
        np.testing.assert_allclose(
            acc[:, 2], -9.80665 * np.cos(np.radians(euler[:, 1]))
        )

        np.testing.assert_allclose(gyro[:, 0], np.zeros(n))
        np.testing.assert_allclose(gyro[:, 1], np.zeros(n))
        np.testing.assert_allclose(gyro[:, 2], np.zeros(n))

    def test_pure_yaw_mean(self):
        fs = 10.0
        n = 10_000
        t, acc, gyro, euler = benchmark.benchmark_ahrs(
            fs, n, (0.0, 0.0, 0.0), (0.0, 0.0, 5.0), (0.0, 0.0, 0.0), phase_degrees=True
        )

        assert len(t) == int(n)
        assert acc.shape == (n, 3)
        assert gyro.shape == (n, 3)
        assert euler.shape == (n, 3)

        assert t[0] == 0.0
        assert t[-1] == pytest.approx((n - 1) / fs)

        np.testing.assert_allclose(acc[:, 0], np.zeros(n))
        np.testing.assert_allclose(acc[:, 1], np.zeros(n))
        np.testing.assert_allclose(acc[:, 2], -9.80665 * np.ones(n))

        np.testing.assert_allclose(gyro[:, 0], np.zeros(n))
        np.testing.assert_allclose(gyro[:, 0], np.zeros(n))
        np.testing.assert_allclose(gyro[:, 2], np.zeros(n))
