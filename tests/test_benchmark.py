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
        t, y, dydt = beat_signal(fs, n)

        assert len(t) == int(n)
        assert len(t) == len(y)
        assert len(t) == len(dydt)

        assert t[0] == 0.0
        assert t[-1] == pytest.approx((n - 1) / fs)

        assert y.max() == pytest.approx(1.0)
        assert y.min() == pytest.approx(-1.0)
        assert y.mean() == pytest.approx(0.0)
        assert y.std() == pytest.approx(0.5)

        np.testing.assert_allclose(
            dydt, np.gradient(y, t), atol=0.0025
        )  # Verified tolerance

        _, y_phase, dydt_phase = beat_signal(fs, n, phase=30.)
        assert not np.array_equal(y, y_phase)
        assert not np.array_equal(dydt, dydt_phase)
