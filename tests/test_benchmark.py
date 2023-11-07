import numpy as np
import pytest

from smsfusion import benchmark


class Test_BeatSignal:
    def test_init(self):
        beat_signal = benchmark.BeatSignal(0.1, 0.01)
        assert beat_signal._f_main == 0.1 * 2. * np.pi
        assert beat_signal._f_beat == 0.01 * 2. * np.pi

    def test_init_rads(self):
        beat_signal = benchmark.BeatSignal(0.1, 0.01, freq_hz=False)
        assert beat_signal._f_main == 0.1
        assert beat_signal._f_beat == 0.01

    def test_signal(self):
        beat_signal = benchmark.BeatSignal(0.1, 0.01, freq_hz=True)

        fs = 10.
        duration = 200.
        t, y, dydt = beat_signal(fs, duration, amp=5.0)

        assert len(t) == int(duration * fs)
        assert len(t) == len(y)
        assert len(t) == len(dydt)

        assert y.max() == pytest.approx(5.0)
        assert y.min() == pytest.approx(-5.0)
        assert y.mean() == pytest.approx(0.)
        assert y.std() == pytest.approx(2.5)

        np.testing.assert_allclose(dydt, np.gradient(y, t), atol=0.0025)  # Verified tolerance
