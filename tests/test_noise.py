from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smsfusion.noise import gauss_markov, random_walk, white_noise
from smsfusion.noise._noise import _gen_seeds, _standard_normal

TEST_PATH = Path(__file__).parent


def test__standard_normal():
    x = _standard_normal(100)

    assert len(x) == 100
    assert np.mean(x) == pytest.approx(0.0, abs=0.2)  # mean value is zero
    assert np.std(x) == pytest.approx(1.0, abs=0.2)  # std is 1


def test__standard_normal_seeds():
    x0 = _standard_normal(100, seed=0)
    x1 = _standard_normal(100, seed=0)
    x2 = _standard_normal(100, seed=1)

    np.testing.assert_array_almost_equal(x0, x1)
    assert not np.array_equal(x0, x2)


def test_white_noise():
    N, fs, n = 3, 10.0, 100_000
    wn_out = white_noise(N, fs, n, seed=123)

    wn_expect = pd.read_csv(
        TEST_PATH / "testdata" / "white_noise.csv", index_col=0
    ).values.flatten()

    np.testing.assert_array_almost_equal(wn_out, wn_expect)


def test_random_walk():
    K, fs, n = 3, 10.0, 100_000
    rw_out = random_walk(K, fs, n, seed=123)

    rw_expect = pd.read_csv(
        TEST_PATH / "testdata" / "random_walk.csv", index_col=0
    ).values.flatten()

    np.testing.assert_array_almost_equal(rw_out, rw_expect)


def test_gauss_markov():
    sigma, tau_c, fs, n = 3, 5, 10.0, 100_000

    gm_out = gauss_markov(sigma, tau_c, fs, n, seed=123)

    gm_expect = pd.read_csv(
        TEST_PATH / "testdata" / "gauss_markov.csv", index_col=0
    ).values.flatten()

    np.testing.assert_array_almost_equal(gm_out, gm_expect)


class Test_gen_seed:
    def test_one_int(self):
        seeds_out = _gen_seeds(123, 1)
        assert len(seeds_out) == 1
        assert isinstance(seeds_out[0], np.uint64)
        assert seeds_out != 123  # could be the same, but very unlikely

    def test_one_none(self):
        seeds_out = _gen_seeds(None, 1)
        assert len(seeds_out) == 1
        assert isinstance(seeds_out[0], np.uint64)

    def test_multiple_int(self):
        seeds_out = _gen_seeds(123, 3)
        assert len(seeds_out) == 3
        assert len(np.unique(seeds_out)) == 3
        for i in range(3):
            assert isinstance(seeds_out[i], np.uint64)
            assert seeds_out[i] != 123  # could be the same, but very unlikely

    def test_multiple_none(self):
        seeds_out = _gen_seeds(None, 3)
        assert len(seeds_out) == 3
        assert len(np.unique(seeds_out)) == 3
        for i in range(3):
            assert isinstance(seeds_out[i], np.uint64)
