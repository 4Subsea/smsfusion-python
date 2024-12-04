from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smsfusion.noise import (
    IMUNoise,
    NoiseModel,
    allan_var,
    gauss_markov,
    random_walk,
    white_noise,
)
from smsfusion.noise._noise import _gen_seeds, _standard_normal

TEST_PATH = Path(__file__).parent


def test__standard_normal():
    x = _standard_normal(100)

    assert len(x) == 100
    assert np.mean(x) == pytest.approx(0.0, abs=0.3)  # mean value is zero
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


class Test_NoiseModel:
    def test__init__(self):
        noise = NoiseModel(1, 2, 3, 4, 5, 6, 7)

        assert noise._N == 1
        assert noise._B == 2
        assert noise._tau_cb == 3
        assert noise._K == 4
        assert noise._tau_ck == 5
        assert noise._bc == 6
        assert noise._seed == 7

    def test__init__default(self):
        noise = NoiseModel(1, 2, 3)

        assert noise._N == 1
        assert noise._B == 2
        assert noise._tau_cb == 3
        assert noise._K is None
        assert noise._tau_ck is None
        assert noise._bc == 0.0
        assert noise._seed is None

    def test__call__GM(self):
        N = 4.0e-4
        B = 3.0e-4
        tau_cb = 10
        K = 3.0e-5
        tau_ck = 5e5  # Gauss-Markov (GM) drift model
        bc = 0.1
        noise = NoiseModel(N, B, tau_cb, K, tau_ck, bc, seed=123)
        x_out = noise(10.24, 10_000)

        x_expect = pd.read_csv(
            TEST_PATH / "testdata" / "NoiseModel_GMdrift.csv", index_col=0
        ).values.flatten()

        np.testing.assert_array_almost_equal(x_out, x_expect)

    def test__call__RW(self):
        """
        Random walk drift model.
        """
        N = 4.0e-4
        B = 3.0e-4
        tau_cb = 10
        K = 3.0e-5
        tau_ck = None  # Random walk (RW) drift model
        bc = 0.1
        noise = NoiseModel(N, B, tau_cb, K, tau_ck, bc, seed=123)
        x_out = noise(10.24, 10_000)

        x_expect = pd.read_csv(
            TEST_PATH / "testdata" / "NoiseModel_RWdrift.csv", index_col=0
        ).values.flatten()

        np.testing.assert_array_almost_equal(x_out, x_expect, decimal=5)

    def test__call__nodrift(self):
        N = 4.0e-4
        B = 3.0e-4
        tau_cb = 10
        K = None  # No drift
        tau_ck = None
        bc = 0.1
        noise = NoiseModel(N, B, tau_cb, K, tau_ck, bc, seed=123)
        x_out = noise(10.24, 10_000)

        x_expect = pd.read_csv(
            TEST_PATH / "testdata" / "NoiseModel_nodrift.csv", index_col=0
        ).values.flatten()

        np.testing.assert_array_almost_equal(x_out, x_expect, decimal=5)

    def test__call__constant_bias(self):
        N, B, K, tau_cb, tau_ck, bc = 0.0, 0.0, 0.0, 10, None, 1.0
        noise = NoiseModel(N, B, tau_cb, K, tau_ck, bc, 123)
        x_out = noise(10.24, 100)

        assert np.mean(x_out) == 1.0


class Test_IMUNoise:
    def test__init__(self):
        err_acc = {
            "bc": (1.0, 2.0, 3.0),
            "N": (4.0, 5.0, 6.0),
            "B": (7.0, 8.0, 9.0),
            "K": (10.0, 11.0, 12.0),
            "tau_cb": (13.0, 14.0, 15.0),
            "tau_ck": (16.0, 17.0, 18.0),
        }
        err_gyro = {
            "bc": (10.0, 20.0, 30.0),
            "N": (40.0, 50.0, 60.0),
            "B": (70.0, 80.0, 90.0),
            "K": (100.0, 110.0, 120.0),
            "tau_cb": (130.0, 140.0, 150.0),
            "tau_ck": (160.0, 170.0, 180.0),
        }
        noise = IMUNoise(err_acc=err_acc, err_gyro=err_gyro, seed=123)

        assert noise._err_acc == err_acc
        assert noise._err_gyro == err_gyro
        assert noise._seed == 123

        err_list_expect = [
            {
                "bc": 1.0,
                "N": 4.0,
                "B": 7.0,
                "K": 10.0,
                "tau_cb": 13.0,
                "tau_ck": 16.0,
            },
            {
                "bc": 2.0,
                "N": 5.0,
                "B": 8.0,
                "K": 11.0,
                "tau_cb": 14.0,
                "tau_ck": 17.0,
            },
            {
                "bc": 3.0,
                "N": 6.0,
                "B": 9.0,
                "K": 12.0,
                "tau_cb": 15.0,
                "tau_ck": 18.0,
            },
            {
                "bc": 10.0,
                "N": 40.0,
                "B": 70.0,
                "K": 100.0,
                "tau_cb": 130.0,
                "tau_ck": 160.0,
            },
            {
                "bc": 20.0,
                "N": 50.0,
                "B": 80.0,
                "K": 110.0,
                "tau_cb": 140.0,
                "tau_ck": 170.0,
            },
            {
                "bc": 30.0,
                "N": 60.0,
                "B": 90.0,
                "K": 120.0,
                "tau_cb": 150.0,
                "tau_ck": 180.0,
            },
        ]
        assert noise._err_list == err_list_expect

    def test__init__default(self):
        acc_err_expect = {
            "bc": (0.0, 0.0, 0.0),
            "N": (4.0e-4, 4.0e-4, 4.5e-4),
            "B": (1.5e-4, 1.5e-4, 3.0e-4),
            "K": (4.5e-6, 4.5e-6, 1.5e-5),
            "tau_cb": (50, 50, 30),
            "tau_ck": (5e5, 5e5, 5e5),
        }
        gyro_err_expect = {
            "bc": (0.0, 0.0, 0.0),
            "N": (1.9e-3, 1.9e-3, 1.7e-3),
            "B": (7.5e-4, 4.0e-4, 8.8e-4),
            "K": (2.5e-5, 2.5e-5, 4.0e-5),
            "tau_cb": (50, 50, 50),
            "tau_ck": (5e5, 5e5, 5e5),
        }
        noise = IMUNoise()

        assert noise._err_acc == acc_err_expect
        assert noise._err_gyro == gyro_err_expect
        assert noise._seed is None

    def test__init__raises_keys(self):
        err_acc = {
            "invalid_key": (1.0, 2.0, 3.0),
            "N": (4.0, 5.0, 6.0),
            "B": (7.0, 8.0, 9.0),
            "K": (10.0, 11.0, 12.0),
            "tau_cb": (13.0, 14.0, 15.0),
            "tau_ck": (16.0, 17.0, 18.0),
        }
        err_gyro = {
            "bc": (10.0, 20.0, 30.0),
            "N": (40.0, 50.0, 60.0),
            "B": (70.0, 80.0, 90.0),
            "K": (100.0, 110.0, 120.0),
            "tau_cb": (130.0, 140.0, 150.0),
            "tau_ck": (160.0, 170.0, 180.0),
        }
        with pytest.raises(ValueError):
            IMUNoise(err_acc=err_acc, err_gyro=err_gyro, seed=123)

        with pytest.raises(ValueError):
            IMUNoise(err_acc={"bc": 1.0}, seed=123)

    def test__init__raises_values(self):
        err_acc = {
            "bc": (1.0, 2.0),  # one value missing
            "N": (4.0, 5.0, 6.0),
            "B": (7.0, 8.0, 9.0),
            "K": (10.0, 11.0, 12.0),
            "tau_cb": (13.0, 14.0, 15.0),
            "tau_ck": (16.0, 17.0, 18.0),
        }
        err_gyro = {
            "bc": (10.0, 20.0, 30.0, 40.0),  # one value extra
            "N": (40.0, 50.0, 60.0),
            "B": (70.0, 80.0, 90.0),
            "K": (100.0, 110.0, 120.0),
            "tau_cb": (130.0, 140.0, 150.0),
            "tau_ck": (160.0, 170.0, 180.0),
        }
        with pytest.raises(ValueError):
            IMUNoise(err_acc=err_acc, err_gyro=err_gyro, seed=123)

        with pytest.raises(ValueError):
            IMUNoise(err_acc=err_acc, seed=123)

        with pytest.raises(ValueError):
            IMUNoise(err_gyro=err_gyro, seed=123)

    def test__to_list(self):
        dict_in = {"a": [1, 2, 3], "b": [4, 5, 6]}
        list_expect = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
        list_out = IMUNoise._to_list(dict_in)
        assert list_out == list_expect

    def test__call__(self):
        err_acc = {
            "bc": (0.0, 0.0, 0.0),
            "N": (4.0e-4, 4.0e-4, 4.5e-4),
            "B": (1.5e-4, 1.5e-4, 3.0e-4),
            "K": (4.5e-6, 4.5e-6, 1.5e-5),
            "tau_cb": (50, 50, 30),
            "tau_ck": (5e5, 5e5, 5e5),
        }
        err_gyro = {
            "bc": (0.0, 0.0, 0.0),
            "N": (1.9e-3, 1.9e-3, 1.7e-3),
            "B": (7.5e-4, 4.0e-4, 8.8e-4),
            "K": (2.5e-5, 2.5e-5, 4.0e-5),
            "tau_cb": (50, 50, 50),
            "tau_ck": (5e5, 5e5, 5e5),
        }

        noise = IMUNoise(err_acc=err_acc, err_gyro=err_gyro, seed=123)
        x_out = noise(10.24, 1_000)

        x_expect = pd.read_csv(
            TEST_PATH / "testdata" / "IMUNoise.csv", index_col=0
        ).values

        assert x_out.shape == (1_000, 6)
        np.testing.assert_array_almost_equal(x_out, x_expect)

    def test_different_seeds(self):
        # All channels given same noise parameters
        err_acc = {
            "bc": (0.0, 0.0, 0.0),
            "N": (5.0e-4, 5.0e-4, 5.0e-4),
            "B": (1.0e-4, 1.0e-4, 1.0e-4),
            "K": (3.0e-5, 3.0e-5, 3.0e-5),
            "tau_cb": (50, 50, 50),
            "tau_ck": (5e5, 5e5, 5e5),
        }
        err_gyro = {
            "bc": (0.0, 0.0, 0.0),
            "N": (5.0e-4, 5.0e-4, 5.0e-4),
            "B": (1.0e-4, 1.0e-4, 1.0e-4),
            "K": (3.0e-5, 3.0e-5, 3.0e-5),
            "tau_cb": (50, 50, 50),
            "tau_ck": (5e5, 5e5, 5e5),
        }

        noise = IMUNoise(err_acc=err_acc, err_gyro=err_gyro, seed=123)
        x = noise(10.24, 100)

        for i, j in product([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]):
            if i == j:
                continue
            else:
                # check that all channels are given different seeds
                assert not np.array_equal(x[:, i], x[:, j])


class Test_allan_var_overlapping:
    def test_no_tqdm(self, monkeypatch):
        import sys

        monkeypatch.delitem(sys.modules, "tqdm", raising=False)
        with pytest.raises(ImportError):
            y = np.random.random(1_000)
            tau, avar = allan_var(y, 10.0, progress=True)

    def test_with_progress(self, monkeypatch):
        import sys

        class SimpleMock:
            @staticmethod
            def trange(*args, **kwargs):
                return range(*args)

        monkeypatch.setitem(sys.modules, "tqdm", SimpleMock)

        y = np.random.random(1_000)
        tau, avar = allan_var(y, 10.0, progress=False)
        tau_p, avar_p = allan_var(y, 10.0, progress=True)

        np.testing.assert_almost_equal(tau, tau_p)
        np.testing.assert_almost_equal(avar, avar_p)

    def test_single_signal_1d_shape(self):
        y = np.random.random(1_000)
        tau, avar = allan_var(y, 10.0)
        assert avar.shape == (len(tau), 1)

    def test_single_signal_2d_shape(self):
        y = np.random.random((1_000, 1))
        tau, avar = allan_var(y, 10.0)
        assert avar.shape == (len(tau), 1)

    def test_2d_shape(self):
        y = np.random.random((1_000, 3))
        tau, avar = allan_var(y, 10.0)
        assert avar.shape == (len(tau), 3)

    def test_white_noise(self):
        """
        Check if Allan variance is as expected for white noise.
        """
        N = 1.0
        fs = 10.0
        y = N * np.sqrt(fs) * np.random.default_rng().standard_normal(100_000)
        tau, avar = allan_var(y, 10.0)

        log_intercept, log_slope = (
            np.polynomial.Polynomial.fit(np.log(tau), 0.5 * np.log(avar.flatten()), 1)
            .convert()
            .coef
        )
        N_est = np.exp(log_intercept)
        assert log_slope == pytest.approx(-0.5, rel=0.1)
        assert N_est == pytest.approx(N, rel=0.1)

    def test_brown_noise(self):
        """
        Check if Allan variance is as expected for Brown noise.
        """
        K = 1.0
        fs = 10.0
        y = K / np.sqrt(fs) * np.random.default_rng().standard_normal(100_000)
        y = np.cumsum(y)
        tau, avar = allan_var(y, 10.0)

        log_intercept, log_slope = (
            np.polynomial.Polynomial.fit(np.log(tau), 0.5 * np.log(avar.flatten()), 1)
            .convert()
            .coef
        )
        K_est = np.exp(log_intercept + log_slope * np.log(3))
        assert log_slope == pytest.approx(0.5, rel=0.1)
        assert K_est == pytest.approx(K, rel=0.1)

    def test_white_brown_noise(self):
        """
        Check if Allan variance is as expected for white and Brown noise when given in
        one go.
        """
        fs = 10.0
        n = 100_000

        # white noise
        N = 1
        y1 = N * np.sqrt(fs) * np.random.default_rng().standard_normal(n)

        # brown noise
        K = 1.0
        y2 = K / np.sqrt(fs) * np.random.default_rng().standard_normal(n)
        y2 = np.cumsum(y2)

        y = np.column_stack([y1, y2])

        tau, avar = allan_var(y, fs)

        assert avar.shape == (len(tau), 2)

        log_intercept1, log_slope1 = (
            np.polynomial.Polynomial.fit(
                np.log(tau), 0.5 * np.log(avar[:, 0].flatten()), 1
            )
            .convert()
            .coef
        )
        N_est = np.exp(log_intercept1)
        assert log_slope1 == pytest.approx(-0.5, rel=0.1)
        assert N_est == pytest.approx(N, 0.1)

        log_intercept2, log_slope2 = (
            np.polynomial.Polynomial.fit(
                np.log(tau), 0.5 * np.log(avar[:, 1].flatten()), 1
            )
            .convert()
            .coef
        )
        K_est = np.exp(log_intercept2 + log_slope2 * np.log(3))
        assert log_slope2 == pytest.approx(0.5, rel=0.1)
        assert K_est == pytest.approx(K, rel=0.1)
