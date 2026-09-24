from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import smsfusion as sf
from smsfusion.noise import (
    IMUNoise,
    NoiseModel,
    allan_var,
    gauss_markov,
    random_walk,
    white_noise,
)

TEST_PATH = Path(__file__).parent


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


class Test_NoiseModel:
    def test__init__(self):
        noise = NoiseModel(1, 2, 3, 4, 5, 6, 7)

        assert noise._N == 1
        assert noise._B == 2
        assert noise._tau_cb == 3
        assert noise._K == 4
        assert noise._tau_ck == 5
        assert noise._bc == 6
        np.testing.assert_array_equal(
            noise._rng.random(10), np.random.default_rng(7).random(10)
        )

    def test__init__default(self):
        noise = NoiseModel(1, 2, 3)

        assert noise._N == 1
        assert noise._B == 2
        assert noise._tau_cb == 3
        assert noise._K is None
        assert noise._tau_ck is None
        assert noise._bc == 0.0
        assert isinstance(noise._rng, np.random.Generator)

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

    def test_multiple_calls_varying_output(self):
        # Repeated calls to NoiseModel should yield different outputs
        N = 4.0e-4
        B = 3.0e-4
        tau_cb = 10
        K = 3.0e-5
        tau_ck = None  # Random walk (RW) drift model
        bc = 0.1
        noise = NoiseModel(N, B, tau_cb, K, tau_ck, bc, seed=123)
        num_samples = 100
        sampling_frequency = 10.24
        x_out1 = noise(sampling_frequency, num_samples)
        x_out2 = noise(sampling_frequency, num_samples)
        assert not np.array_equal(x_out1, x_out2)


class Test_IMUNoise:
    @pytest.fixture
    def params_scalar(self):
        params = {
            "acc_noise_density": 1.0e-4,
            "acc_bias_stability": 1.0e-5,
            "acc_bias_corr_time": 10.0,
            "acc_drift_rate": 1.0e-6,
            "acc_drift_corr_time": 1_000.0,
            "acc_constant_bias": 0.1,
            "gyro_noise_density": 4.0e-4,
            "gyro_bias_stability": 4.0e-5,
            "gyro_bias_corr_time": 40.0,
            "gyro_drift_rate": 4.0e-6,
            "gyro_drift_corr_time": 4_000.0,
            "gyro_constant_bias": 0.4,
        }
        return params

    @pytest.fixture
    def params_full(self):
        params = {
            "acc_noise_density": [1.0e-4, 2.0e-4, 3e-4],
            "acc_bias_stability": [1.0e-5, 2.0e-5, 3.0e-5],
            "acc_bias_corr_time": [10.0, 20.0, 30.0],
            "acc_drift_rate": [1.0e-6, 2.0e-6, 3.0e-6],
            "acc_drift_corr_time": [1_000.0, 2_000.0, 3_000.0],
            "acc_constant_bias": [0.1, 0.2, 0.3],
            "gyro_noise_density": [4.0e-4, 5.0e-4, 6e-4],
            "gyro_bias_stability": [4.0e-5, 5.0e-5, 6.0e-5],
            "gyro_bias_corr_time": [40.0, 50.0, 60.0],
            "gyro_drift_rate": [4.0e-6, 5.0e-6, 6.0e-6],
            "gyro_drift_corr_time": [4_000.0, 5_000.0, 6_000.0],
            "gyro_constant_bias": [0.4, 0.5, 0.6],
        }
        return params

    def test__init__full(self, params_full):
        noise = IMUNoise(**params_full, seed=123)

        assert noise._seed == 123

        err_list_expect = [
            {
                "bc": 0.1,
                "N": 1.0e-4,
                "B": 1.0e-5,
                "K": 1.0e-6,
                "tau_cb": 10.0,
                "tau_ck": 1_000.0,
            },
            {
                "bc": 0.2,
                "N": 2.0e-4,
                "B": 2.0e-5,
                "K": 2.0e-6,
                "tau_cb": 20.0,
                "tau_ck": 2_000.0,
            },
            {
                "bc": 0.3,
                "N": 3.0e-4,
                "B": 3.0e-5,
                "K": 3.0e-6,
                "tau_cb": 30.0,
                "tau_ck": 3_000.0,
            },
            {
                "bc": 0.4,
                "N": 4.0e-4,
                "B": 4.0e-5,
                "K": 4.0e-6,
                "tau_cb": 40.0,
                "tau_ck": 4_000.0,
            },
            {
                "bc": 0.5,
                "N": 5.0e-4,
                "B": 5.0e-5,
                "K": 5.0e-6,
                "tau_cb": 50.0,
                "tau_ck": 5_000.0,
            },
            {
                "bc": 0.6,
                "N": 6.0e-4,
                "B": 6.0e-5,
                "K": 6.0e-6,
                "tau_cb": 60.0,
                "tau_ck": 6_000.0,
            },
        ]
        assert noise._err_list == err_list_expect

    def test__init__scalar(self, params_scalar):
        noise = IMUNoise(**params_scalar)

        err_acc_expect = {
            "bc": 0.1,
            "N": 1.0e-4,
            "B": 1.0e-5,
            "K": 1.0e-6,
            "tau_cb": 10.0,
            "tau_ck": 1_000.0,
        }
        err_gyro_expect = {
            "bc": 0.4,
            "N": 4.0e-4,
            "B": 4.0e-5,
            "K": 4.0e-6,
            "tau_cb": 40.0,
            "tau_ck": 4_000.0,
        }
        err_list_expect = [err_acc_expect] * 3 + [err_gyro_expect] * 3

        assert noise._seed is None
        assert noise._err_list == err_list_expect

    def test__init__default(self):
        noise = IMUNoise()

        err_acc_expect = {
            "N": 0.0007,
            "B": 0.0005,
            "tau_cb": 50.0,
            "K": None,
            "tau_ck": None,
            "bc": 0.0,
        }
        err_gyro_expect = {
            "N": 0.00005,
            "B": 0.00005,
            "tau_cb": 50.0,
            "K": None,
            "tau_ck": None,
            "bc": 0.0,
        }
        err_list_expect = [err_acc_expect] * 3 + [err_gyro_expect] * 3

        assert noise._seed is None
        assert noise._err_list == err_list_expect

    def test__init__default_matches_constants(self):
        # Default values should correspond to SMS Motion 2 noise levels
        noise = IMUNoise()
        for err in noise._err_list[:3]:
            assert err["N"] == sf.constants.ACC_NOISE_DENSITY
            assert err["B"] == sf.constants.ACC_BIAS_STABILITY
            assert err["tau_cb"] == sf.constants.ACC_BIAS_CORR_TIME
        for err in noise._err_list[3:]:
            assert err["N"] == sf.constants.GYRO_NOISE_DENSITY
            assert err["B"] == sf.constants.GYRO_BIAS_STABILITY
            assert err["tau_cb"] == sf.constants.GYRO_BIAS_CORR_TIME

    def test__init__mixed(self):
        noise = IMUNoise(
            acc_noise_density=(1.0e-4, 2.0e-4, 3.0e-4),
            gyro_bias_stability=4.0e-5,
        )

        assert [err["N"] for err in noise._err_list[:3]] == [1.0e-4, 2.0e-4, 3.0e-4]
        assert [err["N"] for err in noise._err_list[3:]] == [0.00005] * 3
        assert [err["B"] for err in noise._err_list[:3]] == [0.0005] * 3
        assert [err["B"] for err in noise._err_list[3:]] == [4.0e-5] * 3

    def test__init__positional_raises(self):
        with pytest.raises(TypeError):
            IMUNoise(0.0007)

    @pytest.mark.parametrize(
        "key",
        [
            "acc_noise_density",
            "acc_bias_stability",
            "acc_bias_corr_time",
            "acc_drift_rate",
            "acc_drift_corr_time",
            "acc_constant_bias",
            "gyro_noise_density",
            "gyro_bias_stability",
            "gyro_bias_corr_time",
            "gyro_drift_rate",
            "gyro_drift_corr_time",
            "gyro_constant_bias",
        ],
    )
    def test__init__raises_values(self, params_full, key):
        with pytest.raises(ValueError):  # missing value
            IMUNoise(**(params_full | {key: [0.1, 0.2]}), seed=123)
        with pytest.raises(ValueError):  # extra value
            IMUNoise(**(params_full | {key: [0.1, 0.2, 0.3, 0.4]}), seed=123)

    def test__to_list(self):
        dict_in = {"a": [1, 2, 3], "b": [4, 5, 6]}
        list_expect = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
        list_out = IMUNoise._to_list(dict_in)
        assert list_out == list_expect

    def test__call__(self):
        noise = IMUNoise(
            acc_noise_density=(4.0e-4, 4.0e-4, 4.5e-4),
            acc_bias_stability=(1.5e-4, 1.5e-4, 3.0e-4),
            acc_bias_corr_time=(50, 50, 30),
            acc_drift_rate=(4.5e-6, 4.5e-6, 1.5e-5),
            acc_drift_corr_time=(5e5, 5e5, 5e5),
            acc_constant_bias=(0.0, 0.0, 0.0),
            gyro_noise_density=(1.9e-3, 1.9e-3, 1.7e-3),
            gyro_bias_stability=(7.5e-4, 4.0e-4, 8.8e-4),
            gyro_bias_corr_time=(50, 50, 50),
            gyro_drift_rate=(2.5e-5, 2.5e-5, 4.0e-5),
            gyro_drift_corr_time=(5e5, 5e5, 5e5),
            gyro_constant_bias=(0.0, 0.0, 0.0),
            seed=123,
        )
        x_out = noise(10.24, 1_000)

        x_expect = pd.read_csv(
            TEST_PATH / "testdata" / "IMUNoise.csv", index_col=0
        ).values

        assert x_out.shape == (1_000, 6)
        np.testing.assert_array_almost_equal(x_out, x_expect)

    def test_different_seeds(self):
        # All channels given same noise parameters
        noise = IMUNoise(
            acc_noise_density=1.0e-4,
            acc_bias_stability=1.0e-5,
            gyro_noise_density=1.0e-4,
            gyro_bias_stability=1.0e-5,
            seed=123,
        )
        x = noise(10.24, 100)

        for i, j in product([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]):
            if i == j:
                continue
            else:
                # check that all channels are given different seeds
                assert not np.array_equal(x[:, i], x[:, j])


class Test_allan_var_overlapping:

    @staticmethod
    def _tqdm_installed():
        import importlib.util

        return importlib.util.find_spec("tqdm") is not None

    @pytest.mark.skipif(_tqdm_installed(), reason="tqdm is installed")
    def test_no_tqdm(self):
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
