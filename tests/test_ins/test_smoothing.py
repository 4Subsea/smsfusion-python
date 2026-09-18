import numpy as np
import pytest
from scipy.signal import resample_poly

import smsfusion as sf
from smsfusion import PVAMEKF, ConingScullingAlg
from smsfusion._ins._smoothing import FixedIntervalSmoother
from smsfusion.benchmark import (
    benchmark_full_pva_beat_202311A,
    benchmark_full_pva_chirp_202311A,
    benchmark_pure_attitude_beat_202311A,
    benchmark_pure_attitude_chirp_202311A,
)


class Test_FixedIntervalSmoother:

    FS = 10.0

    @classmethod
    def _run(cls, n_samples=50, seed=0, bg0=(0.0, 0.0, 0.0), **smoother_kwargs):
        """
        Run a forward filter and a smoother over identical measurements. The
        measurements describe a nominally stationary and level body.
        """
        rng = np.random.default_rng(seed)
        dvel = np.array([0.0, 0.0, -sf.gravity() / cls.FS]) + rng.normal(
            0.0, 1.0e-3, (n_samples, 3)
        )
        dtheta = rng.normal(0.0, 1.0e-3, (n_samples, 3))

        mekf = PVAMEKF(cls.FS, bg0=bg0)
        smoother = FixedIntervalSmoother(PVAMEKF(cls.FS, bg0=bg0), **smoother_kwargs)
        for dvel_i, dtheta_i in zip(dvel, dtheta):
            mekf.update(dvel_i, dtheta_i)
            smoother.update(dvel_i, dtheta_i)
        return mekf, smoother

    def test_update_returns_self(self):
        smoother = FixedIntervalSmoother(PVAMEKF(self.FS))
        assert smoother.update(np.array([0.0, 0.0, -0.98]), np.zeros(3)) is smoother

    @pytest.mark.parametrize(
        "method, shape",
        [
            ("position", (50, 3)),
            ("velocity", (50, 3)),
            ("quaternion", (50, 4)),
            ("euler", (50, 3)),
            ("bias_gyro", (50, 3)),
        ],
    )
    def test_state_shapes(self, method, shape):
        _, smoother = self._run(n_samples=shape[0])
        assert getattr(smoother, method)().shape == shape

    def test_P_shape(self):
        _, smoother = self._run(n_samples=50)
        assert smoother.P.shape == (50, 12, 12)

    @pytest.mark.parametrize(
        "method, shape",
        [
            ("position", (0, 3)),
            ("velocity", (0, 3)),
            ("quaternion", (0, 4)),
            ("euler", (0, 3)),
            ("bias_gyro", (0, 3)),
        ],
    )
    def test_state_shapes_without_updates(self, method, shape):
        smoother = FixedIntervalSmoother(PVAMEKF(self.FS))
        assert getattr(smoother, method)().shape == shape

    def test_P_shape_without_updates(self):
        smoother = FixedIntervalSmoother(PVAMEKF(self.FS))
        assert smoother.P.shape == (0, 12, 12)

    @pytest.mark.parametrize(
        "method, attribute",
        [
            ("position", "_p_n"),
            ("velocity", "_v_n"),
            ("quaternion", "_q_nb"),
            ("bias_gyro", "_bg_b"),
            ("P", "_P"),
        ],
    )
    def test_state_methods_return_copies(self, method, attribute):
        _, smoother = self._run(n_samples=10)
        out = getattr(smoother, method)
        out = out if method == "P" else out()  # 'P' is a property
        assert out is not getattr(smoother, attribute)

    @pytest.mark.parametrize(
        "method",
        ["position", "velocity", "quaternion", "euler", "bias_gyro"],
    )
    def test_last_sample_equals_forward_filter(self, method):
        """
        The RTS backward sweep leaves the last time step uncorrected, so it must
        equal the forward filter estimate.
        """
        mekf, smoother = self._run(n_samples=50, bg0=(0.01, -0.02, 0.03))
        np.testing.assert_allclose(
            getattr(smoother, method)()[-1], getattr(mekf, method)()
        )

    def test_P_last_sample_equals_forward_filter(self):
        mekf, smoother = self._run(n_samples=50)
        np.testing.assert_allclose(smoother.P[-1], mekf.P)

    def test_bias_gyro(self):
        bg0 = np.array([0.01, -0.02, 0.03])
        _, smoother = self._run(n_samples=10, bg0=bg0)

        bg_rad = smoother.bias_gyro()
        np.testing.assert_allclose(smoother.bias_gyro(degrees=False), bg_rad)
        np.testing.assert_allclose(smoother.bias_gyro(degrees=True), np.degrees(bg_rad))
        assert smoother.bias_gyro() is not smoother._bg_b  # copy

    def test_euler_degrees(self):
        _, smoother = self._run(n_samples=10)

        euler_rad = smoother.euler()
        np.testing.assert_allclose(smoother.euler(degrees=False), euler_rad)
        np.testing.assert_allclose(smoother.euler(degrees=True), np.degrees(euler_rad))

    def test_euler_matches_quaternion(self):
        _, smoother = self._run(n_samples=10)

        quaternion = smoother.quaternion()
        np.testing.assert_allclose(np.linalg.norm(quaternion, axis=1), 1.0)

        quaternion_expect = np.array(
            [sf.quaternion_from_euler(theta_i) for theta_i in smoother.euler()]
        )
        # The quaternion and its negative describe the same rotation
        quaternion_expect *= np.sign(quaternion_expect[:, 0:1] * quaternion[:, 0:1])
        np.testing.assert_allclose(quaternion, quaternion_expect, atol=1e-12)

    def test_smoothing_is_idempotent(self):
        _, smoother = self._run(n_samples=20)

        first = smoother.position()
        second = smoother.position()
        np.testing.assert_array_equal(first, second)

    def test_accessor_before_further_updates(self):
        """
        Reading an estimate must not prevent the smoother from accounting for
        measurements that are applied afterwards.
        """
        _, smoother = self._run(n_samples=20)
        smoother.position()  # trigger a backward sweep

        for _ in range(5):
            smoother.update(np.array([0.0, 0.0, -0.98]), np.zeros(3))

        _, smoother_expect = self._run(n_samples=20)
        for _ in range(5):
            smoother_expect.update(np.array([0.0, 0.0, -0.98]), np.zeros(3))

        assert smoother.position().shape == (25, 3)
        np.testing.assert_allclose(smoother.position(), smoother_expect.position())

    def test_cov_smoothing_false(self):
        """
        Disabling covariance smoothing returns the forward filter covariances, and
        leaves the smoothed state estimates unchanged.
        """
        n_samples = 30
        rng = np.random.default_rng(0)
        dvel = np.array([0.0, 0.0, -sf.gravity() / self.FS]) + rng.normal(
            0.0, 1.0e-3, (n_samples, 3)
        )
        dtheta = rng.normal(0.0, 1.0e-3, (n_samples, 3))
        aid_kwargs = {"pos_var": (0.01, 0.01, 0.01), "vel_var": (0.01, 0.01, 0.01)}

        mekf = PVAMEKF(self.FS)
        smoother = FixedIntervalSmoother(PVAMEKF(self.FS), cov_smoothing=False)
        smoother_cov = FixedIntervalSmoother(PVAMEKF(self.FS), cov_smoothing=True)

        P_fwd = []
        for dvel_i, dtheta_i in zip(dvel, dtheta):
            mekf.update(dvel_i, dtheta_i, **aid_kwargs)
            smoother.update(dvel_i, dtheta_i, **aid_kwargs)
            smoother_cov.update(dvel_i, dtheta_i, **aid_kwargs)
            P_fwd.append(mekf.P)
        P_fwd = np.array(P_fwd)

        # Covariances are passed through unsmoothed
        np.testing.assert_array_equal(smoother.P, P_fwd)

        # ... whereas smoothing them reduces the uncertainty
        assert np.all(np.diagonal(smoother_cov.P[0]) < np.diagonal(P_fwd[0]))

        # The state estimates are unaffected either way
        np.testing.assert_allclose(smoother.position(), smoother_cov.position())
        np.testing.assert_allclose(smoother.velocity(), smoother_cov.velocity())
        np.testing.assert_allclose(smoother.euler(), smoother_cov.euler())
        np.testing.assert_allclose(smoother.bias_gyro(), smoother_cov.bias_gyro())

    @pytest.mark.parametrize(
        "benchmark_gen",
        [
            benchmark_full_pva_beat_202311A,
            benchmark_full_pva_chirp_202311A,
        ],
    )
    def test_benchmark_full_aiding(self, benchmark_gen):
        """
        Full aiding (position, velocity, and heading).

        All degrees of freedom are observable with this aiding configuration.
        """
        fs_imu = 10.0
        warmup = int(fs_imu * 600.0)  # truncate 600 seconds from the beginning

        # Reference signals (without noise)
        t, pos_ref, vel_ref, euler_ref, acc_ref, gyro_ref = benchmark_gen(fs_imu)

        # IMU and aiding measurements (with noise)
        pos_std = 0.1  # m
        vel_std = 0.01  # m/s
        head_std = np.radians(0.1)  # rad
        err_acc = sf.constants.ERR_ACC_MOTION2
        err_gyro = sf.constants.ERR_GYRO_MOTION2
        noise_model = sf.noise.IMUNoise(err_acc=err_acc, err_gyro=err_gyro, seed=0)
        imu_noise = noise_model(fs_imu, len(t))
        acc_meas = acc_ref + imu_noise[:, :3]
        gyro_meas = gyro_ref + imu_noise[:, 3:]
        rng = np.random.default_rng(0)
        pos_meas = pos_ref + rng.normal(0.0, pos_std, pos_ref.shape)
        vel_meas = vel_ref + rng.normal(0.0, vel_std, vel_ref.shape)
        head_meas = euler_ref[:, 2] + rng.normal(0.0, head_std, len(euler_ref))

        # MEKF
        q0 = sf.quaternion_from_euler(euler_ref[0], degrees=False)
        mekf = PVAMEKF(fs_imu, p0=pos_ref[0], v0=vel_ref[0], q0=q0)
        smoother = FixedIntervalSmoother(
            PVAMEKF(fs_imu, p0=pos_ref[0], v0=vel_ref[0], q0=q0)
        )

        # Coning and sculling corrected IMU increments. The crude approximation,
        # dvel = f * dt and dtheta = w * dt, leaves a deterministic rotation
        # compensation error which the RTS backward sweep integrates coherently.
        coning_sculling = ConingScullingAlg(fs_imu)

        pos_fwd, vel_fwd, euler_fwd = [], [], []
        for f_i, w_i, h_i, p_i, v_i in zip(
            acc_meas, gyro_meas, head_meas, pos_meas, vel_meas
        ):

            coning_sculling.update(f_i, w_i)
            dtheta_i, dvel_i = coning_sculling.flush()

            aid_kwargs = {
                "head": h_i,
                "head_var": head_std**2,
                "head_degrees": False,
                "pos": p_i,
                "pos_var": pos_std**2 * np.ones(3),
                "vel": v_i,
                "vel_var": vel_std**2 * np.ones(3),
                "gref": True,
                "gref_var": (0.1, 0.1, 0.1),
            }
            mekf.update(dvel_i, dtheta_i, degrees=False, **aid_kwargs)
            smoother.update(dvel_i, dtheta_i, degrees=False, **aid_kwargs)

            pos_fwd.append(mekf.position())
            vel_fwd.append(mekf.velocity())
            euler_fwd.append(mekf.euler(degrees=False))

        pos_fwd = np.array(pos_fwd)
        vel_fwd = np.array(vel_fwd)
        euler_fwd = np.array(euler_fwd)

        pos_smth = smoother.position()
        vel_smth = smoother.velocity()
        euler_smth = smoother.euler(degrees=False)

        # Half-sample shift (compensates for the time shift introduced by Euler integration)
        pos_fwd = resample_poly(pos_fwd, 2, 1)[1:-1:2]
        vel_fwd = resample_poly(vel_fwd, 2, 1)[1:-1:2]
        euler_fwd = resample_poly(euler_fwd, 2, 1)[1:-1:2]
        pos_smth = resample_poly(pos_smth, 2, 1)[1:-1:2]
        vel_smth = resample_poly(vel_smth, 2, 1)[1:-1:2]
        euler_smth = resample_poly(euler_smth, 2, 1)[1:-1:2]

        pos_ref = pos_ref[1:, :]
        vel_ref = vel_ref[1:, :]
        euler_ref = euler_ref[1:, :]

        def rmse(ref, est):
            return np.sqrt(np.mean((ref - est) ** 2, axis=0))

        pos_rmse_fwd = rmse(pos_ref[warmup:], pos_fwd[warmup:])
        vel_rmse_fwd = rmse(vel_ref[warmup:], vel_fwd[warmup:])
        euler_rmse_fwd = rmse(euler_ref[warmup:], euler_fwd[warmup:])

        pos_rmse_smth = rmse(pos_ref[warmup:], pos_smth[warmup:])
        vel_rmse_smth = rmse(vel_ref[warmup:], vel_smth[warmup:])
        euler_rmse_smth = rmse(euler_ref[warmup:], euler_smth[warmup:])

        # The smoother should improve on every estimate compared to the forward filter
        assert np.all(pos_rmse_smth < pos_rmse_fwd)
        assert np.all(vel_rmse_smth < vel_rmse_fwd)
        assert np.all(euler_rmse_smth < euler_rmse_fwd)

    @pytest.mark.parametrize(
        "benchmark_gen",
        [
            benchmark_full_pva_beat_202311A,
            benchmark_full_pva_chirp_202311A,
        ],
    )
    def test_benchmark_head_aiding(self, benchmark_gen):
        """
        Heading aiding and the default pseudo zero-position and zero-velocity
        measurements are applied.

        Only the attitude (roll, pitch and yaw) is observable with this aiding
        configuration.
        """
        fs_imu = 10.0
        warmup = int(fs_imu * 600.0)  # truncate 600 seconds from the beginning

        # Reference signals (without noise)
        t, _, _, euler_ref, acc_ref, gyro_ref = benchmark_gen(fs_imu)

        # IMU and aiding measurements (with noise)
        head_std = np.radians(0.1)  # rad
        err_acc = sf.constants.ERR_ACC_MOTION2
        err_gyro = sf.constants.ERR_GYRO_MOTION2
        noise_model = sf.noise.IMUNoise(err_acc=err_acc, err_gyro=err_gyro, seed=0)
        imu_noise = noise_model(fs_imu, len(t))
        acc_meas = acc_ref + imu_noise[:, :3]
        gyro_meas = gyro_ref + imu_noise[:, 3:]
        rng = np.random.default_rng(0)
        head_meas = euler_ref[:, 2] + rng.normal(0.0, head_std, len(euler_ref))

        # MEKF
        q0 = sf.quaternion_from_euler(euler_ref[0], degrees=False)
        mekf = PVAMEKF(fs_imu, q0=q0)
        smoother = FixedIntervalSmoother(PVAMEKF(fs_imu, q0=q0))

        euler_fwd = []
        for f_i, w_i, h_i in zip(acc_meas, gyro_meas, head_meas):

            dvel_i = f_i / fs_imu
            dtheta_i = w_i / fs_imu

            aid_kwargs = {"head": h_i, "head_var": head_std**2, "head_degrees": False}
            mekf.update(dvel_i, dtheta_i, degrees=False, **aid_kwargs)
            smoother.update(dvel_i, dtheta_i, degrees=False, **aid_kwargs)

            euler_fwd.append(mekf.euler(degrees=False))

        euler_fwd = np.array(euler_fwd)
        euler_smth = smoother.euler(degrees=False)

        # Half-sample shift (compensates for the time shift introduced by Euler integration)
        euler_fwd = resample_poly(euler_fwd, 2, 1)[1:-1:2]
        euler_smth = resample_poly(euler_smth, 2, 1)[1:-1:2]
        euler_ref = euler_ref[1:, :]

        def rmse(ref, est):
            return np.sqrt(np.mean((ref - est) ** 2, axis=0))

        euler_rmse_fwd = rmse(euler_ref[warmup:], euler_fwd[warmup:])
        euler_rmse_smth = rmse(euler_ref[warmup:], euler_smth[warmup:])

        assert np.all(euler_rmse_smth < euler_rmse_fwd)

    @pytest.mark.parametrize(
        "benchmark_gen",
        [
            benchmark_pure_attitude_beat_202311A,
            benchmark_pure_attitude_chirp_202311A,
        ],
    )
    def test_benchmark_no_aiding(self, benchmark_gen):
        """
        No external aiding, i.e., only the default pseudo zero-position and
        zero-velocity measurements are applied. The body does not translate in this
        benchmark, so these pseudo measurements are valid.

        Only roll, pitch, and the x- and y-axis gyroscope biases are observable in this
        configuration.
        """
        fs_imu = 10.0
        warmup = int(fs_imu * 600.0)  # truncate 600 seconds from the beginning

        # Reference signals (without noise)
        t, euler_ref, acc_ref, gyro_ref = benchmark_gen(fs_imu)

        # IMU measurements (with noise)
        err_acc = sf.constants.ERR_ACC_MOTION2
        err_gyro = sf.constants.ERR_GYRO_MOTION2
        noise_model = sf.noise.IMUNoise(err_acc=err_acc, err_gyro=err_gyro, seed=0)
        imu_noise = noise_model(fs_imu, len(t))
        acc_meas = acc_ref + imu_noise[:, :3]
        gyro_meas = gyro_ref + imu_noise[:, 3:]

        # MEKF
        q0 = sf.quaternion_from_euler(euler_ref[0], degrees=False)
        mekf = PVAMEKF(fs_imu, q0=q0)
        smoother = FixedIntervalSmoother(PVAMEKF(fs_imu, q0=q0))

        euler_fwd = []
        for f_i, w_i in zip(acc_meas, gyro_meas):

            dvel_i = f_i / fs_imu
            dtheta_i = w_i / fs_imu

            mekf.update(dvel_i, dtheta_i, degrees=False)
            smoother.update(dvel_i, dtheta_i, degrees=False)

            euler_fwd.append(mekf.euler(degrees=False))

        euler_fwd = np.array(euler_fwd)
        euler_smth = smoother.euler(degrees=False)

        # Half-sample shift (compensates for the time shift introduced by Euler integration)
        euler_fwd = resample_poly(euler_fwd, 2, 1)[1:-1:2]
        euler_smth = resample_poly(euler_smth, 2, 1)[1:-1:2]

        euler_ref = euler_ref[1:, :]

        def rmse(ref, est):
            return np.sqrt(np.mean((ref - est) ** 2, axis=0))

        euler_rmse_fwd = rmse(euler_ref[warmup:], euler_fwd[warmup:])
        euler_rmse_smth = rmse(euler_ref[warmup:], euler_smth[warmup:])

        # Only roll and pitch are observable with this aiding configuration
        assert np.all(euler_rmse_smth[:2] < euler_rmse_fwd[:2])
