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

    def test_bias_gyro(self):
        bg0 = np.array([0.01, -0.02, 0.03])
        smoother = FixedIntervalSmoother(PVAMEKF(10.0, bg0=bg0))
        for _ in range(10):
            smoother.update(np.array([0.0, 0.0, -0.98]), np.zeros(3))

        bg_rad = smoother.bias_gyro()
        np.testing.assert_allclose(smoother.bias_gyro(degrees=False), bg_rad)
        np.testing.assert_allclose(smoother.bias_gyro(degrees=True), np.degrees(bg_rad))
        assert smoother.bias_gyro() is not smoother._bg_b  # copy

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
