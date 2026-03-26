import numpy as np
import pytest
from scipy.signal import resample_poly

import smsfusion as sf
from smsfusion._ins_v2 import AHRSv2
from smsfusion.benchmark import (
    benchmark_full_pva_beat_202311A,
    benchmark_full_pva_chirp_202311A,
)


class Test_v2:
    @pytest.mark.parametrize(
        "benchmark_gen",
        [benchmark_full_pva_beat_202311A, benchmark_full_pva_chirp_202311A],
    )
    def test_benchmark(self, benchmark_gen):
        fs_imu = 100.0
        fs_aiding = 1.0
        fs_ratio = np.ceil(fs_imu / fs_aiding)
        warmup = int(fs_imu * 600.0)  # truncate 600 seconds from the beginning
        compass_noise_std = np.radians(0.5)
        vel_noise_std = 0.1

        # Reference signals (without noise)
        t, _, vel_ref, euler_ref, acc_ref, gyro_ref = benchmark_gen(fs_imu)

        # IMU measurements (with noise)
        bg = np.array([0.01, -0.02, 0.015])
        noise_model = sf.noise.IMUNoise(
            err_acc=sf.constants.ERR_ACC_MOTION2,
            err_gyro=sf.constants.ERR_GYRO_MOTION2,
            seed=0,
        )
        imu_noise = noise_model(fs_imu, len(t))
        acc_noise = acc_ref + imu_noise[:, :3]
        gyro_noise = gyro_ref + imu_noise[:, 3:] + bg

        # Aiding measurements (with noise)
        rng = np.random.default_rng(seed=42)
        head_meas = euler_ref[:, 2] + compass_noise_std * rng.standard_normal(
            euler_ref.shape[0]
        )
        vel_meas = vel_ref + vel_noise_std * rng.standard_normal(vel_ref.shape)

        # MEKF
        v0 = vel_ref[0]
        q0 = sf.quaternion_from_euler(euler_ref[0], degrees=False)
        mekf = AHRSv2(
            fs_imu,
            v=v0,
            q=q0,
            acc_noise_density=sf.constants.ERR_ACC_MOTION2["N"],
            gyro_noise_density=sf.constants.ERR_GYRO_MOTION2["N"],
            gyro_bias_stability=sf.constants.ERR_GYRO_MOTION2["B"],
            gyro_bias_corr_time=sf.constants.ERR_GYRO_MOTION2["tau_cb"],
        )

        # Apply filter
        vel_out, euler_out, bias_gyro_out = [], [], []
        for i, (f_i, w_i, v_i, h_i) in enumerate(
            zip(acc_noise, gyro_noise, vel_meas, head_meas)
        ):
            if not (i % fs_ratio):  # with aiding
                mekf.update(
                    f_i / fs_imu,
                    w_i / fs_imu,
                    degrees=False,
                    vel=v_i,
                    vel_var=vel_noise_std**2 * np.ones(3),
                    head=h_i,
                    head_var=compass_noise_std**2,
                    head_degrees=False,
                )
            else:  # without aiding
                mekf.update(f_i / fs_imu, w_i / fs_imu, degrees=False)
            vel_out.append(mekf.velocity())
            euler_out.append(mekf.euler(degrees=False))
            bias_gyro_out.append(mekf.bias_gyro(degrees=False))

        vel_out = np.array(vel_out)
        euler_out = np.array(euler_out)
        bias_gyro_out = np.array(bias_gyro_out)

        # Half-sample shift (compensates for the delay introduced by Euler integration)
        vel_out = resample_poly(vel_out, 2, 1)[1:-1:2]
        vel_ref = vel_ref[:-1, :]
        euler_out = resample_poly(euler_out, 2, 1)[1:-1:2]
        euler_ref = euler_ref[:-1, :]

        vel_x_rms, vel_y_rms, vel_z_rms = np.std((vel_out - vel_ref)[warmup:], axis=0)
        roll_rms, pitch_rms, yaw_rms = np.std((euler_out - euler_ref)[warmup:], axis=0)
        bias_gyro_x_rms, bias_gyro_y_rms, bias_gyro_z_rms = np.std(
            (bias_gyro_out - bg)[warmup:], axis=0
        )

        assert vel_x_rms <= 0.05
        assert vel_y_rms <= 0.05
        assert vel_z_rms <= 0.05
        assert np.degrees(roll_rms) <= 0.1
        assert np.degrees(pitch_rms) <= 0.1
        assert np.degrees(yaw_rms) <= 0.2
        assert np.degrees(bias_gyro_x_rms) <= 0.005
        assert np.degrees(bias_gyro_y_rms) <= 0.005
        assert np.degrees(bias_gyro_z_rms) <= 0.005
