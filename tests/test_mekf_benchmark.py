import numpy as np
import pytest

from smsfusion import MEKF
from smsfusion._transforms import _quaternion_from_euler
from smsfusion.benchmark import (
    benchmark_full_pva_beat_202311A,
    benchmark_full_pva_chirp_202311A,
)
from smsfusion.noise import IMUNoise, white_noise


class Test_MEKF:
    @pytest.mark.parametrize(
        "benchmark_gen",
        [benchmark_full_pva_beat_202311A, benchmark_full_pva_chirp_202311A],
    )
    def test_benchmark(self, benchmark_gen):
        fs_imu = 100.0
        fs_aiding = 1.0
        fs_ratio = np.ceil(fs_imu / fs_aiding)
        warmup = int(fs_imu * 600.0)  # truncate 600 seconds from the beginning
        compass_noise_std = 0.5
        gps_noise_std = 0.1
        vel_noise_std = 0.1

        # Reference signals (without noise)
        t, pos_ref, vel_ref, euler_ref, acc_ref, gyro_ref = benchmark_gen(fs_imu)
        euler_ref = np.degrees(euler_ref)
        gyro_ref = np.degrees(gyro_ref)

        # IMU measurements (with noise)
        err_acc_true = {
            "bc": (0.0, 0.0, 0.0),
            "N": (4.0e-4, 4.0e-4, 4.5e-4),
            "B": (1.5e-4, 1.5e-4, 3.0e-4),
            "K": (4.5e-6, 4.5e-6, 1.5e-5),
            "tau_cb": (50, 50, 30),
            "tau_ck": (5e5, 5e5, 5e5),
        }
        err_gyro_true = {
            "bc": (0.1, 0.2, 0.3),
            "N": (1.9e-3, 1.9e-3, 1.7e-3),
            "B": (7.5e-4, 4.0e-4, 8.8e-4),
            "K": (2.5e-5, 2.5e-5, 4.0e-5),
            "tau_cb": (50, 50, 50),
            "tau_ck": (5e5, 5e5, 5e5),
        }
        noise_model = IMUNoise(acc_err=err_acc_true, gyro_err=err_gyro_true, seed=0)
        imu_noise = noise_model(fs_imu, len(t))
        acc_noise = acc_ref + imu_noise[:, :3]
        gyro_noise = gyro_ref + imu_noise[:, 3:]

        # Compass / heading (aiding) measurements
        head_meas = euler_ref[:, 2] + white_noise(
            compass_noise_std / np.sqrt(fs_aiding), fs_aiding, len(t), seed=1
        )

        # GPS / position (aiding) measurements
        pos_noise = np.column_stack(
            [
                white_noise(
                    gps_noise_std / np.sqrt(fs_aiding), fs_aiding, len(t), seed=2
                ),
                white_noise(
                    gps_noise_std / np.sqrt(fs_aiding), fs_aiding, len(t), seed=3
                ),
                white_noise(
                    gps_noise_std / np.sqrt(fs_aiding), fs_aiding, len(t), seed=4
                ),
            ]
        )
        pos_meas = pos_ref + pos_noise

        # Velocity (aiding) measurements
        vel_noise = np.column_stack(
            [
                white_noise(
                    vel_noise_std / np.sqrt(fs_aiding), fs_aiding, len(t), seed=5
                ),
                white_noise(
                    vel_noise_std / np.sqrt(fs_aiding), fs_aiding, len(t), seed=6
                ),
                white_noise(
                    vel_noise_std / np.sqrt(fs_aiding), fs_aiding, len(t), seed=7
                ),
            ]
        )
        vel_meas = vel_ref + vel_noise

        # MEKF
        err_acc = {"N": 4.0e-4, "B": 1.5e-4, "K": 4.5e-6, "tau_cb": 50}
        err_gyro = {
            "N": (np.pi / 180.0) * 1.9e-3,
            "B": (np.pi / 180.0) * 7.5e-4,
            "tau_cb": 50,
        }
        var_pos = gps_noise_std**2 * np.ones(3)
        var_vel = vel_noise_std**2 * np.ones(3)
        var_compass = ((np.pi / 180.0) * compass_noise_std) ** 2
        var_g = 0.1**2 * np.ones(3)
        P_prior = np.eye(15)
        P_prior[9:12, 9:12] *= 1e-9
        x0 = np.zeros(16)
        x0[0:3] = pos_ref[0]
        x0[3:6] = vel_ref[0]
        x0[6:10] = _quaternion_from_euler(np.radians(euler_ref[0].flatten()))
        mekf = MEKF(
            fs_imu,
            x0,
            err_acc,
            err_gyro,
            var_pos,
            var_vel,
            var_g,
            var_compass,
            cov_error=P_prior,
        )

        # Apply filter
        pos_out, vel_out, euler_out, bias_acc_out, bias_gyro_out = [], [], [], [], []
        for i, (acc_i, gyro_i, pos_i, vel_i, head_i) in enumerate(
            zip(acc_noise, gyro_noise, pos_meas, vel_meas, head_meas)
        ):
            if not (i % fs_ratio):  # with aiding
                mekf.update(
                    acc_i,
                    gyro_i,
                    pos=pos_i,
                    vel=vel_i,
                    head=head_i,
                    degrees=True,
                    head_degrees=True,
                )
            else:  # without aiding
                mekf.update(acc_i, gyro_i, degrees=True, head_degrees=True)
            pos_out.append(mekf.position())
            vel_out.append(mekf.velocity())
            euler_out.append(mekf.euler(degrees=True))
            bias_acc_out.append(mekf.bias_acc())
            bias_gyro_out.append(mekf.bias_gyro(degrees=True))

        pos_out = np.array(pos_out)
        vel_out = np.array(vel_out)
        euler_out = np.array(euler_out)
        bias_acc_out = np.array(bias_acc_out)
        bias_gyro_out = np.array(bias_gyro_out)

        pos_x_rms, pos_y_rms, pos_z_rms = np.std((pos_out - pos_ref)[warmup:], axis=0)
        vel_x_rms, vel_y_rms, vel_z_rms = np.std((vel_out - vel_ref)[warmup:], axis=0)
        roll_rms, pitch_rms, yaw_rms = np.std((euler_out - euler_ref)[warmup:], axis=0)

        assert pos_x_rms <= 0.1
        assert pos_y_rms <= 0.1
        assert pos_z_rms <= 0.1
        assert vel_x_rms <= 0.02
        assert vel_y_rms <= 0.02
        assert vel_z_rms <= 0.02
        assert roll_rms <= 0.03
        assert pitch_rms <= 0.03
        assert yaw_rms <= 0.1
