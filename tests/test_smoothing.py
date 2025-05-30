import pytest

import numpy as np
from scipy.signal import resample_poly
import smsfusion as sf
from smsfusion import FixedIntervalSmoother
from smsfusion.benchmark import (
    benchmark_full_pva_beat_202311A,
    benchmark_full_pva_chirp_202311A,
)


class Test_FixedIntervalSmoother:
    @pytest.fixture
    def ains(self):
        x0 = np.zeros(16)
        x0[6:10] = np.array([1.0, 0.0, 0.0, 0.0])
        ains = sf.AidedINS(10.24, x0)
        return ains

    def test__init__(self, ains):
        smoother = FixedIntervalSmoother(ains)
        assert smoother._ains is ains
        assert smoother._cov_smoothing is True
        assert smoother.x.size == 0
        assert smoother.P.size == 0
        assert smoother.position().size == 0
        assert smoother.velocity().size == 0
        assert smoother.quaternion().size == 0
        assert smoother.bias_acc().size == 0
        assert smoother.bias_gyro().size == 0

    def test_benchmark(self):
        fs_imu = 10.0
        fs_aiding = 1.0
        fs_ratio = np.ceil(fs_imu / fs_aiding)
        warmup = int(fs_imu * 600.0)  # truncate 600 seconds from the beginning
        compass_noise_std = 0.5
        gps_noise_std = 0.1
        vel_noise_std = 0.1

        # Reference signals (without noise)
        t, pos_ref, vel_ref, euler_ref, acc_ref, gyro_ref = benchmark_full_pva_beat_202311A(fs_imu)
        euler_ref = np.degrees(euler_ref)
        gyro_ref = np.degrees(gyro_ref)

        rng = np.random.default_rng(seed=1)

        # IMU measurements (with noise)
        err_acc = sf.constants.ERR_ACC_MOTION2
        err_gyro = sf.constants.ERR_GYRO_MOTION2
        noise_model = sf.noise.IMUNoise(err_acc, err_gyro, seed=0)
        imu_noise = noise_model(fs_imu, len(t))
        acc_meas = acc_ref + imu_noise[:, :3]
        gyro_meas = gyro_ref + np.degrees(imu_noise[:, 3:])

        # Compass / heading (aiding) measurements
        head_noise = compass_noise_std * rng.standard_normal(len(t))
        head_meas = euler_ref[:, 2] + head_noise

        # GPS / position (aiding) measurements
        pos_noise = gps_noise_std * rng.standard_normal((len(t), 3))
        pos_meas = pos_ref + pos_noise

        # Velocity (aiding) measurements
        vel_noise = vel_noise_std * rng.standard_normal((len(t), 3))
        vel_meas = vel_ref + vel_noise

        # MEKF
        p0 = pos_ref[0]  # position [m]
        v0 = vel_ref[0]  # velocity [m/s]
        q0 = sf.quaternion_from_euler(euler_ref[0])  # attitude as unit quaternion
        ba0 = np.zeros(3)  # accelerometer bias [m/s^2]
        bg0 = np.zeros(3)  # gyroscope bias [rad/s]
        x0 = np.concatenate((p0, v0, q0, ba0, bg0))
        P0 = sf.constants.P0
        ains = sf.AidedINS(fs_imu, x0, P0, err_acc, err_gyro)

        smoother = FixedIntervalSmoother(ains, cov_smoothing=True)

        pos_ains, vel_ains, euler_ains, bias_acc_ains, bias_gyro_ains = [], [], [], [], []
        for i, (acc_i, gyro_i, pos_i, vel_i, head_i) in enumerate(
            zip(acc_meas, gyro_meas, pos_meas, vel_meas, head_meas)
        ):
            if not (i % fs_ratio):  # with aiding
                smoother.update(
                    acc_i,
                    gyro_i,
                    degrees=True,
                    pos=pos_i,
                    pos_var=gps_noise_std**2 * np.ones(3),
                    vel=vel_i,
                    vel_var=vel_noise_std**2 * np.ones(3),
                    head=head_i,
                    head_var=compass_noise_std**2,
                    head_degrees=True,
                    g_ref=True,
                    g_var=0.1**2 * np.ones(3),
                )
            else:  # without aiding
                smoother.update(acc_i, gyro_i, degrees=True)

            pos_ains.append(smoother.ains.position())
            vel_ains.append(smoother.ains.velocity())
            euler_ains.append(smoother.ains.euler(degrees=True))
            bias_acc_ains.append(smoother.ains.bias_acc())
            bias_gyro_ains.append(smoother.ains.bias_gyro(degrees=True))

        pos_ains = np.array(pos_ains)
        vel_ains = np.array(vel_ains)
        euler_ains = np.array(euler_ains)
        bias_acc_ains = np.array(bias_acc_ains)
        bias_gyro_ains = np.array(bias_gyro_ains)

        pos_smth = smoother.position()
        vel_smth = smoother.velocity()
        euler_smth = smoother.euler(degrees=True)
        bias_acc_smth = smoother.bias_acc()
        bias_gyro_smth = smoother.bias_gyro(degrees=True)

        pos_smth = np.array(pos_smth)
        vel_smth = np.array(vel_smth)
        euler_smth = np.array(euler_smth)
        bias_acc_smth = np.array(bias_acc_smth)
        bias_gyro_smth = np.array(bias_gyro_smth)

        # Half-sample shift (compensates for the delay introduced by Euler integration)
        pos_ains = resample_poly(pos_ains, 2, 1)[1:-1:2]
        pos_smth = resample_poly(pos_smth, 2, 1)[1:-1:2]
        pos_ref = pos_ref[:-1, :]
        vel_ains = resample_poly(vel_ains, 2, 1)[1:-1:2]
        vel_smth = resample_poly(vel_smth, 2, 1)[1:-1:2]
        vel_ref = vel_ref[:-1, :]
        euler_ains = resample_poly(euler_ains, 2, 1)[1:-1:2]
        euler_smth = resample_poly(euler_smth, 2, 1)[1:-1:2]
        euler_ref = euler_ref[:-1, :]

        pos_err_smth = np.std((pos_smth - pos_ref)[warmup:], axis=0)
        pos_err_ains = np.std((pos_ains - pos_ref)[warmup:], axis=0)
        assert pos_err_smth[0] < pos_err_ains[0]
