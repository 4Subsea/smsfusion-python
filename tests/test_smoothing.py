import numpy as np
import pytest
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

    def test_benchmark_ains(self):
        # Reference signal
        fs = 10.24  # sampling rate in Hz
        _, pos, vel, euler, acc, gyro = benchmark_full_pva_beat_202311A(fs)
        euler = np.degrees(euler)
        gyro = np.degrees(gyro)
        head = euler[:, 2]

        # IMU measurements
        err_acc = sf.constants.ERR_ACC_MOTION2  # m/s^2
        err_gyro = sf.constants.ERR_GYRO_MOTION2  # rad/s
        imu_noise = sf.noise.IMUNoise(err_acc, err_gyro)(fs, len(acc))
        acc_imu = acc + imu_noise[:, :3]
        gyro_imu = gyro + np.degrees(imu_noise[:, 3:])

        # Aiding measurements
        pos_noise_std = 0.1  # m
        head_noise_std = 1.0  # deg
        rng = np.random.default_rng(0)
        pos_aid = pos + pos_noise_std * rng.standard_normal(pos.shape)
        head_aid = head + head_noise_std * rng.standard_normal(head.shape)

        # AINS
        p0 = pos[0]  # position [m]
        v0 = vel[0]  # velocity [m/s]
        q0 = sf.quaternion_from_euler(euler[0], degrees=True)  # unit quaternion
        ba0 = np.zeros(3)  # accelerometer bias [m/s^2]
        bg0 = np.zeros(3)  # gyroscope bias [rad/s]
        x0 = np.concatenate((p0, v0, q0, ba0, bg0))
        P0 = np.eye(12) * 1e-3
        err_acc = sf.constants.ERR_ACC_MOTION2  # m/s^2
        err_gyro = sf.constants.ERR_GYRO_MOTION2  # rad/s
        ains = sf.AidedINS(fs, x0, P0, err_acc, err_gyro)

        smoother = sf.FixedIntervalSmoother(ains, cov_smoothing=True)

        pos_ains, vel_ains, euler_ains = [], [], []
        for f_i, w_i, p_i, h_i in zip(acc_imu, gyro_imu, pos_aid, head_aid):
            smoother.update(
                f_i,
                w_i,
                degrees=True,
                pos=p_i,
                pos_var=pos_noise_std**2 * np.ones(3),
                head=h_i,
                head_var=head_noise_std**2,
                head_degrees=True,
            )

            pos_ains.append(smoother.ains.position())
            vel_ains.append(smoother.ains.velocity())
            euler_ains.append(smoother.ains.euler(degrees=True))

        # Forward filter state estimates
        pos_ains = np.array(pos_ains)
        vel_ains = np.array(vel_ains)
        euler_ains = np.array(euler_ains)

        # Smoothed state estimates
        pos_smth = smoother.position()
        vel_smth = smoother.velocity()
        euler_smth = smoother.euler(degrees=True)

        # Half-sample shift
        # # (compensates for the delay introduced by Euler integration)
        pos_ains = resample_poly(pos_ains, 2, 1)[1:-1:2]
        pos_smth = resample_poly(pos_smth, 2, 1)[1:-1:2]
        pos_ref = pos[:-1, :]
        vel_ains = resample_poly(vel_ains, 2, 1)[1:-1:2]
        vel_smth = resample_poly(vel_smth, 2, 1)[1:-1:2]
        vel_ref = vel[:-1, :]
        euler_ains = resample_poly(euler_ains, 2, 1)[1:-1:2]
        euler_smth = resample_poly(euler_smth, 2, 1)[1:-1:2]
        euler_ref = euler[:-1, :]

        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning

        pos_err_smth = np.std((pos_smth - pos_ref)[warmup:], axis=0)
        pos_err_ains = np.std((pos_ains - pos_ref)[warmup:], axis=0)
        np.testing.assert_array_less(pos_err_smth, pos_err_ains)

        vel_err_smth = np.std((vel_smth - vel_ref)[warmup:], axis=0)
        vel_err_ains = np.std((vel_ains - vel_ref)[warmup:], axis=0)
        np.testing.assert_array_less(vel_err_smth, vel_err_ains)

        euler_err_smth = np.std((euler_smth - euler_ref)[warmup:], axis=0)
        euler_err_ains = np.std((euler_ains - euler_ref)[warmup:], axis=0)
        np.testing.assert_array_less(euler_err_smth, euler_err_ains)

    def test_benchmark_ahrs(self):
        # Reference signal
        fs = 10.24  # sampling rate in Hz
        _, pos, vel, euler, acc, gyro = benchmark_full_pva_beat_202311A(fs)
        euler = np.degrees(euler)
        gyro = np.degrees(gyro)
        head = euler[:, 2]

        # IMU measurements
        err_acc = sf.constants.ERR_ACC_MOTION2  # m/s^2
        err_gyro = sf.constants.ERR_GYRO_MOTION2  # rad/s
        imu_noise = sf.noise.IMUNoise(err_acc, err_gyro)(fs, len(acc))
        acc_imu = acc + imu_noise[:, :3]
        gyro_imu = gyro + np.degrees(imu_noise[:, 3:])

        # Aiding measurements
        head_noise_std = 1.0  # deg
        rng = np.random.default_rng(0)
        head_aid = head + head_noise_std * rng.standard_normal(head.shape)

        # AINS
        p0 = pos[0]  # position [m]
        v0 = vel[0]  # velocity [m/s]
        q0 = sf.quaternion_from_euler(euler[0], degrees=True)  # unit quaternion
        ba0 = np.zeros(3)  # accelerometer bias [m/s^2]
        bg0 = np.zeros(3)  # gyroscope bias [rad/s]
        x0 = np.concatenate((p0, v0, q0, ba0, bg0))
        P0 = np.eye(12) * 1e-3
        err_acc = sf.constants.ERR_ACC_MOTION2  # m/s^2
        err_gyro = sf.constants.ERR_GYRO_MOTION2  # rad/s
        ains = sf.AHRS(fs, x0, P0, err_acc, err_gyro)

        smoother = sf.FixedIntervalSmoother(ains, cov_smoothing=True)

        euler_ains, err_ains = [], []
        for f_i, w_i, h_i in zip(acc_imu, gyro_imu, head_aid):
            smoother.update(
                f_i,
                w_i,
                degrees=True,
                head=h_i,
                head_var=head_noise_std**2,
                head_degrees=True,
            )

            euler_ains.append(smoother.ains.euler(degrees=True))
            err_ains.append(smoother.ains.P.diagonal())

        # Forward filter state estimates
        euler_ains = np.array(euler_ains)
        err_ains = np.array(err_ains)

        # Smoothed state estimates
        euler_smth = smoother.euler(degrees=True)
        err_smth = np.array([P_i.diagonal() for P_i in smoother.P])

        # Half-sample shift
        # # (compensates for the delay introduced by Euler integration)
        euler_ains = resample_poly(euler_ains, 2, 1)[1:-1:2]
        euler_smth = resample_poly(euler_smth, 2, 1)[1:-1:2]
        euler = euler[:-1, :]

        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning

        euler_err_smth = np.std((euler_smth - euler)[warmup:], axis=0)
        euler_err_ains = np.std((euler_ains - euler)[warmup:], axis=0)
        np.testing.assert_array_less(euler_err_smth, euler_err_ains)

        smoother.P.shape == (len(acc_imu), 12, 12)
        np.testing.assert_array_less(err_smth[warmup:], err_ains[warmup:] + 1e-12)

    def test_benchmark_vru(self):
        # Reference signal
        fs = 10.24  # sampling rate in Hz
        _, pos, vel, euler, acc, gyro = benchmark_full_pva_beat_202311A(fs)
        euler = np.degrees(euler)
        gyro = np.degrees(gyro)

        # IMU measurements
        err_acc = sf.constants.ERR_ACC_MOTION2  # m/s^2
        err_gyro = sf.constants.ERR_GYRO_MOTION2  # rad/s
        imu_noise = sf.noise.IMUNoise(err_acc, err_gyro)(fs, len(acc))
        acc_imu = acc + imu_noise[:, :3]
        gyro_imu = gyro + np.degrees(imu_noise[:, 3:])

        # AINS
        p0 = pos[0]  # position [m]
        v0 = vel[0]  # velocity [m/s]
        q0 = sf.quaternion_from_euler(euler[0], degrees=True)  # unit quaternion
        ba0 = np.zeros(3)  # accelerometer bias [m/s^2]
        bg0 = np.zeros(3)  # gyroscope bias [rad/s]
        x0 = np.concatenate((p0, v0, q0, ba0, bg0))
        P0 = np.eye(12) * 1e-3
        err_acc = sf.constants.ERR_ACC_MOTION2  # m/s^2
        err_gyro = sf.constants.ERR_GYRO_MOTION2  # rad/s
        ains = sf.VRU(fs, x0, P0, err_acc, err_gyro)

        smoother = sf.FixedIntervalSmoother(ains, cov_smoothing=True)

        euler_ains, err_ains = [], []
        for f_i, w_i in zip(acc_imu, gyro_imu):
            smoother.update(
                f_i,
                w_i,
                degrees=True,
            )

            euler_ains.append(smoother.ains.euler(degrees=True))
            err_ains.append(smoother.ains.P.diagonal())

        # Forward filter state estimates
        euler_ains = np.array(euler_ains)
        err_ains = np.array(err_ains)

        # Smoothed state estimates
        euler_smth = smoother.euler(degrees=True)
        err_smth = np.array([P_i.diagonal() for P_i in smoother.P])

        # Half-sample shift
        # # (compensates for the delay introduced by Euler integration)
        euler_ains = resample_poly(euler_ains, 2, 1)[1:-1:2]
        euler_smth = resample_poly(euler_smth, 2, 1)[1:-1:2]
        euler = euler[:-1, :]

        # Drop yaw
        euler = euler[:, :2]
        euler_ains = euler_ains[:, :2]
        euler_smth = euler_smth[:, :2]

        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning

        euler_err_smth = np.std((euler_smth - euler)[warmup:], axis=0)
        euler_err_ains = np.std((euler_ains - euler)[warmup:], axis=0)
        np.testing.assert_array_less(euler_err_smth, euler_err_ains)

        smoother.P.shape == (len(acc_imu), 12, 12)
        np.testing.assert_array_less(err_smth[warmup:], err_ains[warmup:] + 1e-12)
