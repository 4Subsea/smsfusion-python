import numpy as np
import pytest
from scipy.signal import resample_poly

import smsfusion as sf
from smsfusion import PVAMEKF
from smsfusion._ins._smoothing import FixedIntervalSmoother
from smsfusion.benchmark import (
    benchmark_full_pva_beat_202311A,
    benchmark_full_pva_chirp_202311A,
)


class Test_FixedIntervalSmoother:

    @pytest.mark.parametrize(
        "benchmark_gen",
        [
            benchmark_full_pva_beat_202311A,
            benchmark_full_pva_chirp_202311A,
        ],
    )
    def test_benchmark_default_aiding(self, benchmark_gen):
        fs_imu = 10.0
        warmup = int(fs_imu * 600.0)  # truncate 600 seconds from the beginning

        # Reference signals (without noise)
        t, _, _, euler_ref, acc_ref, gyro_ref = benchmark_gen(fs_imu)

        # IMU and aiding measurements (with noise)
        head_std = np.radians(0.1)  # rad
        err_acc = sf.constants.ERR_ACC_MOTION2
        err_gyro = sf.constants.ERR_GYRO_MOTION2
        noise_model = sf.noise.IMUNoise(err_acc=err_acc, err_gyro=err_gyro, seed=0)
        bg = np.array([0.01, -0.02, 0.0])  # rad/s
        imu_noise = noise_model(fs_imu, len(t))
        acc_meas = acc_ref + imu_noise[:, :3]
        gyro_meas = gyro_ref + imu_noise[:, 3:] + bg
        head_meas = euler_ref[:, 2] + np.random.normal(0.0, head_std, len(euler_ref))

        # MEKF
        q0 = sf.quaternion_from_euler(euler_ref[0], degrees=False)
        mekf = PVAMEKF(fs_imu, q0=q0)
        smoother = FixedIntervalSmoother(PVAMEKF(fs_imu, q0=q0))

        euler_fwd, euler_smth = [], []
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

        rmse_fwd = rmse(euler_ref[warmup:], euler_fwd[warmup:])
        rmse_smth = rmse(euler_ref[warmup:], euler_smth[warmup:])

        assert np.all(rmse_smth < rmse_fwd)
