import numpy as np
import pytest
from scipy.signal import resample_poly

import smsfusion as sf
from smsfusion._ins._vru import VRU, _state_transition_matrix_init, _state_transition_matrix_update, _measurement_matrix_init, _reset, _process_noise_covariance_matrix
from smsfusion.benchmark import (
    benchmark_pure_attitude_beat_202311A,
    benchmark_pure_attitude_chirp_202311A
)


def test_state_transition_matrix_init():
    dt = 0.1
    dtheta = np.ones(3) * 0.02
    gbc = 0.01

    phi_out = _state_transition_matrix_init(dt, dtheta, gbc)
    phi_expected = np.array([
        [1.0, 0.02, -0.02, -dt, 0.0, 0.0],
        [-0.02, 1.0, 0.02, 0.0, -dt, 0.0],
        [0.02, -0.02, 1.0, 0.0, 0.0, -dt],
        [0.0, 0.0, 0.0, 1.0 - dt / gbc, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0 - dt / gbc, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 - dt / gbc],
    ])

    np.testing.assert_almost_equal(phi_out, phi_expected)


def test_state_transition_matrix_update():
    dt = 0.1
    dtheta = np.ones(3) * 0.02
    gbc = 0.01

    phi_init = _state_transition_matrix_init(dt, dtheta, gbc)

    dtheta_update = np.ones(3) * 0.01
    phi_out = _state_transition_matrix_update(phi_init, dtheta=dtheta_update)

    phi_expected = np.array([
        [1.0, 0.01, -0.01, -dt, 0.0, 0.0],
        [-0.01, 1.0, 0.01, 0.0, -dt, 0.0],
        [0.01, -0.01, 1.0, 0.0, 0.0, -dt],
        [0.0, 0.0, 0.0, 1.0 - dt / gbc, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0 - dt / gbc, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 - dt / gbc],
    ])

    np.testing.assert_almost_equal(phi_out, phi_expected)


def test_measurement_matrix_init():
    np.testing.assert_array_equal(_measurement_matrix_init(), np.zeros((3, 6)))


def test_process_noise_covariance_matrix():
    dt = 0.1
    arw = 0.00005
    gbs = 0.00005
    gbc = 50.0
    Q_out = _process_noise_covariance_matrix(dt, arw, gbs, gbc)
    Q_expect = np.array([
        [dt * arw**2, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, dt * arw**2, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, dt * arw**2, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, dt * (2.0 * gbs**2 / gbc), 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, dt * (2.0 * gbs**2 / gbc), 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, dt * (2.0 * gbs**2 / gbc)],
    ])

    np.testing.assert_allclose(Q_out, Q_expect)

def test_reset():
    q_nb = np.array([1.0, 0.0, 0.0, 0.0])
    bg_b = np.zeros(3)
    dx = np.array([0.01, 0.0, 0.0, 0.1, -0.1, 0.2])

    dx, q_nb, bg_b = _reset(dx, q_nb, bg_b)

    np.testing.assert_allclose(dx, np.zeros_like(dx))
    np.testing.assert_allclose(bg_b, np.array([0.1, -0.1, 0.2]))
    np.testing.assert_allclose(q_nb, np.array([np.cos(0.01/2), np.sin(0.01/2), 0.0, 0.0]), atol=1e-6)


def test_vru_init():
    mekf = VRU(
        10.0
        )

    np.testing.assert_allclose(mekf.quaternion(), np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(mekf.bias_gyro(), np.zeros(3))
    np.testing.assert_allclose(mekf.P, np.array(sf._ins._vru.P0))


@pytest.mark.parametrize("nav_frame, scale", (["NED", 1.0], ["ENU", -1.0]))
def test_vru_nav_frame(nav_frame, scale):
    mekf = VRU(
        10.0,
        nav_frame=nav_frame
        )

    assert mekf._nz2vg == scale


def test_vru_methods():
    euler_init = np.array([10.0, 20.0, 30.0])
    quaternion_init = sf.quaternion_from_euler(euler_init, degrees=True)
    bg_init = np.array([0.01, -0.01, 0.02])

    mekf = VRU(
        10.0,
        q=quaternion_init,
        bg=bg_init
        )

    np.testing.assert_allclose(mekf.euler(), np.radians(euler_init))
    np.testing.assert_allclose(mekf.euler(degrees=True), euler_init)
    np.testing.assert_allclose(mekf.quaternion(), quaternion_init)
    np.testing.assert_allclose(mekf.bias_gyro(), bg_init)
    np.testing.assert_allclose(mekf.bias_gyro(degrees=True), np.degrees(bg_init))


@pytest.mark.parametrize(
    "benchmark_gen, degrees",
    [(benchmark_pure_attitude_beat_202311A, False), (benchmark_pure_attitude_chirp_202311A, True)],
)
def test_vru_benchmark(benchmark_gen, degrees):
    fs_imu = 100.0
    warmup = int(fs_imu * 600.0)  # truncate 600 seconds from the beginning

    # Reference signals (without noise)
    t, euler_ref, acc_ref, gyro_ref = benchmark_gen(fs_imu)

    # IMU measurements (with noise)
    bg = np.array([0.01, -0.02, 0.0])
    noise_model = sf.noise.IMUNoise(
        err_acc=sf.constants.ERR_ACC_MOTION2,
        err_gyro=sf.constants.ERR_GYRO_MOTION2,
        seed=0,
    )
    imu_noise = noise_model(fs_imu, len(t))
    acc_noise = acc_ref + imu_noise[:, :3]
    gyro_noise = gyro_ref + imu_noise[:, 3:] + bg

    if degrees:
        gyro_noise = np.degrees(gyro_noise)

    # MEKF
    q0 = sf.quaternion_from_euler(euler_ref[0], degrees=False)
    mekf = VRU(
        fs_imu,
        q=q0,
        gyro_noise_density=sf.constants.ERR_GYRO_MOTION2["N"],
        gyro_bias_stability=sf.constants.ERR_GYRO_MOTION2["B"],
        gyro_bias_corr_time=sf.constants.ERR_GYRO_MOTION2["tau_cb"],
    )

    # Apply filter
    euler_out, bias_gyro_out = [], []
    for i, (f_i, w_i) in enumerate(
        zip(acc_noise, gyro_noise)
    ):

        dvel = f_i / fs_imu
        dtheta = w_i / fs_imu

        mekf.update(
            dvel,
            dtheta,
            degrees=degrees,
            gref=True
        )

        euler_out.append(mekf.euler(degrees=False))
        bias_gyro_out.append(mekf.bias_gyro(degrees=False))

    euler_out = np.array(euler_out)
    bias_gyro_out = np.array(bias_gyro_out)

    # Half-sample shift (compensates for the delay introduced by Euler integration)
    euler_out = resample_poly(euler_out, 2, 1)[1:-1:2]
    euler_ref = euler_ref[:-1, :]

    roll_rms, pitch_rms, yaw_rms = np.std((euler_out - euler_ref)[warmup:], axis=0)
    bias_gyro_x_rms, bias_gyro_y_rms, bias_gyro_z_rms = np.std(
        (bias_gyro_out - bg)[warmup:], axis=0
    )

    assert np.degrees(roll_rms) <= 0.1
    assert np.degrees(pitch_rms) <= 0.1
    assert np.degrees(bias_gyro_x_rms) <= 0.005
    assert np.degrees(bias_gyro_y_rms) <= 0.005
