import numpy as np
import pytest
from scipy.signal import resample_poly

import smsfusion as sf
from smsfusion._ins._ains_ import (
    AINS,
    _measurement_matrix_init,
    _process_noise_covariance_matrix,
    _reset,
    _state_transition_matrix_init,
    _state_transition_matrix_update,
)
from smsfusion._transforms import _rot_matrix_from_quaternion
from smsfusion._vectorops import _skew_symmetric
from smsfusion.benchmark import (
    benchmark_full_pva_beat_202311A,
    benchmark_full_pva_chirp_202311A,
)


def test_state_transition_matrix_init():
    dt = 0.1
    dvel = np.ones(3) * 0.01
    dtheta = np.ones(3) * 0.02
    R_nb = np.eye(3)
    gbc = 0.01

    phi_out = _state_transition_matrix_init(dt, dvel, dtheta, R_nb, gbc)
    phi_expected = np.array(
        [
            [1.0, 0.0, 0.0, dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.01, -0.01, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.01, 0.0, 0.01, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.01, -0.01, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.02, -0.02, -dt, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02, 1.0, 0.02, 0.0, -dt, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, -0.02, 1.0, 0.0, 0.0, -dt],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 - dt / gbc, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 - dt / gbc, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 - dt / gbc],
        ]
    )

    np.testing.assert_almost_equal(phi_out, phi_expected)


def test_state_transition_matrix_update():
    dt = 0.1
    dvel = np.ones(3) * 0.01
    dtheta = np.ones(3) * 0.02
    R_nb = np.eye(3)
    gbc = 0.01

    phi = _state_transition_matrix_init(dt, dvel, dtheta, R_nb, gbc)

    dtheta_update = np.ones(3) * 0.01
    dvel_update = np.ones(3) * 0.1
    _state_transition_matrix_update(
        phi, dvel=dvel_update, dtheta=dtheta_update, R_nb=R_nb
    )

    phi_expected = np.array(
        [
            [1.0, 0.0, 0.0, dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1, -0.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1, -0.1, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.01, -0.01, -dt, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01, 1.0, 0.01, 0.0, -dt, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, -0.01, 1.0, 0.0, 0.0, -dt],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 - dt / gbc, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 - dt / gbc, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 - dt / gbc],
        ]
    )

    np.testing.assert_almost_equal(phi, phi_expected)


def test_measurement_matrix_init():
    q_nb = np.array([1.0, 0.0, 0.0, 0.0])
    lever_arm = np.array([2.0, 3.0, 4.0])

    expect = np.zeros((7, 12))
    expect[0:3, 0:3] = np.eye(3)
    expect[0:3, 6:9] = -_rot_matrix_from_quaternion(q_nb) @ _skew_symmetric(lever_arm)
    expect[3:6, 3:6] = np.eye(3)
    expect[6, 6:9] = np.array([0.0, 0.0, 1.0])  # kappa -> zero due to unit quat

    np.testing.assert_array_equal(_measurement_matrix_init(q_nb, lever_arm), expect)


def test_process_noise_covariance_matrix():
    dt = 0.1
    vrw = 0.0005
    arw = 0.00005
    gbs = 0.00005
    gbc = 50.0
    Q_out = _process_noise_covariance_matrix(dt, vrw, arw, gbs, gbc)
    Q_expect = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, dt * vrw**2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, dt * vrw**2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, dt * vrw**2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dt * arw**2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dt * arw**2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dt * arw**2, 0.0, 0.0, 0.0],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                dt * (2.0 * gbs**2 / gbc),
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                dt * (2.0 * gbs**2 / gbc),
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                dt * (2.0 * gbs**2 / gbc),
            ],
        ]
    )

    np.testing.assert_allclose(Q_out, Q_expect)


def test_reset():
    p_n = np.array([1.0, 0.0, 0.0])
    v_n = np.array([0.0, 2.0, 0.0])
    q_nb = np.array([1.0, 0.0, 0.0, 0.0])
    bg_b = np.zeros(3)
    dx = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.01, 0.0, 0.0, 0.1, -0.1, 0.2])

    _reset(dx, p_n, v_n, q_nb, bg_b)

    np.testing.assert_allclose(dx, np.zeros_like(dx))
    np.testing.assert_allclose(p_n, np.array([1.1, 0.2, 0.3]))
    np.testing.assert_allclose(v_n, np.array([0.4, 2.5, 0.6]))
    np.testing.assert_allclose(bg_b, np.array([0.1, -0.1, 0.2]))
    np.testing.assert_allclose(
        q_nb, np.array([np.cos(0.01 / 2), np.sin(0.01 / 2), 0.0, 0.0]), atol=1e-6
    )


def test_ains_init():
    mekf = AINS(10.0)
    np.testing.assert_allclose(mekf.position(), np.zeros(3))
    np.testing.assert_allclose(mekf.velocity(), np.zeros(3))
    np.testing.assert_allclose(mekf.quaternion(), np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(mekf.bias_gyro(), np.zeros(3))
    np.testing.assert_allclose(mekf.P, np.array(sf._ins._ains.P0))
    assert mekf._g == 9.80665
    assert mekf._nav_frame == "ned"


@pytest.mark.parametrize("nav_frame, scale", (["NED", 1.0], ["ENU", -1.0]))
def test_ains_nav_frame(nav_frame, scale):
    mekf = AINS(10.0, nav_frame=nav_frame)

    assert mekf._nav_frame == nav_frame.lower()
    np.testing.assert_allclose(mekf._g_n, np.array([0.0, 0.0, mekf._g * scale]))


def test_ains_methods():
    pos_init = np.array([0.1, 10.0, -0.2])
    vel_init = np.array([0.0, 0.1, -0.2])
    euler_init = np.array([10.0, 20.0, 30.0])
    quaternion_init = sf.quaternion_from_euler(euler_init, degrees=True)
    bg_init = np.array([0.01, -0.01, 0.02])

    mekf = AINS(10.0, p0=pos_init, v0=vel_init, q0=quaternion_init, bg0=bg_init)

    np.testing.assert_allclose(mekf.position(), pos_init)
    np.testing.assert_allclose(mekf.velocity(), vel_init)
    np.testing.assert_allclose(mekf.euler(), np.radians(euler_init))
    np.testing.assert_allclose(mekf.euler(degrees=True), euler_init)
    np.testing.assert_allclose(mekf.quaternion(), quaternion_init)
    np.testing.assert_allclose(mekf.bias_gyro(), bg_init)
    np.testing.assert_allclose(mekf.bias_gyro(degrees=True), np.degrees(bg_init))


@pytest.mark.parametrize(
    "benchmark_gen, gyro_degrees",
    [
        (benchmark_full_pva_beat_202311A, False),
        (benchmark_full_pva_chirp_202311A, True),
    ],
)
def test_ains_benchmark(benchmark_gen, gyro_degrees):
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
    bg = np.array([0.01, -0.02, 0.0])
    imu_noise = noise_model(fs_imu, len(t))
    acc_imu = acc_ref + imu_noise[:, :3]
    gyro_imu = gyro_ref + imu_noise[:, 3:] + bg
    pos_meas = pos_ref + np.random.normal(0.0, pos_std, pos_ref.shape)
    vel_meas = vel_ref + np.random.normal(0.0, vel_std, vel_ref.shape)
    head_meas = euler_ref[:, 2] + np.random.normal(0.0, head_std, len(euler_ref))

    if gyro_degrees:
        gyro_imu = np.degrees(gyro_imu)

    # MEKF
    mekf = AINS(
        fs_imu,
        p0=pos_ref[0],
        v0=vel_ref[0],
        q0=sf.quaternion_from_euler(euler_ref[0], degrees=False),
        gyro_noise_density=err_gyro["N"],
        gyro_bias_stability=err_gyro["B"],
        gyro_bias_corr_time=err_gyro["tau_cb"],
    )

    pos_est, vel_est, euler_est, bias_gyro_est = [], [], [], []
    for f_i, w_i, h_i, p_i, v_i in zip(
        acc_imu, gyro_imu, head_meas, pos_meas, vel_meas
    ):

        dvel_i = f_i / fs_imu
        dtheta_i = w_i / fs_imu

        mekf.update(
            dvel_i,
            dtheta_i,
            degrees=gyro_degrees,
            head=h_i,
            head_var=head_std**2,
            head_degrees=False,
            pos=p_i,
            pos_var=pos_std**2 * np.ones(3),
            vel=v_i,
            vel_var=vel_std**2 * np.ones(3),
        )
        pos_est.append(mekf.position())
        vel_est.append(mekf.velocity())
        euler_est.append(mekf.euler(degrees=False))
        bias_gyro_est.append(mekf.bias_gyro())

    pos_est = np.array(pos_est)
    vel_est = np.array(vel_est)
    euler_est = np.array(euler_est)
    bias_gyro_est = np.array(bias_gyro_est)

    # Half-sample shift (compensates for the delay introduced by Euler integration)
    pos_est = resample_poly(pos_est, 2, 1)[1:-1:2]
    vel_est = resample_poly(vel_est, 2, 1)[1:-1:2]
    euler_est = resample_poly(euler_est, 2, 1)[1:-1:2]
    bias_gyro_est = resample_poly(bias_gyro_est, 2, 1)[1:-1:2]
    pos_ref = pos_ref[:-1, :]
    vel_ref = vel_ref[:-1, :]
    euler_ref = euler_ref[:-1, :]
    bias_gyro_ref = np.tile(bg, (len(bias_gyro_est), 1))

    def rmse(ref, est):
        return np.sqrt(np.mean((ref - est) ** 2, axis=0))

    px_rmse, py_rmse, pz_rmse = rmse(pos_ref[warmup:], pos_est[warmup:])
    vx_rmse, vy_rmse, vz_rmse = rmse(vel_ref[warmup:], vel_est[warmup:])
    roll_rmse, pitch_rmse, yaw_rmse = rmse(euler_ref[warmup:], euler_est[warmup:])
    bgx_rmse, bgy_rmse, bgz_rmse = rmse(bias_gyro_ref[warmup:], bias_gyro_est[warmup:])

    assert px_rmse <= 0.1
    assert py_rmse <= 0.1
    assert pz_rmse <= 0.1
    assert vx_rmse <= 0.1
    assert vy_rmse <= 0.1
    assert vz_rmse <= 0.1
    assert np.degrees(roll_rmse) <= 0.5
    assert np.degrees(pitch_rmse) <= 0.5
    assert np.degrees(yaw_rmse) <= 0.5
    assert np.degrees(bgx_rmse) <= 0.01
    assert np.degrees(bgy_rmse) <= 0.01
    assert np.degrees(bgz_rmse) <= 0.01
