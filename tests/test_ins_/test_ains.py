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
    benchmark_pure_attitude_beat_202311A,
    benchmark_pure_attitude_chirp_202311A,
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

    phi_init = _state_transition_matrix_init(dt, dvel, dtheta, R_nb, gbc)

    dtheta_update = np.ones(3) * 0.01
    dvel_update = np.ones(3) * 0.1
    phi_out = _state_transition_matrix_update(
        phi_init, dvel=dvel_update, dtheta=dtheta_update, R_nb=R_nb
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

    np.testing.assert_almost_equal(phi_out, phi_expected)


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

    dx, p_n, v_n, q_nb, bg_b = _reset(dx, p_n, v_n, q_nb, bg_b)

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

    mekf = AINS(10.0, pos=pos_init, vel=vel_init, q=quaternion_init, bg=bg_init)

    np.testing.assert_allclose(mekf.position(), pos_init)
    np.testing.assert_allclose(mekf.velocity(), vel_init)
    np.testing.assert_allclose(mekf.euler(), np.radians(euler_init))
    np.testing.assert_allclose(mekf.euler(degrees=True), euler_init)
    np.testing.assert_allclose(mekf.quaternion(), quaternion_init)
    np.testing.assert_allclose(mekf.bias_gyro(), bg_init)
    np.testing.assert_allclose(mekf.bias_gyro(degrees=True), np.degrees(bg_init))


@pytest.mark.parametrize(
    "benchmark_gen, degrees",
    [
        (benchmark_full_pva_beat_202311A, False),
        # (benchmark_full_pva_chirp_202311A, True),
    ],
)
def test_ains_benchmark(benchmark_gen, degrees):
    fs_imu = 10.0
    warmup = int(fs_imu * 600.0)  # truncate 600 seconds from the beginning

    # Reference signals (without noise)
    t, pos_ref, vel_ref, euler_ref, acc_ref, gyro_ref = benchmark_gen(fs_imu)

    # IMU and aiding measurements (with noise)
    pos_std = 0.1  # m
    vel_std = 0.01  # m/s
    head_std = np.radians(1.0)  # rad
    err_acc = sf.constants.ERR_ACC_MOTION2
    err_gyro = sf.constants.ERR_GYRO_MOTION2
    noise_model = sf.noise.IMUNoise(err_acc=err_acc, err_gyro=err_gyro, seed=0)
    bg = np.array([0.01, -0.02, 0.0])
    imu_noise = noise_model(fs_imu, len(t))
    acc_imu = acc_ref + imu_noise[:, :3]
    gyro_imu = gyro_ref + imu_noise[:, 3:] + bg
    pos_aid = pos_ref + np.random.normal(0.0, pos_std, pos_ref.shape)
    vel_aid = vel_ref + np.random.normal(0.0, vel_std, vel_ref.shape)
    head_aid = euler_ref[:, 2] + np.random.normal(0.0, head_std, len(euler_ref))

    if degrees:
        gyro_imu = np.degrees(gyro_imu)

    # Position and velocity aiding measurements

    # MEKF
    q0 = sf.quaternion_from_euler(euler_ref[0], degrees=False)
    mekf = AINS(
        fs_imu,
        q=q0,
        gyro_noise_density=sf.constants.ERR_GYRO_MOTION2["N"],
        gyro_bias_stability=sf.constants.ERR_GYRO_MOTION2["B"],
        gyro_bias_corr_time=sf.constants.ERR_GYRO_MOTION2["tau_cb"],
    )

    for i, (f_i, w_i, h_i, p_i, v_i) in enumerate(
        zip(acc_imu, gyro_imu, head_aid, pos_aid, vel_aid)
    ):

        dvel_i = f_i / fs_imu
        dtheta_i = w_i / fs_imu

        mekf.update(
            dvel_i,
            dtheta_i,
            degrees=degrees,
            head=h_i,
            head_var=head_std**2,
            pos=p_i,
            pos_var=pos_std**2 * np.ones(3),
            vel=v_i,
            vel_var=vel_std**2 * np.ones(3),
        )


# @pytest.mark.parametrize(
#     "benchmark_gen, degrees",
#     [
#         (benchmark_pure_attitude_beat_202311A, False),
#         (benchmark_pure_attitude_chirp_202311A, True),
#     ],
# )
# def test_ains_no_head_aiding_benchmark(benchmark_gen, degrees):
#     fs_imu = 100.0
#     warmup = int(fs_imu * 600.0)  # truncate 600 seconds from the beginning

#     # Reference signals (without noise)
#     t, euler_ref, acc_ref, gyro_ref = benchmark_gen(fs_imu)

#     # IMU measurements (with noise)
#     bg = np.array([0.01, -0.02, 0.0])
#     noise_model = sf.noise.IMUNoise(
#         err_acc=sf.constants.ERR_ACC_MOTION2,
#         err_gyro=sf.constants.ERR_GYRO_MOTION2,
#         seed=0,
#     )
#     imu_noise = noise_model(fs_imu, len(t))
#     acc_noise = acc_ref + imu_noise[:, :3]
#     gyro_noise = gyro_ref + imu_noise[:, 3:] + bg

#     if degrees:
#         gyro_noise = np.degrees(gyro_noise)

#     # MEKF
#     q0 = sf.quaternion_from_euler(euler_ref[0], degrees=False)
#     mekf = AINS(
#         fs_imu,
#         q=q0,
#         gyro_noise_density=sf.constants.ERR_GYRO_MOTION2["N"],
#         gyro_bias_stability=sf.constants.ERR_GYRO_MOTION2["B"],
#         gyro_bias_corr_time=sf.constants.ERR_GYRO_MOTION2["tau_cb"],
#     )

#     # Apply filter
#     euler_out, bias_gyro_out = [], []
#     for i, (f_i, w_i) in enumerate(zip(acc_noise, gyro_noise)):

#         dvel = f_i / fs_imu
#         dtheta = w_i / fs_imu

#         mekf.update(
#             dvel,
#             dtheta,
#             degrees=degrees,
#             pos=np.zeros(3),
#             vel=np.zeros(3),
#         )

#         euler_out.append(mekf.euler(degrees=False))
#         bias_gyro_out.append(mekf.bias_gyro(degrees=False))

#     euler_out = np.array(euler_out)
#     bias_gyro_out = np.array(bias_gyro_out)

#     # Half-sample shift (compensates for the delay introduced by Euler integration)
#     euler_out = resample_poly(euler_out, 2, 1)[1:-1:2]
#     euler_ref = euler_ref[:-1, :]

#     roll_rms, pitch_rms, yaw_rms = np.std((euler_out - euler_ref)[warmup:], axis=0)
#     bias_gyro_x_rms, bias_gyro_y_rms, bias_gyro_z_rms = np.std(
#         (bias_gyro_out - bg)[warmup:], axis=0
#     )

#     assert np.degrees(roll_rms) <= 0.1
#     assert np.degrees(pitch_rms) <= 0.1
#     assert np.degrees(bias_gyro_x_rms) <= 0.005
#     assert np.degrees(bias_gyro_y_rms) <= 0.005


# @pytest.mark.parametrize(
#     "benchmark_gen, degrees",
#     [
#         (benchmark_pure_attitude_beat_202311A, False),
#         (benchmark_pure_attitude_chirp_202311A, True),
#     ],
# )
# def test_ains_attitude_benchmark(benchmark_gen, degrees):
#     fs_imu = 100.0
#     warmup = int(fs_imu * 600.0)  # truncate 600 seconds from the beginning

#     # Reference signals (without noise)
#     t, euler_ref, acc_ref, gyro_ref = benchmark_gen(fs_imu)

#     # IMU measurements (with noise)
#     bg = np.array([0.01, -0.02, 0.0])
#     noise_model = sf.noise.IMUNoise(
#         err_acc=sf.constants.ERR_ACC_MOTION2,
#         err_gyro=sf.constants.ERR_GYRO_MOTION2,
#         seed=0,
#     )
#     imu_noise = noise_model(fs_imu, len(t))
#     acc_noise = acc_ref + imu_noise[:, :3]
#     gyro_noise = gyro_ref + imu_noise[:, 3:] + bg

#     head_std = np.radians(1.0)
#     head_noise = euler_ref[:, -1] + np.random.normal(0.0, head_std, len(euler_ref))

#     if degrees:
#         gyro_noise = np.degrees(gyro_noise)
#         head_noise = np.degrees(head_noise)
#         head_std = np.degrees(head_std)

#     # MEKF
#     q0 = sf.quaternion_from_euler(euler_ref[0], degrees=False)
#     mekf = AINS(
#         fs_imu,
#         q=q0,
#         gyro_noise_density=sf.constants.ERR_GYRO_MOTION2["N"],
#         gyro_bias_stability=sf.constants.ERR_GYRO_MOTION2["B"],
#         gyro_bias_corr_time=sf.constants.ERR_GYRO_MOTION2["tau_cb"],
#     )

#     vel_aid = (0.0, 0.0, 0.0)
#     vel_var = (100.0, 100.0, 100.0)  # (10.0 m/s)^2
#     # Apply filter
#     euler_out, bias_gyro_out = [], []
#     for i, (f_i, w_i, head_i) in enumerate(zip(acc_noise, gyro_noise, head_noise)):

#         dvel = f_i / fs_imu
#         dtheta = w_i / fs_imu

#         mekf.update(
#             dvel,
#             dtheta,
#             degrees=degrees,
#             vel=vel_aid,
#             vel_var=vel_var,
#             head=head_i,
#             head_degrees=degrees,
#             head_var=head_std**2,
#         )

#         euler_out.append(mekf.euler(degrees=False))
#         bias_gyro_out.append(mekf.bias_gyro(degrees=False))

#     euler_out = np.array(euler_out)
#     bias_gyro_out = np.array(bias_gyro_out)

#     # Half-sample shift (compensates for the delay introduced by Euler integration)
#     euler_out = resample_poly(euler_out, 2, 1)[1:-1:2]
#     euler_ref = euler_ref[:-1, :]

#     roll_rms, pitch_rms, yaw_rms = np.std((euler_out - euler_ref)[warmup:], axis=0)
#     bias_gyro_x_rms, bias_gyro_y_rms, bias_gyro_z_rms = np.std(
#         (bias_gyro_out - bg)[warmup:], axis=0
#     )

#     assert np.degrees(roll_rms) <= 0.1
#     assert np.degrees(pitch_rms) <= 0.1
#     assert np.degrees(yaw_rms) <= 0.1
#     assert np.degrees(bias_gyro_x_rms) <= 0.005
#     assert np.degrees(bias_gyro_y_rms) <= 0.005
#     assert np.degrees(bias_gyro_z_rms) <= 0.005
