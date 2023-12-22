import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from smsfusion import MEKF, StrapdownINS, gravity
from smsfusion._mekf import _dhda, _gibbs, _h
from smsfusion._transforms import _rot_matrix_from_quaternion
from smsfusion._vectorops import _skew_symmetric


@pytest.mark.parametrize(
    "angle, axis",
    [
        (np.radians([0.0, 0.0, 35.0]), np.array([0.0, 0.0, 1.0])),
        (np.radians([0.0, -35.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        (np.radians([-10.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
    ],
)
def test__gibbs(angle, axis):
    q = Rotation.from_euler("ZYX", angle[::-1], degrees=False).as_quat()
    q = np.r_[q[3], q[:3]]

    gibbs_expected = 2.0 * axis * np.tan(angle / 2)
    np.testing.assert_almost_equal(_gibbs(q), gibbs_expected)


@pytest.mark.parametrize(
    "angles",
    [
        np.radians([0.0, 0.0, 35.0]),
        np.radians([25.0, 180.0, -125.0]),
        np.radians([10.0, 95.0, 1.0]),
    ],
)
def test__h(angles):
    alpha, beta, gamma = np.radians((0.0, 0.0, 15.0))

    quaternion = Rotation.from_euler(
        "ZYX", (gamma, beta, alpha), degrees=False
    ).as_quat()
    gibbs_vector = 2.0 * quaternion[:3] / quaternion[3]

    gamma_expect = _h(gibbs_vector)
    assert gamma_expect == pytest.approx(gamma)


@pytest.mark.parametrize(
    "gibbs_vector, dhda_expect",
    [
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 10.0, 20.0]) / (4.0 + 1.0) ** 2),
        (np.array([0.0, 1.0, 0.0]), np.array([6.0, 0.0, 12.0]) / (4.0 - 1.0) ** 2),
        (
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 20.0]) / ((4.0 - 1.0) ** 2 * (1 + (4.0 / 3.0) ** 2)),
        ),
    ],
)
def test__dhda(gibbs_vector, dhda_expect):
    dhda_out = _dhda(gibbs_vector)
    np.testing.assert_array_almost_equal(dhda_out, dhda_expect)


class Test_MEKF:
    @staticmethod
    def quaternion(alpha=-10.0, beta=5.0, gamma=25.0, degrees=True):
        """
        Convert Euler to quaternions using SciPy.
        """
        q = Rotation.from_euler("ZYX", (gamma, beta, alpha), degrees=degrees).as_quat()
        q = np.r_[q[3], q[:3]]
        return q

    @staticmethod
    def rot_matrix_from_quaternion(q):
        """
        Convert quaternion to rotation matrix using SciPy.
        """
        q = np.r_[q[1:], q[0]]
        return Rotation.from_quat(q).as_matrix()

    @pytest.fixture
    def mekf(self):
        fs = 10.24

        p_init = np.array([0.1, 0.0, 0.0])
        v_init = np.array([0.0, -0.1, 0.0])

        q_init = self.quaternion()

        bias_acc_init = np.array([0.0, 0.0, 0.1])
        bias_gyro_init = np.array([-0.1, 0.0, 0.0])

        x0 = np.r_[p_init, v_init, q_init, bias_acc_init, bias_gyro_init]

        err_acc = {"N": 4.0e-4, "B": 2.0e-4, "tau_cb": 50}
        err_gyro = {
            "N": (np.pi) / 180.0 * 2.0e-3,
            "B": (np.pi) / 180.0 * 8.0e-4,
            "tau_cb": 50,
        }

        var_pos = [0.1, 0.1, 0.1]
        var_vel = [0.1, 0.1, 0.1]
        var_g = (0.1) ** 2 * np.ones(3)
        var_compass = ((np.pi / 180.0) * 0.5) ** 2

        mekf = MEKF(fs, x0, err_acc, err_gyro, var_pos, var_vel, var_g, var_compass)
        return mekf

    def test__init__(self):
        fs = 10.24

        p_init = np.array([0.0, 0.0, 0.0])
        v_init = np.array([0.0, 0.0, 0.0])
        q_init = np.array([1.0, 0.0, 0.0, 0.0])
        bias_acc_init = np.array([0.0, 0.0, 0.0])
        bias_gyro_init = np.array([0.0, 0.0, 0.0])

        x0 = np.r_[p_init, v_init, q_init, bias_acc_init, bias_gyro_init]

        err_acc = {"N": 4.0e-4, "B": 2.0e-4, "tau_cb": 50}
        err_gyro = {
            "N": (np.pi) / 180.0 * 2.0e-3,
            "B": (np.pi) / 180.0 * 8.0e-4,
            "tau_cb": 50,
        }

        var_pos = [0.1, 0.1, 0.1]
        var_vel = [0.1, 0.1, 0.1]
        var_g = (0.1) ** 2 * np.ones(3)
        var_compass = ((np.pi / 180.0) * 0.5) ** 2

        ains = MEKF(fs, x0, err_acc, err_gyro, var_pos, var_vel, var_g, var_compass)

        assert isinstance(ains, MEKF)
        assert ains._fs == 10.24
        assert ains._dt == 1.0 / 10.24
        assert ains._err_acc == err_acc
        assert ains._err_gyro == err_gyro
        assert isinstance(ains._ins, StrapdownINS)
        np.testing.assert_array_almost_equal(ains._var_pos, var_pos)
        np.testing.assert_array_almost_equal(ains._var_vel, var_vel)
        np.testing.assert_array_almost_equal(ains._var_g, var_g)
        np.testing.assert_array_almost_equal(ains._var_compass, var_compass)
        np.testing.assert_array_almost_equal(ains._x_ins, x0)
        np.testing.assert_array_almost_equal(ains._P_prior, 1e-9 * np.eye(15))

        R = _rot_matrix_from_quaternion  # body-to-ned rotation matrix
        S = _skew_symmetric  # skew symmetric matrix

        # Dummy values
        f_ins = np.array([0.0, 0.0, 0.0])
        w_ins = np.array([0.0, 0.0, 0.0])

        # "State" matrix
        dfdx_expect = np.zeros((15, 15))
        dfdx_expect[0:3, 3:6] = np.eye(3)
        dfdx_expect[3:6, 6:9] = -R(q_init) @ S(f_ins)
        dfdx_expect[3:6, 9:12] = -R(q_init)
        dfdx_expect[6:9, 6:9] = -S(w_ins)  # NB! update each time step
        dfdx_expect[6:9, 12:15] = -np.eye(3)
        dfdx_expect[9:12, 9:12] = -(1.0 / err_acc["tau_cb"]) * np.eye(3)
        dfdx_expect[12:15, 12:15] = -(1.0 / err_gyro["tau_cb"]) * np.eye(3)
        np.testing.assert_array_almost_equal(ains._dfdx, dfdx_expect)

        # "Input (white noise)"" matrix
        dfdw_expect = np.zeros((15, 12))
        dfdw_expect[3:6, 0:3] = -R(q_init)  # NB! update each time step
        dfdw_expect[6:9, 3:6] = -np.eye(3)
        dfdw_expect[9:12, 6:9] = np.eye(3)
        dfdw_expect[12:15, 9:12] = np.eye(3)
        np.testing.assert_array_almost_equal(ains._dfdw, dfdw_expect)

        # White noise power spectral density matrix
        W_expect = np.eye(12)
        W_expect[0:3, 0:3] *= err_acc["N"] ** 2
        W_expect[3:6, 3:6] *= err_gyro["N"] ** 2
        W_expect[6:9, 6:9] *= 2.0 * err_acc["B"] ** 2 * (1.0 / err_acc["tau_cb"])
        W_expect[9:12, 9:12] *= 2.0 * err_gyro["B"] ** 2 * (1.0 / err_gyro["tau_cb"])
        np.testing.assert_array_almost_equal(ains._W, W_expect)

    def test_x(self, mekf):
        ains = mekf
        x_expect = np.array(
            [
                0.1,
                0.0,
                0.0,
                0.0,
                -0.1,
                0.0,
                *self.quaternion(),
                0.0,
                0.0,
                0.1,
                -0.1,
                0.0,
                0.0,
            ]
        )
        x_out = ains.x

        np.testing.assert_array_almost_equal(x_out, x_expect)
        assert x_out is not ains._x

    def test_position(self, mekf):
        ains = mekf
        pos_out = ains.position()
        pos_expect = np.array([0.1, 0.0, 0.0])

        np.testing.assert_array_almost_equal(pos_out, pos_expect)
        assert pos_out is not ains._p

    def test_velocity(self, mekf):
        ains = mekf
        vel_out = ains.velocity()
        vel_expect = np.array([0.0, -0.1, 0.0])

        np.testing.assert_array_almost_equal(vel_out, vel_expect)
        assert vel_out is not ains._v

    def test_euler_radians(self, mekf):
        ains = mekf
        theta_out = ains.euler(degrees=False)
        theta_expect = np.radians(np.array([-10.0, 5.0, 25.0]))

        np.testing.assert_array_almost_equal(theta_out, theta_expect)

    def test_euler_degrees(self, mekf):
        ains = mekf
        theta_out = ains.euler(degrees=True)
        theta_expect = np.array([-10.0, 5.0, 25.0])

        np.testing.assert_array_almost_equal(theta_out, theta_expect)

    def test_quaternion(self, mekf):
        ains = mekf
        quaternion_out = ains.quaternion()
        quaternion_expect = self.quaternion()

        np.testing.assert_array_almost_equal(quaternion_out, quaternion_expect)
        assert quaternion_out is not ains._q

    def test__prep_dfdx_matrix(self):
        err_acc = {"N": 4.0e-4, "B": 2.0e-4, "tau_cb": 50}
        err_gyro = {
            "N": (np.pi) / 180.0 * 2.0e-3,
            "B": (np.pi) / 180.0 * 8.0e-4,
            "tau_cb": 50,
        }

        quaternion = self.quaternion()

        dfdw_out = MEKF._prep_dfdx_matrix(err_acc, err_gyro, quaternion)

        R = self.rot_matrix_from_quaternion  # body-to-ned rotation matrix
        S = _skew_symmetric  # skew symmetric matrix

        # Dummy values
        f_ins = np.array([0.0, 0.0, 0.0])
        w_ins = np.array([0.0, 0.0, 0.0])

        # "State" matrix
        dfdx_expect = np.zeros((15, 15))
        dfdx_expect[0:3, 3:6] = np.eye(3)
        dfdx_expect[3:6, 6:9] = -R(quaternion) @ S(f_ins)
        dfdx_expect[3:6, 9:12] = -R(quaternion)
        dfdx_expect[6:9, 6:9] = -S(w_ins)  # NB! update each time step
        dfdx_expect[6:9, 12:15] = -np.eye(3)
        dfdx_expect[9:12, 9:12] = -(1.0 / err_acc["tau_cb"]) * np.eye(3)
        dfdx_expect[12:15, 12:15] = -(1.0 / err_gyro["tau_cb"]) * np.eye(3)

        np.testing.assert_array_almost_equal(dfdw_out, dfdx_expect)

    def test__update_dfdx_matrix(self, mekf):
        ains = mekf
        quaternion_init = ains.quaternion()

        R = self.rot_matrix_from_quaternion  # body-to-ned rotation matrix
        S = _skew_symmetric  # skew symmetric matrix

        # Dummy values
        f_ins_init = np.array([0.0, 0.0, 0.0])
        w_ins_init = np.array([0.0, 0.0, 0.0])

        dfdx_init = ains._dfdx.copy()

        quaternion = self.quaternion()

        f_ins = np.array([0.0, 0.0, -gravity()])
        w_ins = np.array([0.01, -0.01, 0.01])

        ains._update_dfdx_matrix(quaternion, f_ins, w_ins)

        delta_dfdx_expect = np.zeros_like(dfdx_init)
        delta_dfdx_expect[3:6, 6:9] = -R(quaternion) @ S(f_ins) - (
            -R(quaternion_init) @ S(f_ins_init)
        )
        delta_dfdx_expect[3:6, 9:12] = -R(quaternion) - (-R(quaternion_init))
        delta_dfdx_expect[6:9, 6:9] = -S(w_ins) - (-S(w_ins_init))

        np.testing.assert_array_almost_equal(ains._dfdx - dfdx_init, delta_dfdx_expect)

    def test__prep_dfdw_matrix(self):
        quaternion = self.quaternion()

        dfdw_out = MEKF._prep_dfdw_matrix(quaternion)

        R = self.rot_matrix_from_quaternion

        dfdw_expect = np.zeros((15, 12))
        dfdw_expect[3:6, 0:3] = -R(quaternion)  # NB! update each time step
        dfdw_expect[6:9, 3:6] = -np.eye(3)
        dfdw_expect[9:12, 6:9] = np.eye(3)
        dfdw_expect[12:15, 9:12] = np.eye(3)

        np.testing.assert_array_almost_equal(dfdw_out, dfdw_expect)

    def test__update_dfdw_matrix(self, mekf):
        ains = mekf
        quaternion_init = ains.quaternion()

        R = self.rot_matrix_from_quaternion  # body-to-ned rotation matrix

        dfdw_init = ains._dfdw.copy()

        quaternion = self.quaternion()

        ains._update_dfdw_matrix(quaternion)

        delta_dfdw_expect = np.zeros_like(dfdw_init)
        delta_dfdw_expect[3:6, 0:3] = -R(quaternion) - (-R(quaternion_init))
        np.testing.assert_array_almost_equal(ains._dfdw - dfdw_init, delta_dfdw_expect)

    def test__prep_W_matrix(self):
        err_acc = {"N": 0.01, "B": 0.002, "tau_cb": 1000.0}
        err_gyro = {"N": 0.03, "B": 0.004, "tau_cb": 2000.0}

        W_out = MEKF._prep_W_matrix(err_acc, err_gyro)

        # White noise power spectral density matrix
        W_expect = np.eye(12)
        W_expect[0:3, 0:3] *= err_acc["N"] ** 2
        W_expect[3:6, 3:6] *= err_gyro["N"] ** 2
        W_expect[6:9, 6:9] *= 2.0 * err_acc["B"] ** 2 * (1.0 / err_acc["tau_cb"])
        W_expect[9:12, 9:12] *= 2.0 * err_gyro["B"] ** 2 * (1.0 / err_gyro["tau_cb"])

        np.testing.assert_array_almost_equal(W_out, W_expect)

    def test__prep_dhdx_matrix(self):
        # TODO: add tests for dhdx + all supporting functions.
        raise Exception

    def test_update_return_self(self, mekf):
        ains = mekf

        g = gravity()
        f_imu = np.array([0.0, 0.0, -g])
        w_imu = np.zeros(3)
        head = 0.0
        pos = np.zeros(3)
        vel = np.zeros(3)

        update_return = ains.update(
            f_imu, w_imu, pos, vel, head, degrees=True, head_degrees=True
        )
        assert update_return is ains

    def test_update_standstill(self):
        fs = 10.24

        x0 = np.zeros(16)
        x0[6] = 1.0

        err_acc = {"N": 0.01, "B": 0.002, "tau_cb": 1000.0}
        err_gyro = {"N": 0.03, "B": 0.004, "tau_cb": 2000.0}
        var_pos = np.ones(3)
        var_vel = np.ones(3)
        var_g = np.ones(3)
        var_compass = 1.0

        ains = MEKF(fs, x0, err_acc, err_gyro, var_pos, var_vel, var_g, var_compass)

        g = gravity()
        f_imu = np.array([0.0, 0.0, -g])
        w_imu = np.zeros(3)
        head = 0.0
        pos = np.zeros(3)
        vel = np.zeros(3)

        for _ in range(5):
            ains.update(f_imu, w_imu, pos, vel, head, degrees=True, head_degrees=True)
            np.testing.assert_array_almost_equal(ains.x, x0)

    def test_update_irregular_aiding(self):
        fs = 10.24

        x0 = np.zeros(16)
        x0[6] = 1.0

        err_acc = {"N": 0.01, "B": 0.002, "tau_cb": 1000.0}
        err_gyro = {"N": 0.03, "B": 0.004, "tau_cb": 2000.0}
        var_pos = np.ones(3)
        var_vel = np.ones(3)
        var_g = np.ones(3)
        var_compass = 1.0

        ains = MEKF(fs, x0, err_acc, err_gyro, var_pos, var_vel, var_g, var_compass)

        g = gravity()
        f_imu = np.array([0.0, 0.0, -g])
        w_imu = np.zeros(3)
        head = 0.0
        pos = np.zeros(3)
        vel = np.zeros(3)

        ains.update(f_imu, w_imu, pos, vel, head, degrees=True, head_degrees=True)
        np.testing.assert_array_almost_equal(ains.x, x0)

        ains.update(
            f_imu, w_imu, pos=None, vel=None, head=None, degrees=True, head_degrees=True
        )
        np.testing.assert_array_almost_equal(ains.x, x0)

        ains.update(
            f_imu, w_imu, pos=None, vel=vel, head=head, degrees=True, head_degrees=True
        )
        np.testing.assert_array_almost_equal(ains.x, x0)

        ains.update(
            f_imu, w_imu, pos=pos, vel=None, head=head, degrees=True, head_degrees=True
        )
        np.testing.assert_array_almost_equal(ains.x, x0)

        ains.update(
            f_imu, w_imu, pos=None, vel=None, head=head, degrees=True, head_degrees=True
        )
        np.testing.assert_array_almost_equal(ains.x, x0)

    # def test_update_reference_case(self, ains_ref_data):
    #     """Test that succesive calls goes through"""
    #     fs = 10.24

    #     # Measurement data
    #     f_imu = ains_ref_data[["Ax_meas", "Ay_meas", "Az_meas"]].values
    #     w_imu = ains_ref_data[["Gx_meas", "Gy_meas", "Gz_meas"]].values
    #     head = ains_ref_data[["Gamma_meas"]].values

    #     pos_gnss = ains_ref_data[["X_meas", "Y_meas", "Z_meas"]].values

    #     # Measurement noise uncertainty
    #     var_pos = 0.05**2 * np.ones(3)  # 5 cm std uncertainity
    #     var_ahrs = np.radians(2.0) ** 2 * np.ones(3)  # 2 deg std uncertainity

    #     # AHRS
    #     Kp = 0.27
    #     Ki = 0.025

    #     q_init = _quaternion_from_euler(
    #         np.radians(ains_ref_data[["Alpha", "Beta", "Gamma"]].values[0])
    #     )
    #     ahrs = AHRS(fs, Kp, Ki, q_init=q_init)

    #     # AINS
    #     ACC_NOISE = {"N": 5.0e-1, "B": 5.0e-1, "tau_cb": 50.0}

    #     GYRO_NOISE = {
    #         "N": (np.pi / 180.0) * 5e-3,
    #         "B": (np.pi / 180.0) * 5e-3,
    #         "tau_cb": 50.0,
    #     }

    #     x0 = np.r_[
    #         ains_ref_data[["X", "Y", "Z", "VX", "VY", "VZ"]].values[0],
    #         np.radians(ains_ref_data[["Alpha", "Beta", "Gamma"]].values[0]),
    #         np.zeros(6),
    #     ].reshape(15, 1)
    #     ains = AidedINS(fs, x0, ACC_NOISE, GYRO_NOISE, var_pos, var_ahrs, ahrs)

    #     pos_aiding_inc = 10  # Position aiding every N update step

    #     pos_out = []
    #     vel_out = []
    #     euler_out = []
    #     for k, (f_imu_k, w_imu_k, head_k, pos_k) in enumerate(
    #         zip(f_imu, w_imu, head, pos_gnss)
    #     ):
    #         if k % pos_aiding_inc:
    #             pos_k = None

    #         ains.update(
    #             f_imu_k, w_imu_k, head_k, pos_k, degrees=True, head_degrees=True
    #         )
    #         pos_out.append(ains.position().flatten())
    #         vel_out.append(ains.velocity().flatten())
    #         euler_out.append(ains.euler(degrees=True).flatten())

    #     pos_out = np.asarray(pos_out)[600:]
    #     vel_out = np.asarray(vel_out)[600:]
    #     euler_out = np.asarray(euler_out)[600:]

    #     pos_expected = ains_ref_data.loc[:, ["X", "Y", "Z"]].iloc[600:]
    #     vel_expected = ains_ref_data.loc[:, ["VX", "VY", "VZ"]].iloc[600:]
    #     euler_expected = ains_ref_data.loc[:, ["Alpha", "Beta", "Gamma"]].iloc[600:]

    #     pos_rms = (pos_out - pos_expected).std(axis=0)
    #     assert pos_rms.shape == (3,)
    #     assert all(pos_rms <= 0.10)

    #     vel_rms = (vel_out - vel_expected).std(axis=0)
    #     assert vel_rms.shape == (3,)
    #     assert all(vel_rms <= 0.15)

    #     euler_rms = (euler_out - euler_expected).std(axis=0)
    #     assert euler_rms.shape == (3,)
    #     assert all(euler_rms <= 0.04)

    # @pytest.mark.parametrize(
    #     "benchmark_gen",
    #     [benchmark_full_pva_beat_202311A, benchmark_full_pva_chirp_202311A],
    # )
    # def test_benchmark(self, benchmark_gen):
    #     fs_imu = 100.0
    #     fs_gps = 1.0
    #     fs_ratio = np.ceil(fs_imu / fs_gps)
    #     warmup = int(fs_imu * 600.0)  # truncate 600 seconds from the beginning
    #     compass_noise_std = 0.5
    #     gps_noise_std = 0.1

    #     # Reference signals (without noise)
    #     t, pos_ref, vel_ref, euler_ref, acc_ref, gyro_ref = benchmark_gen(fs_imu)
    #     euler_ref = np.degrees(euler_ref)
    #     gyro_ref = np.degrees(gyro_ref)

    #     # Add measurement noise
    #     noise_model = IMUNoise(seed=96)
    #     imu_noise = noise_model(fs_imu, len(t))

    #     acc_noise = acc_ref + imu_noise[:, :3]
    #     gyro_noise = gyro_ref + imu_noise[:, 3:]

    #     compass = euler_ref[:, 2] + white_noise(
    #         compass_noise_std / np.sqrt(fs_imu), fs_imu, len(t)
    #     )

    #     gps_noise = np.column_stack(
    #         [
    #             white_noise(gps_noise_std / np.sqrt(fs_gps), fs_gps, len(t))
    #             for _ in range(3)
    #         ]
    #     )
    #     gps = pos_ref + gps_noise

    #     omega_e = 2.0 * np.pi / 40.0
    #     delta = np.sqrt(3.0) / 2

    #     # AHRS
    #     Ki = omega_e**2
    #     Kp = delta * omega_e
    #     ahrs = AHRS(fs_imu, Kp, Ki)

    #     # AINS
    #     err_acc = {"N": 4.0e-4, "B": 1.5e-4, "K": 4.5e-6, "tau_cb": 50}
    #     err_gyro = {
    #         "N": (np.pi / 180.0) * 1.9e-3,
    #         "B": (np.pi / 180.0) * 7.5e-4,
    #         "tau_cb": 50,
    #     }
    #     var_pos = gps_noise_std**2 * np.ones(3)
    #     var_ahrs = 0.01 * np.ones(3)
    #     x0 = np.zeros(15)
    #     x0[0:3] = pos_ref[0]
    #     x0[3:6] = vel_ref[0]
    #     x0[6:9] = np.degrees(euler_ref[0])
    #     ains = AidedINS(fs_imu, x0, err_acc, err_gyro, var_pos, var_ahrs, ahrs)

    #     # Apply filter
    #     pos_out = []
    #     vel_out = []
    #     euler_out = []
    #     for i, (acc_i, gyro_i, head_i, pos_i) in enumerate(
    #         zip(acc_noise, gyro_noise, compass, gps)
    #     ):
    #         if not (i % fs_ratio):  # GPS aiding
    #             ains.update(
    #                 acc_i, gyro_i, head_i, pos_i, degrees=True, head_degrees=True
    #             )
    #         else:  # no GPS aiding
    #             ains.update(acc_i, gyro_i, head_i, degrees=True, head_degrees=True)
    #         pos_out.append(ains.position())
    #         vel_out.append(ains.velocity())
    #         euler_out.append(ains.euler(degrees=True))

    #     pos_out = np.array(pos_out)
    #     vel_out = np.array(vel_out)
    #     euler_out = np.array(euler_out)

    #     pos_x_rms, pos_y_rms, pos_z_rms = np.std((pos_out - pos_ref)[warmup:], axis=0)
    #     vel_x_rms, vel_y_rms, vel_z_rms = np.std((vel_out - vel_ref)[warmup:], axis=0)
    #     roll_rms, pitch_rms, yaw_rms = np.std((euler_out - euler_ref)[warmup:], axis=0)

    #     assert pos_z_rms <= 0.2
    #     assert vel_z_rms <= 0.02
    #     assert roll_rms <= 0.3
    #     assert pitch_rms <= 0.3
    #     assert yaw_rms <= 0.3
