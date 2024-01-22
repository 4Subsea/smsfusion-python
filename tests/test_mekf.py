import numpy as np
import pytest
from scipy.signal import resample_poly
from scipy.spatial.transform import Rotation

from smsfusion import MEKF, StrapdownINS, gravity
from smsfusion._mekf import _dhda, _gibbs, _h
from smsfusion._transforms import _quaternion_from_euler
from smsfusion._vectorops import _skew_symmetric
from smsfusion.benchmark import (
    benchmark_full_pva_beat_202311A,
    benchmark_full_pva_chirp_202311A,
)
from smsfusion.noise import IMUNoise, white_noise


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
        np.testing.assert_array_almost_equal(ains._P_prior, np.eye(15))

        assert ains._dfdx.shape == (15, 15)
        assert ains._dfdw.shape == (15, 12)
        assert ains._W.shape == (12, 12)
        assert ains._dhdx.shape == (10, 15)

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

        quaternion = self.quaternion(alpha=0.0, beta=-12.0, gamma=45, degrees=True)

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

        quaternion = self.quaternion(alpha=0.0, beta=-12.0, gamma=45, degrees=True)

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
        quaternion = self.quaternion(alpha=0.0, beta=-12.0, gamma=45, degrees=True)

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

        quaternion = self.quaternion(alpha=0.0, beta=-12.0, gamma=45, degrees=True)

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
        R = self.rot_matrix_from_quaternion  # body-to-ned rotation matrix
        S = _skew_symmetric  # skew symmetric matrix

        q = self.quaternion(alpha=0.0, beta=-12.0, gamma=45, degrees=True)

        v01_ned = np.array([0.0, 0.0, 1.0])

        dhdx_expected = np.zeros((10, 15))
        dhdx_expected[0:3, 0:3] = np.eye(3)  # position
        dhdx_expected[3:6, 3:6] = np.eye(3)  # velocity
        dhdx_expected[6:9, 6:9] = S(R(q).T @ v01_ned)  # gravity reference vector
        dhdx_expected[9:10, 6:9] = _dhda(_gibbs(q))  # compass

        dhdx_out = MEKF._prep_dhdx_matrix(q)
        np.testing.assert_array_almost_equal(dhdx_out, dhdx_expected)

    def test__update_dhdx_matrix(self, mekf):
        ains = mekf
        quaternion_init = ains.quaternion()

        v01_ned = np.array([0.0, 0.0, 1.0])

        R = self.rot_matrix_from_quaternion  # body-to-ned rotation matrix
        S = _skew_symmetric  # skew symmetric matrix

        dhdx_init = ains._dhdx.copy()

        quaternion = self.quaternion(alpha=0.0, beta=-12.0, gamma=45, degrees=True)

        ains._update_dhdx_matrix(quaternion)

        delta_dhdx_expect = np.zeros_like(dhdx_init)
        delta_dhdx_expect[6:9, 6:9] = S(R(quaternion).T @ v01_ned) - S(
            R(quaternion_init).T @ v01_ned
        )
        delta_dhdx_expect[9:10, 6:9] = _dhda(_gibbs(quaternion)) - _dhda(
            _gibbs(quaternion_init)
        )
        np.testing.assert_array_almost_equal(ains._dhdx - dhdx_init, delta_dhdx_expect)

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
        noise_model = IMUNoise(err_acc=err_acc_true, err_gyro=err_gyro_true, seed=0)
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

        # Half-sample shift (compensates for the delay introduced by Euler integration)
        pos_out = resample_poly(pos_out, 2, 1)[1:-1:2]
        pos_ref = pos_ref[:-1, :]
        vel_out = resample_poly(vel_out, 2, 1)[1:-1:2]
        vel_ref = vel_ref[:-1, :]
        euler_out = resample_poly(euler_out, 2, 1)[1:-1:2]
        euler_ref = euler_ref[:-1, :]

        pos_x_rms, pos_y_rms, pos_z_rms = np.std((pos_out - pos_ref)[warmup:], axis=0)
        vel_x_rms, vel_y_rms, vel_z_rms = np.std((vel_out - vel_ref)[warmup:], axis=0)
        roll_rms, pitch_rms, yaw_rms = np.std((euler_out - euler_ref)[warmup:], axis=0)
        bias_acc_x_rms, bias_acc_y_rms, bias_acc_z_rms = np.std(
            (bias_acc_out - err_acc_true["bc"])[warmup:], axis=0
        )
        bias_gyro_x_rms, bias_gyro_y_rms, bias_gyro_z_rms = np.std(
            (bias_gyro_out - err_gyro_true["bc"])[warmup:], axis=0
        )

        assert pos_x_rms <= 0.1
        assert pos_y_rms <= 0.1
        assert pos_z_rms <= 0.1
        assert vel_x_rms <= 0.02
        assert vel_y_rms <= 0.02
        assert vel_z_rms <= 0.02
        assert roll_rms <= 0.02
        assert pitch_rms <= 0.02
        assert yaw_rms <= 0.1
        assert bias_acc_x_rms <= 1e-3
        assert bias_acc_y_rms <= 1e-3
        assert bias_acc_z_rms <= 1e-3
        assert bias_gyro_x_rms <= 1e-3
        assert bias_gyro_y_rms <= 1e-3
        assert bias_gyro_z_rms <= 1e-3
