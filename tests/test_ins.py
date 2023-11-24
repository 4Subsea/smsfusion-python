"""
IMPORTANT
---------

SciPy Rotation implementation is used as reference in tests. However, SciPy
operates with active rotations, whereas passive rotations are considered here. Keep in
mind that passive rotations is simply the inverse active rotations and vice versa.
"""


from pathlib import Path

import numpy as np
import pytest
from pandas import read_parquet
from scipy.spatial.transform import Rotation

from smsfusion._ins import AHRS, AidedINS, StrapdownINS, gravity
from smsfusion._transforms import (
    _angular_matrix_from_euler,
    _quaternion_from_euler,
    _rot_matrix_from_euler,
)
from smsfusion.benchmark import benchmark_9dof_beat_202311A, benchmark_9dof_chirp_202311A
from smsfusion.noise import IMUNoise, white_noise


@pytest.mark.parametrize(
    "mu, g_expect",
    [
        (None, 9.80665),
        (0.0, 9.780325335903891718546),
        (90.0, 9.8321849378634),
        (59.91, 9.81910618638375),
    ],
)
def test_gravity(mu, g_expect):
    g_out = gravity(mu)
    assert g_out == pytest.approx(g_expect)


@pytest.fixture
def ains_ref_data():
    """Reference data for AINS testing."""
    return read_parquet(
        Path(__file__).parent / "testdata" / "ains_ahrs_imu.parquet", engine="pyarrow"
    )


@pytest.mark.filterwarnings("ignore")
class Test_StrapdownINS:
    @pytest.fixture
    def ins(self):
        x0 = np.zeros((9, 1))
        ins = StrapdownINS(x0)
        return ins

    def test__init__(self):
        x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.pi, np.pi / 2.0, np.pi / 4.0])
        ins = StrapdownINS(x0)

        np.testing.assert_array_equal(ins._x0, x0.reshape(-1, 1))
        np.testing.assert_array_equal(ins._x, x0.reshape(-1, 1))

    def test_reset(self, ins):
        x = np.random.random((9, 1))
        ins.reset(x)

        np.testing.assert_array_equal(ins._x, x)

    def test_x(self, ins):
        x = np.random.random(9)
        ins.reset(x)

        x_out = ins.x
        x_expect = x

        assert x_out.shape == (9,)
        assert x_out is not ins._x
        np.testing.assert_array_equal(x_out, x_expect)

    def test_position(self, ins):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.pi, np.pi / 2.0, np.pi / 4.0])
        ins.reset(x)

        p_out = ins.position()
        assert p_out is not ins._p
        p_expect = np.array([1.0, 2.0, 3.0])

        assert p_out.shape == (3,)
        np.testing.assert_array_almost_equal(p_out, p_expect)

    def test_velocity(self, ins):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.pi, np.pi / 2.0, np.pi / 4.0])
        ins.reset(x)

        v_out = ins.velocity()
        assert v_out is not ins._v
        v_expect = np.array([4.0, 5.0, 6.0])

        assert v_out.shape == (3,)
        np.testing.assert_array_almost_equal(v_out, v_expect)

    def test_euler_rad(self, ins):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.pi, np.pi / 2.0, np.pi / 4.0])
        ins.reset(x)

        theta_out = ins.euler(degrees=False)
        theta_expect = np.array([np.pi, np.pi / 2.0, np.pi / 4.0])

        assert theta_out.shape == (3,)
        assert theta_out is not ins._theta
        np.testing.assert_array_almost_equal(theta_out, theta_expect)

    def test_euler_deg(self, ins):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.pi, np.pi / 2.0, np.pi / 4.0])
        ins.reset(x)

        theta_out = ins.euler(degrees=True)
        theta_expect = np.array([180.0, 90.0, 45.0])

        assert theta_out.shape == (3,)
        assert theta_out is not ins._theta
        np.testing.assert_array_almost_equal(theta_out, theta_expect)

    def test_quaternion(self, ins):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.pi, np.pi / 2.0, np.pi / 4.0])
        ins.reset(x)

        quaternion_out = ins.quaternion()
        q_expected = Rotation.from_euler(
            "ZYX", np.array([np.pi, np.pi / 2.0, np.pi / 4.0])[::-1], degrees=False
        ).as_quat()
        q_expected = np.r_[q_expected[-1], q_expected[:-1]]

        assert quaternion_out.shape == (4,)
        np.testing.assert_array_almost_equal(quaternion_out, q_expected)

    def test_update_return_self(self):
        x0 = np.zeros((9, 1))
        strapdownins = StrapdownINS(x0)

        dt = 0.1
        g = 9.80665
        f = np.array([0.0, 0.0, -g]).reshape(-1, 1)
        w = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)

        update_return = strapdownins.update(dt, f, w)
        assert update_return is strapdownins

    def test_update(self):
        x0 = np.zeros((9, 1))
        ins = StrapdownINS(x0)

        h = 0.1
        g = ins._g
        f = np.array([1.0, 2.0, 3.0]).reshape(-1, 1) - g
        w = np.array([4.0, 5.0, 6.0]).reshape(-1, 1)

        x0_out = ins.x
        ins.update(h, f, w)
        x1_out = ins.x

        x0_expect = np.zeros(9)
        x1_expect = np.array([0.005, 0.01, 0.015, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        np.testing.assert_array_almost_equal(x0_out, x0_expect)
        np.testing.assert_array_almost_equal(x1_out, x1_expect)

    def test_update_deg(self):
        x0 = np.zeros((9, 1))
        ins = StrapdownINS(x0)

        dt = 0.1
        g = ins._g
        f_imu = np.array([1.0, 2.0, 3.0]).reshape(-1, 1) - g
        w_imu = np.array([4.0, 5.0, 6.0]).reshape(-1, 1)

        x0_out = ins.x
        ins.update(dt, f_imu, w_imu, degrees=True)
        x1_out = ins.x

        x0_expect = np.zeros(9)
        x1_expect = np.array(
            [
                0.005,
                0.01,
                0.015,
                0.1,
                0.2,
                0.3,
                (np.pi / 180.0) * 0.4,
                (np.pi / 180.0) * 0.5,
                (np.pi / 180.0) * 0.6,
            ]
        )

        np.testing.assert_array_almost_equal(x0_out, x0_expect)
        np.testing.assert_array_almost_equal(x1_out, x1_expect)

    def test_update_twise(self):
        x0 = np.zeros((9, 1))
        ins = StrapdownINS(x0)

        dt = 0.1
        g = ins._g
        f_imu = np.array([1.0, 2.0, 3.0]).reshape(-1, 1) - g
        w_imu = np.array([4.0, 5.0, 6.0]).reshape(-1, 1)

        x0_out = ins.x
        ins.update(dt, f_imu, w_imu)
        x1_out = ins.x
        ins.update(dt, f_imu, w_imu)
        x2_out = ins.x

        x0_expect = np.zeros((9, 1))

        # Calculate x1
        R0_expect = np.eye(3)
        T0_expect = np.eye(3)
        a0_expect = R0_expect @ f_imu + g
        x1_expect = np.zeros((9, 1))
        x1_expect[0:3] = (
            x0_expect[0:3] + dt * x0_expect[3:6] + 0.5 * dt**2 * a0_expect
        )
        x1_expect[3:6] = x0_expect[3:6] + dt * a0_expect
        x1_expect[6:9] = x0_expect[6:9] + dt * T0_expect @ w_imu

        # Calculate x2 by forward Euler
        R1_expect = Rotation.from_euler("ZYX", x1_expect[8:5:-1].flatten()).as_matrix()
        T1_expect = _angular_matrix_from_euler(x1_expect[6:9].flatten())
        a1_expect = R1_expect @ f_imu + g
        x2_expect = np.zeros((9, 1))
        x2_expect[0:3] = (
            x1_expect[0:3] + dt * x1_expect[3:6] + 0.5 * dt**2 * a1_expect
        )
        x2_expect[3:6] = x1_expect[3:6] + dt * a1_expect
        x2_expect[6:9] = x1_expect[6:9] + dt * T1_expect @ w_imu

        np.testing.assert_array_almost_equal(x0_out, x0_expect.flatten())
        np.testing.assert_array_almost_equal(x1_out, x1_expect.flatten())
        np.testing.assert_array_almost_equal(x2_out, x2_expect.flatten())

    def test_update_R_T(self):
        ins = StrapdownINS(np.zeros((9, 1)))

        h = 0.1
        g = ins._g
        f_imu = np.array([1.0, 2.0, 3.0]).reshape(-1, 1) - g
        w_imu = np.array([4.0, 5.0, 6.0])
        ins.update(h, f_imu, w_imu, theta_ext=(0.0, 0.0, 0.0))

        x_out = ins.x
        x_expect = np.array([0.005, 0.01, 0.015, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        np.testing.assert_array_almost_equal(x_out, x_expect)

        ins.update(h, f_imu, w_imu, theta_ext=(0.0, 0.0, 0.0))

        x_out = ins.x
        x_expect = np.array([0.02, 0.04, 0.06, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        np.testing.assert_array_almost_equal(x_out, x_expect)


class Test_AidedINS:
    @pytest.fixture
    def ains(self):
        fs = 10.24

        p0 = np.array([1.0, 2.0, 3.0])
        v0 = np.array([0.1, 0.2, 0.3])
        theta0 = np.array([np.pi / 4, np.pi / 8, np.pi / 16])
        b_acc0 = np.array([0.001, 0.002, 0.003])
        b_gyro0 = np.array([0.004, 0.005, 0.006])
        x0 = np.r_[p0, v0, theta0, b_acc0, b_gyro0]
        err_acc = {"N": 0.01, "B": 0.002, "tau_cb": 1000.0}
        err_gyro = {"N": 0.03, "B": 0.004, "tau_cb": 2000.0}
        var_pos = [1.0, 2.0, 3.0]
        var_ahrs = [4.0, 5.0, 6.0]
        ahrs = AHRS(fs, 0.050, 0.035)
        ains = AidedINS(fs, x0, err_acc, err_gyro, var_pos, var_ahrs, ahrs)
        return ains

    def test__init__(self):
        fs = 10.24

        x0 = np.zeros(15)
        err_acc = {"N": 0.01, "B": 0.002, "tau_cb": 1000.0}
        err_gyro = {"N": 0.03, "B": 0.004, "tau_cb": 2000.0}
        var_pos = [1.0, 2.0, 3.0]
        var_ahrs = [4.0, 5.0, 6.0]
        ahrs = AHRS(fs, 0.050, 0.035)
        ains = AidedINS(fs, x0, err_acc, err_gyro, var_pos, var_ahrs, ahrs)

        assert isinstance(ains, AidedINS)
        assert ains._fs == 10.24
        assert ains._dt == 1.0 / 10.24
        assert ains._err_acc == err_acc
        assert ains._err_gyro == err_gyro
        assert ains.ahrs._Kp == 0.050
        assert ains.ahrs._Ki == 0.035
        assert isinstance(ains.ahrs, AHRS)
        assert isinstance(ains._ins, StrapdownINS)
        np.testing.assert_array_almost_equal(ains._x_ins, np.zeros((15, 1)))
        np.testing.assert_array_almost_equal(ains._P_prior, np.eye(15))

        # State matrix
        F_expect = np.zeros((15, 15))
        F_expect[0:3, 3:6] = np.eye(3)
        F_expect[3:6, 9:12] = -np.eye(3)
        F_expect[6:9, 12:15] = -np.eye(3)
        F_expect[9:12, 9:12] = -(1.0 / err_acc["tau_cb"]) * np.eye(3)
        F_expect[12:15, 12:15] = -(1.0 / err_gyro["tau_cb"]) * np.eye(3)
        np.testing.assert_array_almost_equal(ains._F, F_expect)

        # Input (white noise) matrix
        G_expect = np.zeros((15, 12))
        G_expect[3:6, 0:3] = -np.eye(3)
        G_expect[6:9, 3:6] = -np.eye(3)
        G_expect[9:12, 6:9] = np.eye(3)
        G_expect[12:15, 9:12] = np.eye(3)
        np.testing.assert_array_almost_equal(ains._G, G_expect)

        # White noise power spectral density matrix
        W_expect = np.eye(12)
        W_expect[0:3, 0:3] *= err_acc["N"] ** 2
        W_expect[3:6, 3:6] *= err_gyro["N"] ** 2
        W_expect[6:9, 6:9] *= 2.0 * err_acc["B"] ** 2 * (1.0 / err_acc["tau_cb"])
        W_expect[9:12, 9:12] *= 2.0 * err_gyro["B"] ** 2 * (1.0 / err_gyro["tau_cb"])
        np.testing.assert_array_almost_equal(ains._W, W_expect)

        # Measurement noise covariance matrix
        R_expect = np.diag(np.r_[var_pos, var_ahrs])
        np.testing.assert_array_almost_equal(ains._R, R_expect)

    def test__init__wrong_ahrs_type(self):
        fs = 10.24

        x0 = np.zeros(15)
        err_acc = {"N": 0.01, "B": 0.002, "tau_cb": 1000.0}
        err_gyro = {"N": 0.03, "B": 0.004, "tau_cb": 2000.0}
        var_pos = [1.0, 2.0, 3.0]
        var_ahrs = [4.0, 5.0, 6.0]
        ahrs = None
        with pytest.raises(TypeError):
            _ = AidedINS(fs, x0, err_acc, err_gyro, var_pos, var_ahrs, ahrs)

    def test__init__ahrs_fs_mismatch(self):
        fs = 10.24

        x0 = np.zeros(15)
        err_acc = {"N": 0.01, "B": 0.002, "tau_cb": 1000.0}
        err_gyro = {"N": 0.03, "B": 0.004, "tau_cb": 2000.0}
        var_pos = [1.0, 2.0, 3.0]
        var_ahrs = [4.0, 5.0, 6.0]
        ahrs = AHRS(2.0 * fs, 0.050, 0.035)
        with pytest.raises(ValueError):
            _ = AidedINS(fs, x0, err_acc, err_gyro, var_pos, var_ahrs, ahrs)

    def test_x(self, ains):
        x_expect = np.array(
            [
                1.0,
                2.0,
                3.0,
                0.1,
                0.2,
                0.3,
                np.pi / 4,
                np.pi / 8,
                np.pi / 16,
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.006,
            ]
        )
        x_out = ains.x

        assert x_out.shape == (15,)
        assert x_out is not ains._x
        np.testing.assert_array_almost_equal(x_out, x_expect)
        assert x_out is not ains._x

    def test_position(self, ains):
        pos_out = ains.position()
        pos_expect = np.array([1.0, 2.0, 3.0])

        assert pos_out.shape == (3,)
        assert pos_out is not ains._p
        np.testing.assert_array_almost_equal(pos_out, pos_expect)

    def test_velocity(self, ains):
        vel_out = ains.velocity()
        vel_expect = np.array([0.1, 0.2, 0.3])

        assert vel_out.shape == (3,)
        assert vel_out is not ains._v
        np.testing.assert_array_almost_equal(vel_out, vel_expect)

    def test_euler_radians(self, ains):
        theta_out = ains.euler(degrees=False)
        theta_expect = np.array([np.pi / 4, np.pi / 8, np.pi / 16])

        assert theta_out.shape == (3,)
        assert theta_out is not ains._theta
        np.testing.assert_array_almost_equal(theta_out, theta_expect)

    def test_euler_degrees(self, ains):
        theta_out = ains.euler(degrees=True)
        theta_expect = (180.0 / np.pi) * np.array([np.pi / 4, np.pi / 8, np.pi / 16])

        assert theta_out.shape == (3,)
        assert theta_out is not ains._theta
        np.testing.assert_array_almost_equal(theta_out, theta_expect)

    def test_quaternion(self, ains):
        quaternion_out = ains.quaternion()

        theta_expect = np.array([np.pi / 4, np.pi / 8, np.pi / 16])
        q_expected = Rotation.from_euler(
            "ZYX", theta_expect[::-1], degrees=False
        ).as_quat()
        q_expected = np.r_[q_expected[-1], q_expected[0:-1]]  # scipy rearrange

        assert quaternion_out.shape == (4,)
        np.testing.assert_array_almost_equal(quaternion_out, q_expected)

    def test__prep_F_matrix(self):
        err_acc = {"N": 0.01, "B": 0.002, "tau_cb": 1000.0}
        err_gyro = {"N": 0.03, "B": 0.004, "tau_cb": 2000.0}
        theta = np.array([np.pi / 8, np.pi / 16, 0.0])

        F_out = AidedINS._prep_F_matrix(err_acc, err_gyro, theta)

        # State matrix
        R_bn = _rot_matrix_from_euler(theta)
        T = _angular_matrix_from_euler(theta)
        F_expect = np.zeros((15, 15))
        F_expect[0:3, 3:6] = np.eye(3)
        F_expect[3:6, 9:12] = -R_bn
        F_expect[6:9, 12:15] = -T
        F_expect[9:12, 9:12] = -(1.0 / err_acc["tau_cb"]) * np.eye(3)
        F_expect[12:15, 12:15] = -(1.0 / err_gyro["tau_cb"]) * np.eye(3)

        np.testing.assert_array_almost_equal(F_out, F_expect)

    def test__prep_G_matrix(self):
        theta = np.array([np.pi / 8, np.pi / 16, 0.0])

        G_out = AidedINS._prep_G_matrix(theta)

        # Input (white noise) matrix
        R_bn = _rot_matrix_from_euler(theta)
        T = _angular_matrix_from_euler(theta)
        G_expect = np.zeros((15, 12))
        G_expect[3:6, 0:3] = -R_bn
        G_expect[6:9, 3:6] = -T
        G_expect[9:12, 6:9] = np.eye(3)
        G_expect[12:15, 9:12] = np.eye(3)

        np.testing.assert_array_almost_equal(G_out, G_expect)

    def test__prep_W_matrix(self):
        err_acc = {"N": 0.01, "B": 0.002, "tau_cb": 1000.0}
        err_gyro = {"N": 0.03, "B": 0.004, "tau_cb": 2000.0}

        W_out = AidedINS._prep_W_matrix(err_acc, err_gyro)

        # White noise power spectral density matrix
        W_expect = np.eye(12)
        W_expect[0:3, 0:3] *= err_acc["N"] ** 2
        W_expect[3:6, 3:6] *= err_gyro["N"] ** 2
        W_expect[6:9, 6:9] *= 2.0 * err_acc["B"] ** 2 * (1.0 / err_acc["tau_cb"])
        W_expect[9:12, 9:12] *= 2.0 * err_gyro["B"] ** 2 * (1.0 / err_gyro["tau_cb"])

        np.testing.assert_array_almost_equal(W_out, W_expect)

    def test__prep_H_matrix(self):
        H_out = AidedINS._prep_H_matrix()

        H_expect = np.zeros((6, 15))
        H_expect[0:3, 0:3] = np.eye(3)
        H_expect[3:6, 6:9] = np.eye(3)

        np.testing.assert_array_almost_equal(H_out, H_expect)

    def test_update_return_self(self):
        fs = 10.24
        Kp = 0.5
        Ki = 0.1

        x0 = np.zeros(15)
        err_acc = {"N": 0.01, "B": 0.002, "tau_cb": 1000.0}
        err_gyro = {"N": 0.03, "B": 0.004, "tau_cb": 2000.0}
        var_pos = np.ones(3)
        var_ahrs = np.ones(3)

        ahrs = AHRS(fs, Kp, Ki)
        ains = AidedINS(fs, x0, err_acc, err_gyro, var_pos, var_ahrs, ahrs)

        g = gravity()
        f_imu = np.array([0.0, 0.0, -g])
        w_imu = np.zeros(3)
        head = 0.0
        pos = np.zeros(3)

        update_return = ains.update(
            f_imu, w_imu, head, pos, degrees=True, head_degrees=True
        )
        assert update_return is ains

    def test_update_standstill(self):
        fs = 10.24

        x0 = np.zeros(15)
        err_acc = {"N": 0.01, "B": 0.002, "tau_cb": 1000.0}
        err_gyro = {"N": 0.03, "B": 0.004, "tau_cb": 2000.0}
        var_pos = np.ones(3)
        var_ahrs = np.ones(3)

        ahrs = AHRS(fs, 0.050, 0.035)
        ains = AidedINS(fs, x0, err_acc, err_gyro, var_pos, var_ahrs, ahrs)

        g = gravity()
        f_imu = np.array([0.0, 0.0, -g])
        w_imu = np.zeros(3)
        head = 0.0
        pos = np.zeros(3)

        ains.update(f_imu, w_imu, head, pos, degrees=True, head_degrees=True)
        np.testing.assert_array_almost_equal(ains.x, np.zeros(15))
        ains.update(f_imu, w_imu, head, pos, degrees=True, head_degrees=True)
        np.testing.assert_array_almost_equal(ains.x, np.zeros(15))

    def test_update_irregular_position_aiding(self):
        fs = 10.24

        x0 = np.zeros(15)
        err_acc = {"N": 0.01, "B": 0.002, "tau_cb": 1000.0}
        err_gyro = {"N": 0.03, "B": 0.004, "tau_cb": 2000.0}
        var_pos = np.ones(3)
        var_ahrs = np.ones(3)

        ahrs = AHRS(fs, 0.050, 0.035)
        ains = AidedINS(fs, x0, err_acc, err_gyro, var_pos, var_ahrs, ahrs)

        g = gravity()
        f_imu = np.array([0.0, 0.0, -g])
        w_imu = np.zeros(3)
        head = 0.0
        pos = np.zeros(3)

        ains.update(f_imu, w_imu, head, None, degrees=True, head_degrees=True)  # no pos
        np.testing.assert_array_almost_equal(ains.x, np.zeros(15))
        ains.update(f_imu, w_imu, head, pos, degrees=True, head_degrees=True)  # pos
        np.testing.assert_array_almost_equal(ains.x, np.zeros(15))
        ains.update(f_imu, w_imu, head, degrees=True, head_degrees=True)  # no pos
        np.testing.assert_array_almost_equal(ains.x, np.zeros(15))

    def test_update_reference_case(self, ains_ref_data):
        """Test that succesive calls goes through"""
        fs = 10.24

        # Measurement data
        f_imu = ains_ref_data[["Ax_meas", "Ay_meas", "Az_meas"]].values
        w_imu = ains_ref_data[["Gx_meas", "Gy_meas", "Gz_meas"]].values
        head = ains_ref_data[["Gamma_meas"]].values

        pos_gnss = ains_ref_data[["X_meas", "Y_meas", "Z_meas"]].values

        # Measurement noise uncertainty
        var_pos = 0.05**2 * np.ones(3)  # 5 cm std uncertainity
        var_ahrs = np.radians(2.0) ** 2 * np.ones(3)  # 2 deg std uncertainity

        # AHRS
        Kp = 0.27
        Ki = 0.025

        q_init = _quaternion_from_euler(
            np.radians(ains_ref_data[["Alpha", "Beta", "Gamma"]].values[0])
        )
        ahrs = AHRS(fs, Kp, Ki, q_init=q_init)

        # AINS
        ACC_NOISE = {"N": 5.0e-1, "B": 5.0e-1, "tau_cb": 50.0}

        GYRO_NOISE = {
            "N": (np.pi / 180.0) * 5e-3,
            "B": (np.pi / 180.0) * 5e-3,
            "tau_cb": 50.0,
        }

        x0 = np.r_[
            ains_ref_data[["X", "Y", "Z", "VX", "VY", "VZ"]].values[0],
            np.radians(ains_ref_data[["Alpha", "Beta", "Gamma"]].values[0]),
            np.zeros(6),
        ].reshape(15, 1)
        ains = AidedINS(fs, x0, ACC_NOISE, GYRO_NOISE, var_pos, var_ahrs, ahrs)

        pos_aiding_inc = 10  # Position aiding every N update step

        pos_out = []
        vel_out = []
        euler_out = []
        for k, (f_imu_k, w_imu_k, head_k, pos_k) in enumerate(
            zip(f_imu, w_imu, head, pos_gnss)
        ):
            if k % pos_aiding_inc:
                pos_k = None

            ains.update(
                f_imu_k, w_imu_k, head_k, pos_k, degrees=True, head_degrees=True
            )
            pos_out.append(ains.position().flatten())
            vel_out.append(ains.velocity().flatten())
            euler_out.append(ains.euler(degrees=True).flatten())

        pos_out = np.asarray(pos_out)[600:]
        vel_out = np.asarray(vel_out)[600:]
        euler_out = np.asarray(euler_out)[600:]

        pos_expected = ains_ref_data.loc[:, ["X", "Y", "Z"]].iloc[600:]
        vel_expected = ains_ref_data.loc[:, ["VX", "VY", "VZ"]].iloc[600:]
        euler_expected = ains_ref_data.loc[:, ["Alpha", "Beta", "Gamma"]].iloc[600:]

        pos_rms = (pos_out - pos_expected).std(axis=0)
        assert pos_rms.shape == (3,)
        assert all(pos_rms <= 0.10)

        vel_rms = (vel_out - vel_expected).std(axis=0)
        assert vel_rms.shape == (3,)
        assert all(vel_rms <= 0.15)

        euler_rms = (euler_out - euler_expected).std(axis=0)
        assert euler_rms.shape == (3,)
        assert all(euler_rms <= 0.04)

    @pytest.mark.parametrize("benchmark_gen", [benchmark_9dof_beat_202311A, benchmark_9dof_chirp_202311A])
    def test_benchmark(self, benchmark_gen):
        fs_imu = 100.0
        fs_gps = 1.0
        fs_ratio = np.ceil(fs_imu / fs_gps)
        warmup = int(fs_imu * 200.0)  # truncate 200 seconds from the beginning
        compass_noise_std = 0.5
        gps_noise_std = 0.1

        # Reference signals (without noise)
        t, pos_ref, vel_ref, euler_ref, acc_ref, gyro_ref = benchmark_gen(fs_imu)
        euler_ref = np.degrees(euler_ref)
        gyro_ref = np.degrees(gyro_ref)

        # Add measurement noise
        noise_model = IMUNoise(seed=96)
        imu_noise = noise_model(fs_imu, len(t))

        acc_noise = acc_ref + imu_noise[:, :3]
        gyro_noise = gyro_ref + imu_noise[:, 3:]

        compass = euler_ref[:, 2] + white_noise(
            compass_noise_std / np.sqrt(fs_imu), fs_imu, len(t)
        )

        gps = pos_ref + np.column_stack(
            [white_noise(gps_noise_std / np.sqrt(fs_gps), fs_gps, len(t)) for _ in range(3)]
        )

        omega_e = 2.0 * np.pi / 40.0
        delta = np.sqrt(3.0) / 2

        # AHRS
        Ki = omega_e**2
        Kp = delta * omega_e
        ahrs = AHRS(fs_imu, Kp, Ki)

        # AINS
        err_acc = {"N": 4.0e-4, "B": 1.5e-4, "K": 4.5e-6, "tau_cb": 50}
        err_gyro = {"N": 1.9e-3, "B": 7.5e-4, "tau_cb": 50}
        var_pos = gps_noise_std**2 * np.ones(3)
        var_ahrs = 0.01 * np.ones(3)
        x0 = np.zeros(15)
        x0[0:3] = pos_ref[0]
        x0[3:6] = vel_ref[0]
        x0[6:9] = np.degrees(euler_ref[0])
        ains = AidedINS(fs_imu, x0, err_acc, err_gyro, var_pos, var_ahrs, ahrs)

        # Apply filter
        pos_out = []
        vel_out = []
        euler_out = []
        for i, (acc_i, gyro_i, head_i, pos_i) in enumerate(zip(acc_noise, gyro_noise, compass, gps)):
            if not (i % fs_ratio):  # GPS aiding
                ains.update(acc_i, gyro_i, head_i, pos_i, degrees=True, head_degrees=True)
            else:  # no GPS aiding
                ains.update(acc_i, gyro_i, head_i, degrees=True, head_degrees=True)
            pos_out.append(ains.position())
            vel_out.append(ains.velocity())
            euler_out.append(ains.euler(degrees=True))

        pos_out = np.array(pos_out)
        vel_out = np.array(vel_out)
        euler_out = np.array(euler_out)

        # half-sample shift
        # euler_out = resample_poly(euler_out, 2, 1)[1:-1:2]
        # euler_ref = euler_ref[1:, :]

        pos_x_rms, pos_y_rms, pos_z_rms = np.std((pos_out - pos_ref)[warmup:], axis=0)
        vel_x_rms, vel_y_rms, vel_z_rms = np.std((vel_out - vel_ref)[warmup:], axis=0)
        roll_rms, pitch_rms, yaw_rms = np.std((euler_out - euler_ref)[warmup:], axis=0)

        assert pos_x_rms <= 0.8
        assert pos_y_rms <= 0.8
        assert pos_z_rms <= 0.8

        assert vel_x_rms <= 0.5
        assert vel_y_rms <= 0.5
        assert vel_z_rms <= 0.5

        assert roll_rms <= 0.2
        assert pitch_rms <= 0.2
        assert yaw_rms <= 0.2
