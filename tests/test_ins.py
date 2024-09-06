"""
IMPORTANT
---------

SciPy Rotation implementation is used as reference in tests. However, SciPy
operates with active rotations, whereas passive rotations are considered here. Keep in
mind that passive rotations is simply the inverse active rotations and vice versa.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from pandas import read_parquet
from scipy.signal import resample_poly
from scipy.spatial.transform import Rotation

from smsfusion._ins import (
    AidedINS,
    FixedNED,
    INSMixin,
    StrapdownINS,
    _dhda_head,
    _h_head,
    _signed_smallest_angle,
    gravity,
)
from smsfusion._transforms import (
    _angular_matrix_from_quaternion,
    _quaternion_from_euler,
    _rot_matrix_from_quaternion,
)
from smsfusion._vectorops import _normalize, _quaternion_product, _skew_symmetric
from smsfusion.benchmark import (
    benchmark_full_pva_beat_202311A,
    benchmark_full_pva_chirp_202311A,
)
from smsfusion.noise import IMUNoise, white_noise


@pytest.mark.parametrize(
    "angle, degrees, angle_expect",
    [
        (0.0, True, 0.0),
        (-180.0, True, -180.0),
        (180.0, True, -180.0),
        (-np.pi, False, -np.pi),
        (np.pi, False, -np.pi),
        (90.0, True, 90.0),
        (-90.0, True, -90.0),
        (181, True, -179.0),
        (-181, True, 179.0),
    ],
)
def test__signed_smallest_angle(angle, degrees, angle_expect):
    assert _signed_smallest_angle(angle, degrees=degrees) == pytest.approx(angle_expect)


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
class Test_INSMixin:
    @pytest.fixture
    def x(self):
        p = np.array([1.0, 2.0, 3.0])
        v = np.array([4.0, 5.0, 6.0])
        q = np.array([1.0, 0.0, 0.0, 0.0])
        ba = np.array([7.0, 8.0, 9.0])
        bg = np.array([10.0, 11.0, 12.0])
        x = np.r_[p, v, q, ba, bg]
        return x

    @pytest.fixture
    def ins(self, x):
        class INS(INSMixin):
            def __init__(self, x):
                self._x = x

        return INS(x)

    def test_x(self, x, ins):
        x_out = ins.x
        x_expect = x
        assert x_out.shape == (16,)
        assert x_out is not ins._x
        np.testing.assert_array_equal(x_out, x_expect)

    def test_position(self, x, ins):
        p_out = ins.position()
        p_expect = np.array([1.0, 2.0, 3.0])
        assert p_out.shape == (3,)
        assert p_out is not ins._pos
        np.testing.assert_array_equal(p_out, p_expect)

    def test_velocity(self, x, ins):
        v_out = ins.velocity()
        v_expect = np.array([4.0, 5.0, 6.0])
        assert v_out.shape == (3,)
        assert v_out is not ins._vel
        np.testing.assert_array_equal(v_out, v_expect)

    def test_quaternion(self, x, ins):
        q_out = ins.quaternion()
        q_expect = np.array([1.0, 0.0, 0.0, 0.0])
        assert q_out.shape == (4,)
        assert q_out is not ins._q_nm
        np.testing.assert_array_equal(q_out, q_expect)

    def test_euler(self, x, ins):
        theta_out = ins.euler()
        theta_expect = np.array([0.0, 0.0, 0.0])
        assert theta_out.shape == (3,)
        np.testing.assert_array_equal(theta_out, theta_expect)

    def test_bias_acc(self, x, ins):
        ba_out = ins.bias_acc()
        ba_expect = np.array([7.0, 8.0, 9.0])
        assert ba_out.shape == (3,)
        assert ba_out is not ins._bias_acc
        np.testing.assert_array_equal(ba_out, ba_expect)

    def test_bias_gyro(self, x, ins):
        bg_out = ins.bias_gyro()
        bg_expect = np.array([10.0, 11.0, 12.0])
        assert bg_out.shape == (3,)
        assert bg_out is not ins._bias_gyro
        np.testing.assert_array_equal(bg_out, bg_expect)


@pytest.mark.filterwarnings("ignore")
class Test_StrapdownINS:
    @pytest.fixture
    def x0(self):
        p0 = np.array([0.0, 0.0, 0.0])
        v0 = np.array([0.0, 0.0, 0.0])
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        ba0 = np.array([0.0, 0.0, 0.0])
        bg0 = np.array([0.0, 0.0, 0.0])
        x0 = np.r_[p0, v0, q0, ba0, bg0]
        return x0

    @pytest.fixture
    def ins(self, x0):
        fs = 10.0
        ins = StrapdownINS(fs, x0)
        return ins

    @pytest.fixture
    def x0_nonzero(self):
        p0 = np.array([1.0, 2.0, 3.0])
        v0 = np.array([4.0, 5.0, 6.0])
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        ba0 = np.array([7.0, 8.0, 9.0])
        bg0 = np.array([10.0, 11.0, 12.0])
        x0 = np.r_[p0, v0, q0, ba0, bg0]
        return x0

    def test__init__(self, x0_nonzero):
        ins = StrapdownINS(10.24, x0_nonzero)

        assert isinstance(ins, INSMixin)
        np.testing.assert_array_equal(ins._x0, x0_nonzero)
        np.testing.assert_array_equal(ins._x, x0_nonzero)

    def test_x(self, x0_nonzero):
        ins = StrapdownINS(10.24, x0_nonzero)

        x_out = ins.x
        x_expect = x0_nonzero

        assert x_out.shape == (16,)
        assert x_out is not ins._x
        np.testing.assert_array_equal(x_out, x_expect)

    def test_position(self, x0_nonzero):
        ins = StrapdownINS(10.24, x0_nonzero)

        p_out = ins.position()
        p_expect = np.array([1.0, 2.0, 3.0])

        assert p_out.shape == (3,)
        assert p_out is not ins._pos
        np.testing.assert_array_equal(p_out, p_expect)

    def test_velocity(self, x0_nonzero):
        ins = StrapdownINS(10.24, x0_nonzero)

        v_out = ins.velocity()
        v_expect = np.array([4.0, 5.0, 6.0])

        assert v_out.shape == (3,)
        assert v_out is not ins._vel
        np.testing.assert_array_equal(v_out, v_expect)

    def test_quaternion(self, x0_nonzero):
        ins = StrapdownINS(10.24, x0_nonzero)

        q_out = ins.quaternion()
        q_expect = np.array([1.0, 0.0, 0.0, 0.0])

        assert q_out.shape == (4,)
        assert q_out is not ins._q_nm
        np.testing.assert_array_equal(q_out, q_expect)

    def test_euler(self, x0_nonzero):
        ins = StrapdownINS(10.24, x0_nonzero)

        theta_out = ins.euler()
        theta_expect = np.array([0.0, 0.0, 0.0])

        assert theta_out.shape == (3,)
        np.testing.assert_array_equal(theta_out, theta_expect)

    def test_bias_acc(self, x0_nonzero):
        ins = StrapdownINS(10.24, x0_nonzero)

        ba_out = ins.bias_acc()
        ba_expect = np.array([7.0, 8.0, 9.0])

        assert ba_out.shape == (3,)
        assert ba_out is not ins._vel
        np.testing.assert_array_equal(ba_out, ba_expect)

    def test_bias_gyro(self, x0_nonzero):
        ins = StrapdownINS(10.24, x0_nonzero)

        bg_out = ins.bias_gyro()
        bg_expect = np.array([10.0, 11.0, 12.0])

        assert bg_out.shape == (3,)
        assert bg_out is not ins._vel
        np.testing.assert_array_equal(bg_out, bg_expect)

    def test_reset(self, ins):
        x = np.random.random(16)
        x[6:10] = x[6:10] / np.linalg.norm(x[6:10])  # unit quaternion
        ins.reset(x)

        np.testing.assert_array_almost_equal(ins.x, x)

    def test_reset_2d(self, ins):
        x = np.random.random(16).reshape(-1, 1)
        x[6:10] = x[6:10] / np.linalg.norm(x[6:10])  # unit quaternion
        ins.reset(x)

        np.testing.assert_array_almost_equal(ins.x, x.flatten())

    def test_update_return_self(self, ins):
        g = 9.80665
        f = np.array([0.0, 0.0, -g])
        w = np.array([0.0, 0.0, 0.0])

        update_return = ins.update(f, w)
        assert update_return is ins

    def test_update(self, ins):
        g = ins._g
        f = np.array([1.0, 2.0, 3.0]) - g
        w = np.array([0.04, 0.05, 0.06])

        x0_out = ins.x
        ins.update(f, w)
        x1_out = ins.x

        x0_expect = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        x1_expect = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.1,
                0.2,
                0.3,
                0.99999,
                0.002,
                0.0025,
                0.003,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        np.testing.assert_array_almost_equal(x0_out, x0_expect)
        np.testing.assert_array_almost_equal(x1_out, x1_expect)

    def test_update_deg(self, ins):
        g = ins._g
        f_imu = np.array([1.0, 2.0, 3.0]) - g
        w_imu = np.array([4.0, 5.0, 6.0])

        x0_out = ins.x
        ins.update(f_imu, w_imu, degrees=True)
        x1_out = ins.x

        x0_expect = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        x1_expect = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.1,
                0.2,
                0.3,
                0.999971,
                0.003491,
                0.004363,
                0.005236,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        np.testing.assert_array_almost_equal(x0_out, x0_expect)
        np.testing.assert_array_almost_equal(x1_out, x1_expect)

    def test_update_with_bias(self, x0):
        ba = np.array([0.1, 0.2, 0.3])
        bg = np.array([0.4, 0.5, 0.6])
        x0[10:13] = ba
        x0[13:16] = bg
        ins = StrapdownINS(10.0, x0)
        g = ins._g
        f = np.array([1.0, 2.0, 3.0]) + ba - g  # IMU measurements w/bias
        w = np.array([0.04, 0.05, 0.06]) + bg  # IMU measurements w/bias

        x0_out = ins.x
        ins.update(f, w, degrees=False)
        x1_out = ins.x

        x0_expect = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
            ]
        )
        x1_expect = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.1,
                0.2,
                0.3,
                0.99999,
                0.002,
                0.0025,
                0.003,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
            ]
        )

        np.testing.assert_array_almost_equal(x0_out, x0_expect)
        np.testing.assert_array_almost_equal(x1_out, x1_expect)

    def test_update_twise(self, ins):
        g = ins._g
        f_imu = np.array([1.0, 2.0, 3.0]) - g
        w_imu = np.array([0.004, 0.005, 0.006])

        x0_out = ins.x
        ins.update(f_imu, w_imu, degrees=False)
        x1_out = ins.x
        ins.update(f_imu, w_imu, degrees=False)
        x2_out = ins.x

        x0_expect = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        dt = 1.0 / ins._fs
        # Calculate x1
        R0_expect = np.eye(3)
        T0_expect = _angular_matrix_from_quaternion(x0_expect[6:10])
        a0_expect = R0_expect @ f_imu + g
        x1_expect = np.zeros(16)
        x1_expect[0:3] = x0_expect[0:3] + dt * x0_expect[3:6]
        x1_expect[3:6] = x0_expect[3:6] + dt * a0_expect
        x1_expect[6:10] = x0_expect[6:10] + dt * T0_expect @ w_imu
        x1_expect[6:10] = x1_expect[6:10] / np.linalg.norm(x1_expect[6:10])

        # Calculate x2 by forward Euler
        R1_expect = _rot_matrix_from_quaternion(x1_expect[6:10])
        T1_expect = _angular_matrix_from_quaternion(x1_expect[6:10])
        a1_expect = R1_expect @ f_imu + g
        x2_expect = np.zeros(16)
        x2_expect[0:3] = x1_expect[0:3] + dt * x1_expect[3:6]
        x2_expect[3:6] = x1_expect[3:6] + dt * a1_expect
        x2_expect[6:10] = x1_expect[6:10] + dt * T1_expect @ w_imu
        x2_expect[6:10] = x2_expect[6:10] / np.linalg.norm(x2_expect[6:10])

        np.testing.assert_array_almost_equal(x0_out, x0_expect.flatten())
        np.testing.assert_array_almost_equal(x1_out, x1_expect.flatten())
        np.testing.assert_array_almost_equal(x2_out, x2_expect.flatten())


@pytest.mark.parametrize(
    "angles",
    [
        np.radians([0.0, 0.0, 35.0]),
        np.radians([25.0, 180.0, -125.0]),
        np.radians([10.0, 95.0, 1.0]),
    ],
)
def test__h_head(angles):
    alpha, beta, gamma = np.radians((0.0, 0.0, 15.0))

    quaternion = Rotation.from_euler(
        "ZYX", (gamma, beta, alpha), degrees=False
    ).as_quat()
    quaternion = np.r_[quaternion[3], quaternion[:3]]

    gamma_expect = _h_head(quaternion)
    assert gamma_expect == pytest.approx(gamma)


@pytest.mark.parametrize(
    "quaternion, dhda_expect",
    [
        (
            np.array([0.89442719, 0.4472136, 0.0, 0.0]),  # gibbs -> [1.0, 0.0, 0.0]
            np.array([0.0, 10.0, 20.0]) / (4.0 + 1.0) ** 2,
        ),
        (
            np.array([0.89442719, 0.0, 0.4472136, 0.0]),  # gibbs -> [0.0, 1.0, 0.0]
            np.array([6.0, 0.0, 12.0]) / (4.0 - 1.0) ** 2,
        ),
        (
            np.array([0.89442719, 0.0, 0.0, 0.4472136]),  # gibbs -> [0.0, 0.0, 1.0]
            np.array([0.0, 0.0, 20.0]) / ((4.0 - 1.0) ** 2 * (1 + (4.0 / 3.0) ** 2)),
        ),
        (
            np.array(
                [0.92387953, 0.22094238, 0.22094238, 0.22094238]
            ),  # gibbs -> [0.47829262, 0.47829262, 0.47829262]
            np.array([0.06751864, 0.29609696, 0.87452584]),
        ),
    ],
)
def test__dhda(quaternion, dhda_expect):
    dhda_out = _dhda_head(quaternion)
    np.testing.assert_array_almost_equal(dhda_out, dhda_expect)


class Test_AidedINS:
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
    def ains(self):
        fs = 10.24

        p_init = np.array([0.1, 0.0, 0.0])
        v_init = np.array([0.0, -0.1, 0.0])

        q_init = self.quaternion()

        bias_acc_init = np.array([0.0, 0.0, 0.1])
        bias_gyro_init = np.array([-0.1, 0.0, 0.0])

        x0 = np.r_[p_init, v_init, q_init, bias_acc_init, bias_gyro_init]
        P0_prior = 1e-6 * np.eye(15)

        err_acc = {"N": 4.0e-4, "B": 2.0e-4, "tau_cb": 50}
        err_gyro = {
            "N": (np.pi) / 180.0 * 2.0e-3,
            "B": (np.pi) / 180.0 * 8.0e-4,
            "tau_cb": 50,
        }

        ains = AidedINS(
            fs,
            x0,
            P0_prior,
            err_acc,
            err_gyro,
            lever_arm=np.ones(3),
            ignore_bias_acc=False,
        )
        return ains

    @pytest.fixture
    def ains_nobias(self):
        fs = 10.24

        p_init = np.array([0.1, 0.0, 0.0])
        v_init = np.array([0.0, -0.1, 0.0])

        q_init = self.quaternion()

        bias_acc_init = np.array([0.0, 0.0, 0.1])
        bias_gyro_init = np.array([-0.1, 0.0, 0.0])

        x0 = np.r_[p_init, v_init, q_init, bias_acc_init, bias_gyro_init]
        P0_prior = 1e-6 * np.eye(12)

        err_acc = {"N": 4.0e-4, "B": 2.0e-4, "tau_cb": 50}
        err_gyro = {
            "N": (np.pi) / 180.0 * 2.0e-3,
            "B": (np.pi) / 180.0 * 8.0e-4,
            "tau_cb": 50,
        }

        ains = AidedINS(
            fs,
            x0,
            P0_prior,
            err_acc,
            err_gyro,
            lever_arm=np.ones(3),
            ignore_bias_acc=True,
        )
        return ains

    def test__init__(self):
        fs = 10.24

        p_init = np.array([0.0, 0.0, 0.0])
        v_init = np.array([0.0, 0.0, 0.0])
        q_init = np.array([1.0, 0.0, 0.0, 0.0])
        bias_acc_init = np.array([0.0, 0.0, 0.0])
        bias_gyro_init = np.array([0.0, 0.0, 0.0])

        x0 = np.r_[p_init, v_init, q_init, bias_acc_init, bias_gyro_init]
        P0_prior = 1e-6 * np.eye(15)

        err_acc = {"N": 4.0e-4, "B": 2.0e-4, "tau_cb": 50}
        err_gyro = {
            "N": (np.pi) / 180.0 * 2.0e-3,
            "B": (np.pi) / 180.0 * 8.0e-4,
            "tau_cb": 50,
        }

        ains = AidedINS(
            fs,
            x0,
            P0_prior,
            err_acc,
            err_gyro,
            lever_arm=(1, 2, 3),
            lat=60.0,
            ignore_bias_acc=False,
        )

        assert isinstance(ains, AidedINS)
        assert isinstance(ains, INSMixin)
        assert ains._fs == 10.24
        assert ains._dt == 1.0 / 10.24
        assert ains._err_acc == err_acc
        assert ains._err_gyro == err_gyro
        assert isinstance(ains._ins, StrapdownINS)
        assert ains._ignore_bias_acc is False

        np.testing.assert_array_almost_equal(ains._x, x0)
        np.testing.assert_array_almost_equal(ains._ins._x, x0)
        np.testing.assert_array_almost_equal(ains._P_prior, P0_prior)
        np.testing.assert_array_almost_equal(ains._lever_arm, (1, 2, 3))

        assert ains._P.shape == (15, 15)

        # Check that correct latitude (and thus gravity) is used
        g_expect = np.array([0.0, 0.0, gravity(60.0)])
        np.testing.assert_array_almost_equal(ains._ins._g, g_expect)

        assert ains._F.shape == (15, 15)
        assert ains._G.shape == (15, 12)
        assert ains._W.shape == (12, 12)
        assert ains._H.shape == (10, 15)

    def test__init__ignore_bias_acc(self):
        fs = 10.24

        p_init = np.array([0.0, 0.0, 0.0])
        v_init = np.array([0.0, 0.0, 0.0])
        q_init = np.array([1.0, 0.0, 0.0, 0.0])
        bias_acc_init = np.array([0.0, 0.0, 0.0])
        bias_gyro_init = np.array([0.0, 0.0, 0.0])

        x0 = np.r_[p_init, v_init, q_init, bias_acc_init, bias_gyro_init]
        P0_prior = 1e-6 * np.eye(12)

        err_acc = {"N": 4.0e-4, "B": 2.0e-4, "tau_cb": 50}
        err_gyro = {
            "N": (np.pi) / 180.0 * 2.0e-3,
            "B": (np.pi) / 180.0 * 8.0e-4,
            "tau_cb": 50,
        }

        ains = AidedINS(
            fs,
            x0,
            P0_prior,
            err_acc,
            err_gyro,
            lever_arm=(1, 2, 3),
            lat=60.0,
            ignore_bias_acc=True,
        )

        assert ains._ignore_bias_acc is True
        np.testing.assert_array_almost_equal(ains._P_prior, P0_prior)
        assert ains._P_prior.shape == (12, 12)
        assert ains._P.shape == (12, 12)
        assert ains._F.shape == (12, 12)
        assert ains._G.shape == (12, 9)
        assert ains._W.shape == (9, 9)
        assert ains._H.shape == (10, 12)

    def test__init__defualt_lever_arm(self):
        x0 = np.random.random(16)
        x0[6:10] = (1.0, 0.0, 0.0, 0.0)
        P0_prior = np.eye(12)

        err_acc = {"N": 4.0e-4, "B": 2.0e-4, "tau_cb": 50}
        err_gyro = {
            "N": np.radians(2.0e-3),
            "B": np.radians(180.0 * 8.0e-4),
            "tau_cb": 50,
        }

        ains = AidedINS(
            10.24,
            x0,
            P0_prior,
            err_acc,
            err_gyro,
            # no lever_arm
        )

        np.testing.assert_array_almost_equal(ains._lever_arm, np.zeros(3))

    def test_dump(self, tmp_path, ains):
        kwargs_out = ains.dump()
        ains_b = AidedINS(**kwargs_out)

        with open(tmp_path / "ains.json", "w") as f:
            json.dump(kwargs_out, f)

        assert isinstance(kwargs_out, dict)
        assert ains_b._fs == ains._fs
        assert ains_b._err_acc == ains._err_acc
        assert ains_b._err_gyro == ains._err_gyro
        np.testing.assert_array_almost_equal(ains_b._lever_arm, ains._lever_arm)
        assert ains_b._lat == ains._lat
        np.testing.assert_array_almost_equal(ains_b._ins._x, ains._ins._x)
        np.testing.assert_array_almost_equal(ains_b._P_prior, ains._P_prior)
        np.testing.assert_array_almost_equal(
            ains_b._ignore_bias_acc, ains._ignore_bias_acc
        )

    def test_x(self, ains):
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

    def test_position(self, ains):
        pos_out = ains.position()
        pos_expect = np.array([0.1, 0.0, 0.0])

        np.testing.assert_array_almost_equal(pos_out, pos_expect)
        assert pos_out is not ains._pos

    def test_velocity(self, ains):
        vel_out = ains.velocity()
        vel_expect = np.array([0.0, -0.1, 0.0])

        np.testing.assert_array_almost_equal(vel_out, vel_expect)
        assert vel_out is not ains._vel

    def test_euler_radians(self, ains):
        theta_out = ains.euler(degrees=False)
        theta_expect = np.radians(np.array([-10.0, 5.0, 25.0]))

        np.testing.assert_array_almost_equal(theta_out, theta_expect)

    def test_euler_degrees(self, ains):
        theta_out = ains.euler(degrees=True)
        theta_expect = np.array([-10.0, 5.0, 25.0])

        np.testing.assert_array_almost_equal(theta_out, theta_expect)

    def test_quaternion(self, ains):
        quaternion_out = ains.quaternion()
        quaternion_expect = self.quaternion()

        np.testing.assert_array_almost_equal(quaternion_out, quaternion_expect)
        assert quaternion_out is not ains._q_nm

    def test__reset_ins(self, ains):
        x_ins = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.5,
            ]
        )
        dx = np.array(
            [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.05,
                0.06,
                0.07,
                0.7,
                0.8,
                0.9,
                0.10,
                0.11,
                0.12,
            ]
        )

        ains._ignore_bias_acc = False
        ains._ins._x = x_ins.copy()
        ains._reset_ins(dx)
        x_out = ains._ins.x

        da = dx[6:9]
        dq = (1.0 / np.sqrt(4.0 + da.T @ da)) * np.r_[2.0, da]

        x_expect = np.r_[
            x_ins[0:6] + dx[0:6],
            _normalize(_quaternion_product(x_ins[6:10], dq)),
            x_ins[10:13] + dx[9:12],
            x_ins[13:16] + dx[12:15],
        ]

        np.testing.assert_array_almost_equal(x_out, x_expect)

    def test__reset_ins_ignore_bias_acc(self, ains):
        x_ins = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.5,
            ]
        )
        dx = np.array(
            [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.05,
                0.06,
                0.07,
                0.7,
                0.8,
                0.9,
                0.10,
                0.11,
                0.12,
            ]
        )

        ains._ignore_bias_acc = True
        ains._ins._x = x_ins.copy()
        ains._reset_ins(dx)
        x_out = ains._ins.x

        da = dx[6:9]
        dq = (1.0 / np.sqrt(4.0 + da.T @ da)) * np.r_[2.0, da]

        x_expect = np.r_[
            x_ins[0:6] + dx[0:6],
            _normalize(_quaternion_product(x_ins[6:10], dq)),
            x_ins[10:13],
            x_ins[13:16] + dx[12:15],
        ]

        np.testing.assert_array_almost_equal(x_out, x_expect)

    def test_x_prior(self, ains):
        x_prior_out = ains.x_prior
        x_prior_expect = ains._ins.x
        np.testing.assert_array_almost_equal(x_prior_out, x_prior_expect)

    def test_P_prior(self, ains, ains_nobias):

        # With bias
        P_prior_out = ains.P_prior
        P_prior_expect = 1e-6 * np.eye(15)
        np.testing.assert_array_almost_equal(P_prior_out, P_prior_expect)
        assert P_prior_out is not ains._P_prior

        # Without bias
        P_prior_out = ains_nobias.P_prior
        P_prior_expect = 1e-6 * np.eye(12)
        np.testing.assert_array_almost_equal(P_prior_out, P_prior_expect)
        assert P_prior_out is not ains_nobias._P_prior

    def test_P(self, ains):
        P = np.random.random((15, 15))
        ains._P = P

        # Permutation matrix for reordering bias terms
        T = np.zeros((15, 15))
        T[:9, :9] = np.eye(9)
        T[9:12, 12:15] = np.eye(3)
        T[12:15, 9:12] = np.eye(3)

        P_out = ains.P
        P_expect = P

        np.testing.assert_array_almost_equal(P_out, P_expect)
        assert P_out is not ains._P

    def test__prep_F(self):
        err_acc = {"N": 4.0e-4, "B": 2.0e-4, "tau_cb": 50}
        err_gyro = {
            "N": (np.pi) / 180.0 * 2.0e-3,
            "B": (np.pi) / 180.0 * 8.0e-4,
            "tau_cb": 50,
        }

        quaternion = self.quaternion(alpha=0.0, beta=-12.0, gamma=45, degrees=True)

        F_matrix_out = AidedINS._prep_F(err_acc, err_gyro, quaternion)

        R = self.rot_matrix_from_quaternion  # body-to-ned rotation matrix
        S = _skew_symmetric  # skew symmetric matrix

        # Dummy values
        f_ins = np.array([0.0, 0.0, 0.0])
        w_ins = np.array([0.0, 0.0, 0.0])

        # "State" matrix
        F_matrix_expect = np.zeros((15, 15))
        F_matrix_expect[0:3, 3:6] = np.eye(3)
        F_matrix_expect[3:6, 6:9] = -R(quaternion) @ S(f_ins)
        F_matrix_expect[3:6, 9:12] = -R(quaternion)
        F_matrix_expect[6:9, 6:9] = -S(w_ins)  # NB! update each time step
        F_matrix_expect[6:9, 12:15] = -np.eye(3)
        F_matrix_expect[9:12, 9:12] = -(1.0 / err_acc["tau_cb"]) * np.eye(3)
        F_matrix_expect[12:15, 12:15] = -(1.0 / err_gyro["tau_cb"]) * np.eye(3)

        np.testing.assert_array_almost_equal(F_matrix_out, F_matrix_expect)

    def test__update_F(self, ains):
        quaternion_init = ains.quaternion()

        R = self.rot_matrix_from_quaternion  # body-to-ned rotation matrix
        S = _skew_symmetric  # skew symmetric matrix

        # Dummy values
        f_ins_init = np.array([0.0, 0.0, 0.0])
        w_ins_init = np.array([0.0, 0.0, 0.0])

        F_matrix_init = ains._F.copy()

        quaternion = self.quaternion(alpha=0.0, beta=-12.0, gamma=45, degrees=True)

        f_ins = np.array([0.0, 0.0, -gravity()])
        w_ins = np.array([0.01, -0.01, 0.01])

        ains._update_F(R(quaternion), f_ins, w_ins)

        delta_F_matrix_expect = np.zeros_like(F_matrix_init)
        delta_F_matrix_expect[3:6, 6:9] = -R(quaternion) @ S(f_ins) - (
            -R(quaternion_init) @ S(f_ins_init)
        )
        delta_F_matrix_expect[3:6, 9:12] = -R(quaternion) - (-R(quaternion_init))
        delta_F_matrix_expect[6:9, 6:9] = -S(w_ins) - (-S(w_ins_init))

        np.testing.assert_array_almost_equal(
            ains._F - F_matrix_init, delta_F_matrix_expect
        )

    def test__prep_G(self):
        quaternion = self.quaternion(alpha=0.0, beta=-12.0, gamma=45, degrees=True)

        G_matrix_out = AidedINS._prep_G(quaternion)

        R = self.rot_matrix_from_quaternion

        G_matrix_expect = np.zeros((15, 12))
        G_matrix_expect[3:6, 0:3] = -R(quaternion)  # NB! update each time step
        G_matrix_expect[6:9, 3:6] = -np.eye(3)
        G_matrix_expect[9:12, 6:9] = np.eye(3)
        G_matrix_expect[12:15, 9:12] = np.eye(3)

        np.testing.assert_array_almost_equal(G_matrix_out, G_matrix_expect)

    def test__update_G(self, ains):
        quaternion_init = ains.quaternion()

        R = self.rot_matrix_from_quaternion  # body-to-ned rotation matrix

        G_matrix_init = ains._G.copy()

        quaternion = self.quaternion(alpha=0.0, beta=-12.0, gamma=45, degrees=True)

        ains._update_G(R(quaternion))

        delta_G_matrix_expect = np.zeros_like(G_matrix_init)
        delta_G_matrix_expect[3:6, 0:3] = -R(quaternion) - (-R(quaternion_init))
        np.testing.assert_array_almost_equal(
            ains._G - G_matrix_init, delta_G_matrix_expect
        )

    def test__prep_W(self):
        err_acc = {"N": 0.01, "B": 0.002, "tau_cb": 1000.0}
        err_gyro = {"N": 0.03, "B": 0.004, "tau_cb": 2000.0}

        W_out = AidedINS._prep_W(err_acc, err_gyro)

        # White noise power spectral density matrix
        W_expect = np.eye(12)
        W_expect[0:3, 0:3] *= err_acc["N"] ** 2
        W_expect[3:6, 3:6] *= err_gyro["N"] ** 2
        W_expect[6:9, 6:9] *= 2.0 * err_acc["B"] ** 2 * (1.0 / err_acc["tau_cb"])
        W_expect[9:12, 9:12] *= 2.0 * err_gyro["B"] ** 2 * (1.0 / err_gyro["tau_cb"])

        np.testing.assert_array_almost_equal(W_out, W_expect)

    def test__prep_H_lever_arm_zero(self, ains):
        R = self.rot_matrix_from_quaternion  # body-to-ned rotation matrix
        S = _skew_symmetric  # skew symmetric matrix

        q = self.quaternion(alpha=0.0, beta=-12.0, gamma=45, degrees=True)

        v01_ned = np.array([0.0, 0.0, 1.0])

        H_matrix_expected = np.zeros((10, 15))
        H_matrix_expected[0:3, 0:3] = np.eye(3)  # position
        H_matrix_expected[3:6, 3:6] = np.eye(3)  # velocity
        H_matrix_expected[6:9, 6:9] = S(R(q).T @ v01_ned)  # gravity reference vector
        H_matrix_expected[9:10, 6:9] = _dhda_head(q)  # compass

        H_matrix_out = AidedINS._prep_H(q, np.zeros(3))
        np.testing.assert_array_almost_equal(H_matrix_out, H_matrix_expected)

    def test__prep_H_lever_arm(self, ains):
        R = self.rot_matrix_from_quaternion  # body-to-ned rotation matrix
        S = _skew_symmetric  # skew symmetric matrix

        q = self.quaternion(alpha=0.0, beta=-12.0, gamma=45, degrees=True)

        v01_ned = np.array([0.0, 0.0, 1.0])
        lever_arm = np.array([1.0, 1.0, 1.0])

        H_matrix_expected = np.zeros((10, 15))
        H_matrix_expected[0:3, 0:3] = np.eye(3)  # position
        H_matrix_expected[0:3, 6:9] = -R(q) @ S(lever_arm)
        H_matrix_expected[3:6, 3:6] = np.eye(3)  # velocity
        H_matrix_expected[6:9, 6:9] = S(R(q).T @ v01_ned)  # gravity reference vector
        H_matrix_expected[9:10, 6:9] = _dhda_head(q)  # compass

        H_matrix_out = AidedINS._prep_H(q, lever_arm)
        np.testing.assert_array_almost_equal(H_matrix_out, H_matrix_expected)

    def test__update_H_lever_arm(self, ains):
        quaternion_init = ains.quaternion()

        v01_ned = np.array([0.0, 0.0, 1.0])
        lever_arm = np.array([1.0, 1.0, 1.0])

        R = self.rot_matrix_from_quaternion  # body-to-ned rotation matrix
        S = _skew_symmetric  # skew symmetric matrix

        H_matrix_init = ains._H.copy()

        quaternion = self.quaternion(alpha=0.0, beta=-12.0, gamma=45, degrees=True)

        ains._update_H(R(quaternion), quaternion, lever_arm)

        delta_H_matrix_expect = np.zeros_like(H_matrix_init)
        delta_H_matrix_expect[6:9, 6:9] = S(R(quaternion).T @ v01_ned) - S(
            R(quaternion_init).T @ v01_ned
        )
        delta_H_matrix_expect[9:10, 6:9] = _dhda_head(quaternion) - _dhda_head(
            quaternion_init
        )
        delta_H_matrix_expect[0:3, 6:9] = -R(quaternion) @ S(lever_arm) + R(
            quaternion_init
        ) @ S(lever_arm)
        np.testing.assert_array_almost_equal(
            ains._H - H_matrix_init, delta_H_matrix_expect
        )

    def test_update_return_self(self, ains):
        g = gravity()
        f_imu = np.array([0.0, 0.0, -g])
        w_imu = np.zeros(3)
        head = 0.0
        pos = np.zeros(3)
        vel = np.zeros(3)
        pos_var = np.ones(3)
        vel_var = np.ones(3)
        head_var = 1.0
        g_var = 0.1**2 * np.ones(3)

        update_return = ains.update(
            f_imu,
            w_imu,
            degrees=True,
            pos=pos,
            pos_var=pos_var,
            vel=vel,
            vel_var=vel_var,
            head=head,
            head_var=head_var,
            head_degrees=True,
            g_ref=True,
            g_var=g_var,
        )
        assert update_return is ains

    def test_update_var(self):
        """Update using aiding variances in update method."""
        fs = 10.24

        x0 = np.zeros(16)
        x0[6] = 1.0
        P0_prior = 1e-6 * np.eye(15)

        err_acc = {"N": 0.01, "B": 0.002, "tau_cb": 1000.0}
        err_gyro = {"N": 0.03, "B": 0.004, "tau_cb": 2000.0}

        # Aiding variances given in __init__
        ains_a = AidedINS(
            fs,
            x0,
            P0_prior,
            err_acc,
            err_gyro,
            ignore_bias_acc=False,
        )

        g = gravity()
        f_imu = np.array([0.0, 0.0, -g])
        w_imu = np.zeros(3)

        pos = np.zeros(3)
        pos_var = np.ones(3)
        vel = np.zeros(3)
        vel_var = np.ones(3)
        head = 0.0
        head_var = 1.0
        g_var = np.ones(3)

        for _ in range(5):
            ains_a.update(
                f_imu,
                w_imu,
                degrees=True,
                pos=pos,
                pos_var=pos_var,
                vel=vel,
                vel_var=vel_var,
                head=head,
                head_var=head_var,
                head_degrees=True,
                g_ref=True,
                g_var=g_var,
            )

    def test_update_var_raises(self):
        """Check that update raise ValueError if no aiding variances are provided."""
        fs = 10.24

        x0 = np.zeros(16)
        x0[6] = 1.0
        P0_prior = 1e-6 * np.eye(15)

        err_acc = {"N": 0.01, "B": 0.002, "tau_cb": 1000.0}
        err_gyro = {"N": 0.03, "B": 0.004, "tau_cb": 2000.0}

        # Aiding variances given in __init__
        ains = AidedINS(
            fs,
            x0,
            P0_prior,
            err_acc,
            err_gyro,
            ignore_bias_acc=False,
        )

        g = gravity()
        f_imu = np.array([0.0, 0.0, -g])
        w_imu = np.zeros(3)
        head = 0.0
        pos = np.zeros(3)
        vel = np.zeros(3)

        with pytest.raises(ValueError):
            ains.update(
                f_imu,
                w_imu,
                degrees=True,
                pos=pos,
                pos_var=None,  # no aiding variance provided
            )

            ains.update(
                f_imu,
                w_imu,
                degrees=True,
                vel=vel,
                vel_var=None,  # no aiding variance provided
            )

            ains.update(
                f_imu,
                w_imu,
                degrees=True,
                head=head,
                head_var=None,  # no aiding variance provided
                head_degrees=True,
            )

            ains.update(
                f_imu,
                w_imu,
                degrees=True,
                g_ref=True,
                g_var=None,  # no aiding variance provided
            )

    def test_update_standstill(self):
        fs = 10.24

        x0 = np.zeros(16)
        x0[6] = 1.0
        P0_prior = 1e-6 * np.eye(15)

        err_acc = {"N": 0.01, "B": 0.002, "tau_cb": 1000.0}
        err_gyro = {"N": 0.03, "B": 0.004, "tau_cb": 2000.0}

        ains = AidedINS(fs, x0, P0_prior, err_acc, err_gyro, ignore_bias_acc=False)

        g = gravity()
        f_imu = np.array([0.0, 0.0, -g])
        w_imu = np.zeros(3)
        head = 0.0
        pos = np.zeros(3)
        vel = np.zeros(3)

        for _ in range(5):
            ains.update(
                f_imu,
                w_imu,
                degrees=True,
                pos=pos,
                pos_var=np.ones(3),
                vel=vel,
                vel_var=np.ones(3),
                head=head,
                head_var=0.1,
                head_degrees=True,
                g_ref=True,
                g_var=np.ones(3),
            )
            np.testing.assert_array_almost_equal(ains.x, x0)

    def test_update_irregular_aiding(self):
        fs = 10.24

        x0 = np.zeros(16)
        x0[6] = 1.0
        P0_prior = 1e-6 * np.eye(15)

        err_acc = {"N": 0.01, "B": 0.002, "tau_cb": 1000.0}
        err_gyro = {"N": 0.03, "B": 0.004, "tau_cb": 2000.0}

        ains = AidedINS(fs, x0, P0_prior, err_acc, err_gyro, ignore_bias_acc=False)

        g = gravity()
        f_imu = np.array([0.0, 0.0, -g])
        w_imu = np.zeros(3)
        pos = np.zeros(3)
        pos_var = np.ones(3)
        vel = np.zeros(3)
        vel_var = np.ones(3)
        head = 0.0
        head_var = 1.0
        g_var = np.ones(3)

        ains.update(
            f_imu,
            w_imu,
            degrees=True,
            pos=pos,
            pos_var=pos_var,
            vel=vel,
            vel_var=vel_var,
            head=head,
            head_var=head_var,
            head_degrees=True,
            g_ref=True,
            g_var=g_var,
        )
        np.testing.assert_array_almost_equal(ains.x, x0)
        ains.update(
            f_imu,
            w_imu,
            degrees=True,
        )
        np.testing.assert_array_almost_equal(ains.x, x0)

        ains.update(
            f_imu,
            w_imu,
            degrees=True,
            pos=pos,
            pos_var=pos_var,
            vel=vel,
            vel_var=vel_var,
            head=head,
            head_var=head_var,
            head_degrees=True,
            g_ref=True,
            g_var=g_var,
        )
        np.testing.assert_array_almost_equal(ains.x, x0)

        ains.update(
            f_imu,
            w_imu,
            degrees=True,
            vel=vel,
            vel_var=vel_var,
            head=head,
            head_var=head_var,
            head_degrees=True,
        )
        np.testing.assert_array_almost_equal(ains.x, x0)

        ains.update(
            f_imu,
            w_imu,
            degrees=True,
            pos=pos,
            pos_var=pos_var,
            head=head,
            head_var=head_var,
            head_degrees=True,
        )
        np.testing.assert_array_almost_equal(ains.x, x0)

        ains.update(
            f_imu,
            w_imu,
            degrees=True,
            head=head,
            head_var=head_var,
            head_degrees=True,
        )
        np.testing.assert_array_almost_equal(ains.x, x0)

        ains.update(f_imu, w_imu, degrees=True, g_ref=True, g_var=g_var)
        np.testing.assert_array_almost_equal(ains.x, x0)

    def test_update_ignore_bias_acc(self):
        fs = 10.24

        x0 = np.zeros(16)
        x0[6] = 1.0

        err_acc = {"N": 0.01, "B": 0.002, "tau_cb": 10.0}
        err_gyro = {"N": 0.03, "B": 0.004, "tau_cb": 10.0}

        ains_a = AidedINS(  # include accelerometer bias
            fs,
            x0,
            np.eye(15),
            err_acc,
            err_gyro,
            ignore_bias_acc=False,
        )

        ains_b = AidedINS(  # ignore accelerometer bias
            fs,
            x0,
            np.eye(12),
            err_acc,
            err_gyro,
            ignore_bias_acc=True,
        )

        g = gravity()
        f_imu = np.random.random(3) - np.array([0.0, 0.0, g])
        w_imu = np.random.random(3)

        pos = np.zeros(3)
        pos_var = np.ones(3)
        vel = np.zeros(3)
        vel_var = np.ones(3)
        head = np.random.random()
        head_var = 1.0

        ains_a.update(
            f_imu,
            w_imu,
            degrees=True,
            pos=pos,
            pos_var=pos_var,
            vel=vel,
            vel_var=vel_var,
            head=head,
            head_var=head_var,
            head_degrees=True,
        )

        ains_a.update(
            f_imu,
            w_imu,
            degrees=True,
            pos=pos,
            pos_var=pos_var,
            vel=vel,
            vel_var=vel_var,
            head=head,
            head_var=head_var,
            head_degrees=True,
        )

        ains_b.update(
            f_imu,
            w_imu,
            degrees=True,
            pos=pos,
            pos_var=pos_var,
            vel=vel,
            vel_var=vel_var,
            head=head,
            head_var=head_var,
            head_degrees=True,
        )

        ains_b.update(
            f_imu,
            w_imu,
            degrees=True,
            pos=pos,
            pos_var=pos_var,
            vel=vel,
            vel_var=vel_var,
            head=head,
            head_var=head_var,
            head_degrees=True,
        )

        assert not np.array_equal(ains_a.bias_acc(), x0[9:12])  # bias is updated
        np.testing.assert_array_almost_equal(ains_b.bias_acc(), x0[9:12])  # no update

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
        P0_prior = np.eye(12)
        x0 = np.zeros(16)
        x0[0:3] = pos_ref[0]
        x0[3:6] = vel_ref[0]
        x0[6:10] = _quaternion_from_euler(np.radians(euler_ref[0].flatten()))
        mekf = AidedINS(fs_imu, x0, P0_prior, err_acc, err_gyro, ignore_bias_acc=True)

        # Apply filter
        pos_out, vel_out, euler_out, bias_acc_out, bias_gyro_out = [], [], [], [], []
        for i, (acc_i, gyro_i, pos_i, vel_i, head_i) in enumerate(
            zip(acc_noise, gyro_noise, pos_meas, vel_meas, head_meas)
        ):
            if not (i % fs_ratio):  # with aiding
                mekf.update(
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
                mekf.update(acc_i, gyro_i, degrees=True)
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


class Test_FixedNed:
    def test_init(self):
        _ = FixedNED(0.0, 0.0, 0.0)

    @pytest.mark.parametrize(
        "lat, lon, height, x, y, z",
        [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0, 0.0, -1.0),
            (0.1, 0.0, 0.0, pytest.approx(11057.4, abs=0.05), 0.0, 0.0),
            (-0.1, 0.0, 0.0, pytest.approx(-11057.4, abs=0.05), 0.0, 0.0),
            (0.0, 0.1, 0.0, 0.0, pytest.approx(11131.9, abs=0.05), 0.0),
            (0.0, -0.1, 0.0, 0.0, pytest.approx(-11131.9, abs=0.05), 0.0),
            (
                0.1,
                0.1,
                0.0,
                pytest.approx(11057.4, abs=0.05),
                pytest.approx(11131.9, abs=0.05),
                0.0,
            ),
            (
                -0.1,
                -0.1,
                0.0,
                pytest.approx(-11057.4, abs=0.05),
                pytest.approx(-11131.9, abs=0.05),
                0.0,
            ),
        ],
    )
    def test_to_xyz(self, lat, lon, height, x, y, z):
        ned = FixedNED(0.0, 0.0, 0.0)

        x_, y_, z_ = ned.to_xyz(lat, lon, height)
        assert x_ == x
        assert y_ == y
        assert z_ == z

    @pytest.mark.parametrize(
        "lat, lon, height, x, y, z",
        [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0, 0.0, -1.0),
            (pytest.approx(0.1, abs=1e-4), 0.0, 0.0, 11057.4, 0.0, 0.0),
            (pytest.approx(-0.1, abs=1e-4), 0.0, 0.0, -11057.4, 0.0, 0.0),
            (0.0, pytest.approx(0.1, abs=1e-4), 0.0, 0.0, 11131.9, 0.0),
            (0.0, pytest.approx(-0.1, abs=1e-4), 0.0, 0.0, -11131.9, 0.0),
            (
                pytest.approx(0.1, abs=1e-4),
                pytest.approx(0.1, abs=1e-4),
                0.0,
                11057.4,
                11131.9,
                0.0,
            ),
            (
                pytest.approx(-0.1, abs=1e-4),
                pytest.approx(-0.1, abs=1e-4),
                0.0,
                -11057.4,
                -11131.9,
                0.0,
            ),
        ],
    )
    def test_to_llh(self, lat, lon, height, x, y, z):
        ned = FixedNED(0.0, 0.0, 0.0)

        lat_, lon_, height_ = ned.to_llh(x, y, z)
        assert lat_ == lat
        assert lon_ == lon
        assert height_ == height
