"""
IMPORTANT
---------

SciPy Rotation implementation is used as reference in tests. However, SciPy
operates with active rotations, whereas passive rotations are considered here. Keep in
mind that passive rotations is simply the inverse active rotations and vice versa.
"""


import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from smsfusion._ins import StrapdownINS, gravity
from smsfusion._transforms import _angular_matrix_from_euler, _rot_matrix_from_euler


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
        x = np.random.random((9, 1))
        ins.reset(x)

        x_out = ins.x
        x_expect = x
        np.testing.assert_array_equal(x_out, x_expect)

    def test_position(self, ins):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.pi, np.pi / 2.0, np.pi / 4.0])
        ins.reset(x)

        p_out = ins.position()
        p_expect = np.array([1.0, 2.0, 3.0]).reshape(-1, 1)

        np.testing.assert_array_almost_equal(p_out, p_expect)

    def test_velocity(self, ins):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.pi, np.pi / 2.0, np.pi / 4.0])
        ins.reset(x)

        v_out = ins.velocity()
        v_expect = np.array([4.0, 5.0, 6.0]).reshape(-1, 1)

        np.testing.assert_array_almost_equal(v_out, v_expect)

    def test_attitude_rad(self, ins):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.pi, np.pi / 2.0, np.pi / 4.0])
        ins.reset(x)

        theta_out = ins.attitude(degrees=False)
        theta_expect = np.array([np.pi, np.pi / 2.0, np.pi / 4.0]).reshape(-1, 1)

        np.testing.assert_array_almost_equal(theta_out, theta_expect)

    def test_attitude_deg(self, ins):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.pi, np.pi / 2.0, np.pi / 4.0])
        ins.reset(x)

        theta_out = ins.attitude(degrees=True)
        theta_expect = np.array([180.0, 90.0, 45.0]).reshape(-1, 1)

        np.testing.assert_array_almost_equal(theta_out, theta_expect)

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

        x0_expect = np.zeros((9, 1))
        x1_expect = np.array(
            [0.005, 0.01, 0.015, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        ).reshape(-1, 1)

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

        x0_expect = np.zeros((9, 1))
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
        ).reshape(-1, 1)

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

        np.testing.assert_array_almost_equal(x0_out, x0_expect)
        np.testing.assert_array_almost_equal(x1_out, x1_expect)
        np.testing.assert_array_almost_equal(x2_out, x2_expect)

    def test_update_R_T(self):
        ins = StrapdownINS(np.zeros((9, 1)))

        h = 0.1
        g = ins._g
        f_imu = np.array([1.0, 2.0, 3.0]).reshape(-1, 1) - g
        w_imu = np.array([4.0, 5.0, 6.0])
        ins.update(h, f_imu, w_imu, theta_ext=(0.0, 0.0, 0.0))

        x_out = ins.x
        x_expect = np.array([0.005, 0.01, 0.015, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape(
            -1, 1
        )
        np.testing.assert_array_almost_equal(x_out, x_expect)

        ins.update(h, f_imu, w_imu, theta_ext=(0.0, 0.0, 0.0))

        x_out = ins.x
        x_expect = np.array([0.02, 0.04, 0.06, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]).reshape(
            -1, 1
        )
        np.testing.assert_array_almost_equal(x_out, x_expect)
