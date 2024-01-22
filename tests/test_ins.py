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

from smsfusion._ins import AHRS, StrapdownINS, _signed_smallest_angle, gravity
from smsfusion._transforms import (
    _angular_matrix_from_euler,
    _angular_matrix_from_quaternion,
    _quaternion_from_euler,
    _rot_matrix_from_euler,
    _rot_matrix_from_quaternion,
)
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
class Test_StrapdownINS:
    @pytest.fixture
    def ins(self):
        x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        ins = StrapdownINS(x0)
        return ins

    def test__init__(self):
        x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0])
        ins = StrapdownINS(x0)

        np.testing.assert_array_equal(ins._x0, x0.reshape(-1, 1))
        np.testing.assert_array_equal(ins._x, x0.reshape(-1, 1))

    def test_x(self):
        x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0])
        ins = StrapdownINS(x0)

        x_out = ins.x
        x_expect = x0

        assert x_out.shape == (10,)
        assert x_out is not ins._x
        np.testing.assert_array_equal(x_out, x_expect)

    def test_position(self):
        x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0])
        ins = StrapdownINS(x0)

        p_out = ins.position()
        p_expect = np.array([1.0, 2.0, 3.0])

        assert p_out.shape == (3,)
        assert p_out is not ins._p
        np.testing.assert_array_equal(p_out, p_expect)

    def test_velocity(self):
        x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0])
        ins = StrapdownINS(x0)

        v_out = ins.velocity()
        v_expect = np.array([4.0, 5.0, 6.0])

        assert v_out.shape == (3,)
        assert v_out is not ins._v
        np.testing.assert_array_equal(v_out, v_expect)

    def test_quaternion(self):
        x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0])
        ins = StrapdownINS(x0)

        q_out = ins.quaternion()
        q_expect = np.array([1.0, 0.0, 0.0, 0.0])

        assert q_out.shape == (4,)
        assert q_out is not ins._q
        np.testing.assert_array_equal(q_out, q_expect)

    def test_euler(self):
        x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0])
        ins = StrapdownINS(x0)

        theta_out = ins.euler()
        theta_expect = np.array([0.0, 0.0, 0.0])

        assert theta_out.shape == (3,)
        np.testing.assert_array_equal(theta_out, theta_expect)

    def test_reset(self, ins):
        x = np.random.random(10)
        x[6:10] = x[6:10] / np.linalg.norm(x[6:10])  # unit quaternion
        ins.reset(x)

        np.testing.assert_array_almost_equal(ins.x, x)

    def test_reset_2d(self, ins):
        x = np.random.random(10).reshape(-1, 1)
        x[6:10] = x[6:10] / np.linalg.norm(x[6:10])  # unit quaternion
        ins.reset(x)

        np.testing.assert_array_almost_equal(ins.x, x.flatten())

    def test_update_return_self(self, ins):
        dt = 0.1
        g = 9.80665
        f = np.array([0.0, 0.0, -g]).reshape(-1, 1)
        w = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)

        update_return = ins.update(dt, f, w)
        assert update_return is ins

    def test_update(self, ins):
        h = 0.1
        g = ins._g
        f = np.array([1.0, 2.0, 3.0]).reshape(-1, 1) - g
        w = np.array([0.04, 0.05, 0.06]).reshape(-1, 1)

        x0_out = ins.x
        ins.update(h, f, w)
        x1_out = ins.x

        x0_expect = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        x1_expect = np.array(
            [0.005, 0.01, 0.015, 0.1, 0.2, 0.3, 0.99999, 0.002, 0.0025, 0.003]
        )

        np.testing.assert_array_almost_equal(x0_out, x0_expect)
        np.testing.assert_array_almost_equal(x1_out, x1_expect)

    def test_update_deg(self, ins):
        dt = 0.1
        g = ins._g
        f_imu = np.array([1.0, 2.0, 3.0]).reshape(-1, 1) - g
        w_imu = np.array([4.0, 5.0, 6.0]).reshape(-1, 1)

        x0_out = ins.x
        ins.update(dt, f_imu, w_imu, degrees=True)
        x1_out = ins.x

        x0_expect = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        x1_expect = np.array(
            [0.005, 0.01, 0.015, 0.1, 0.2, 0.3, 0.999971, 0.003491, 0.004363, 0.005236]
        )

        np.testing.assert_array_almost_equal(x0_out, x0_expect)
        np.testing.assert_array_almost_equal(x1_out, x1_expect)

    def test_update_twise(self, ins):
        dt = 0.1
        g = ins._g
        f_imu = np.array([1.0, 2.0, 3.0]).reshape(-1, 1) - g
        w_imu = np.array([0.004, 0.005, 0.006]).reshape(-1, 1)

        x0_out = ins.x
        ins.update(dt, f_imu, w_imu, degrees=False)
        x1_out = ins.x
        ins.update(dt, f_imu, w_imu, degrees=False)
        x2_out = ins.x

        x0_expect = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ).reshape(-1, 1)

        # Calculate x1
        R0_expect = np.eye(3)
        T0_expect = _angular_matrix_from_quaternion(x0_expect[6:10].flatten())
        a0_expect = R0_expect @ f_imu + g
        x1_expect = np.zeros((10, 1))
        x1_expect[0:3] = (
            x0_expect[0:3] + dt * x0_expect[3:6] + 0.5 * dt**2 * a0_expect
        )
        x1_expect[3:6] = x0_expect[3:6] + dt * a0_expect
        x1_expect[6:10] = x0_expect[6:10] + dt * T0_expect @ w_imu
        x1_expect[6:10] = x1_expect[6:10] / np.linalg.norm(x1_expect[6:10])

        # Calculate x2 by forward Euler
        R1_expect = _rot_matrix_from_quaternion(x1_expect[6:10].flatten())
        T1_expect = _angular_matrix_from_quaternion(x1_expect[6:10].flatten())
        a1_expect = R1_expect @ f_imu + g
        x2_expect = np.zeros((10, 1))
        x2_expect[0:3] = (
            x1_expect[0:3] + dt * x1_expect[3:6] + 0.5 * dt**2 * a1_expect
        )
        x2_expect[3:6] = x1_expect[3:6] + dt * a1_expect
        x2_expect[6:10] = x1_expect[6:10] + dt * T1_expect @ w_imu
        x2_expect[6:10] = x2_expect[6:10] / np.linalg.norm(x2_expect[6:10])

        np.testing.assert_array_almost_equal(x0_out, x0_expect.flatten())
        np.testing.assert_array_almost_equal(x1_out, x1_expect.flatten())
        np.testing.assert_array_almost_equal(x2_out, x2_expect.flatten())
