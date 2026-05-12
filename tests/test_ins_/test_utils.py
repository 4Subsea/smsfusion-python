import numpy as np
import pytest

from smsfusion._ins import _utils
from smsfusion._transforms import _rot_matrix_from_quaternion, quaternion_from_euler


@pytest.mark.parametrize(
    "euler",
    [
        np.radians([10.0, 45.0, 0.0]),
        np.radians([0.0, 0.0, -10.0]),
        np.radians([90.0, 0.0, -45.0]),
        np.radians([180.0, 0.0, 10.0]),
        np.radians([130.0, -28.0, 90.0]),
    ],
)
def test_euler_from_acc(euler):
    R_nm = _rot_matrix_from_quaternion(quaternion_from_euler(euler))  # body-to-nav
    g = _utils.gravity()
    euler_degrees = np.degrees(euler)

    # North-East-Down (NED) frame
    g_ned = np.array([0.0, 0.0, -g])
    acc_ned = R_nm.T @ g_ned
    euler_ned = _utils.euler_from_acc(acc_ned, nav_frame="NED", yaw=euler[2])
    np.testing.assert_allclose(euler_ned, euler)

    euler_ned = _utils.euler_from_acc(acc_ned, nav_frame="NED", yaw=euler[2], degrees=True)
    np.testing.assert_allclose(euler_ned, euler_degrees)


    # North-East-Up (ENU) frame
    g_enu = np.array([0.0, 0.0, g])
    acc_enu = R_nm.T @ g_enu
    euler_enu = _utils.euler_from_acc(acc_enu, nav_frame="ENU", yaw=euler[2])
    np.testing.assert_allclose(euler_enu, euler)

    euler_enu = _utils.euler_from_acc(acc_enu, nav_frame="ENU", yaw=euler[2], degrees=True)
    np.testing.assert_allclose(euler_enu, euler_degrees)


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
    g_out = _utils.gravity(mu)
    assert g_out == pytest.approx(g_expect)


class Test_FixedNed:
    def test_init(self):
        _ = _utils.FixedNED(0.0, 0.0, 0.0)

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
        ned = _utils.FixedNED(0.0, 0.0, 0.0)

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
        ned = _utils.FixedNED(0.0, 0.0, 0.0)

        lat_, lon_, height_ = ned.to_llh(x, y, z)
        assert lat_ == lat
        assert lon_ == lon
        assert height_ == height
