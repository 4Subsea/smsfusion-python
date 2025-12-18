from pathlib import Path
from turtle import down

import numpy as np
import pytest

import smsfusion as sf

TEST_PATH = Path(__file__).parent


@pytest.fixture
def data_ag():
    """
    200 Hz AG data.
    """

    data = np.genfromtxt(
        TEST_PATH
        / "testdata/coning_sculling/coning_sculling_sim_highfreq_20251218A.csv",
        delimiter=",",
        names=True,
        dtype=float,
    )

    gx = data["Gx_rads"]
    gy = data["Gy_rads"]
    gz = data["Gz_rads"]
    ax = data["Ax_ms2"]
    ay = data["Ay_ms2"]
    az = data["Az_ms2"]

    w = np.column_stack((gx, gy, gz))
    f = np.column_stack((ax, ay, az))

    return f, w


@pytest.fixture
def data_dtheta_dvel():
    """
    10 Hz coning/sculling reference data.
    """

    data = np.genfromtxt(
        TEST_PATH
        / "testdata/coning_sculling/coning_sculling_sim_lowfreq_20251218A.csv",
        delimiter=",",
        names=True,
        dtype=float,
    )

    dtheta_x = data["dThetaX_rad"]
    dtheta_y = data["dThetaY_rad"]
    dtheta_z = data["dThetaZ_rad"]
    dvel_x = data["dVelX_ms"]
    dvel_y = data["dVelY_ms"]
    dvel_z = data["dVelZ_ms"]

    dtheta = np.column_stack((dtheta_x, dtheta_y, dtheta_z))
    dvel = np.column_stack((dvel_x, dvel_y, dvel_z))

    return dvel, dtheta


class Test_ConingScullingAlg:

    def test__init__(self):
        alg = sf.ConingScullingAlg(256.0)

        alg._fs == 256.0
        alg._dt == 1.0 / 256.0
        np.testing.assert_allclose(alg._theta, np.zeros(3))
        np.testing.assert_allclose(alg._dtheta_con, np.zeros(3))
        np.testing.assert_allclose(alg._dtheta_prev, np.zeros(3))
        np.testing.assert_allclose(alg._vel, np.zeros(3))
        np.testing.assert_allclose(alg._dvel_scul, np.zeros(3))
        np.testing.assert_allclose(alg._dv_prev, np.zeros(3))

    def test_update(self, data_ag, data_dtheta_dvel):
        f, w = data_ag
        dvel_ref, dtheta_ref = data_dtheta_dvel

        fs_highfreq = 200.0
        fs_lowfreq = 10.0
        step = int(fs_highfreq / fs_lowfreq)
        alg = sf.ConingScullingAlg(200.0)

        dtheta_out = []
        dvel_out = []
        for i, (w_i, f_i) in enumerate(zip(w, f)):

            alg.update(f_i, w_i)

            if (i != 0) and (i % step == 0.0):
                dtheta_i, dvel_i = alg.flush()
                dtheta_out.append(dtheta_i)
                dvel_out.append(dvel_i)

        dtheta_out = np.array(dtheta_out)
        dvel_out = np.array(dvel_out)

        np.testing.assert_allclose(dvel_out, dvel_ref, atol=1e-8)
        np.testing.assert_allclose(dtheta_out, dtheta_ref, atol=1e-8)

    def test_update_pure_roll(self):
        fs = 100.0
        alg = sf.ConingScullingAlg(fs)

        f = np.array([0.0, 0.0, 0.0])  # m/s^2
        w = np.array([np.radians(90.0), 0.0, 0.0])  # rad/s

        for i in range(int(fs * 1.0)):  # 1 second
            alg.update(f, w)

        dtheta_out, dvel_out = alg.flush()

        dtheta_expect = np.array([np.radians(90.0), 0.0, 0.0])
        np.testing.assert_allclose(dtheta_out, dtheta_expect)
        np.testing.assert_allclose(dvel_out, np.zeros(3))

    def test_update_pure_pitch(self):
        fs = 100.0
        alg = sf.ConingScullingAlg(fs)

        f = np.array([0.0, 0.0, 0.0])  # m/s^2
        w = np.array([0.0, np.radians(90.0), 0.0])  # rad/s

        for i in range(int(fs * 1.0)):  # 1 second
            alg.update(f, w)

        dtheta_out, dvel_out = alg.flush()

        dtheta_expect = np.array([0.0, np.radians(90.0), 0.0])
        np.testing.assert_allclose(dtheta_out, dtheta_expect)
        np.testing.assert_allclose(dvel_out, np.zeros(3))

    def test_update_pure_yaw(self):
        fs = 100.0
        alg = sf.ConingScullingAlg(fs)

        f = np.array([0.0, 0.0, 0.0])  # m/s^2
        w = np.array([0.0, 0.0, np.radians(90.0)])  # rad/s

        for i in range(int(fs * 1.0)):  # 1 second
            alg.update(f, w)

        dtheta_out, dvel_out = alg.flush()

        dtheta_expect = np.array([0.0, 0.0, np.radians(90.0)])
        np.testing.assert_allclose(dtheta_out, dtheta_expect)
        np.testing.assert_allclose(dvel_out, np.zeros(3))

    def test_pure_surge(self):
        fs = 100.0
        alg = sf.ConingScullingAlg(fs)

        f = np.array([1.0, 0.0, 0.0])  # m/s^2
        w = np.array([0.0, 0.0, 0.0])  # rad/s

        for i in range(int(fs * 1.0)):  # 1 second
            alg.update(f, w)

        dtheta_out, dvel_out = alg.flush()

        dvel_expect = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(dvel_out, dvel_expect)
        np.testing.assert_allclose(dtheta_out, np.zeros(3))

    def test_pure_sway(self):
        fs = 100.0
        alg = sf.ConingScullingAlg(fs)

        f = np.array([0.0, 1.0, 0.0])  # m/s^2
        w = np.array([0.0, 0.0, 0.0])  # rad/s

        for i in range(int(fs * 1.0)):  # 1 second
            alg.update(f, w)

        dtheta_out, dvel_out = alg.flush()

        dvel_expect = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(dvel_out, dvel_expect)
        np.testing.assert_allclose(dtheta_out, np.zeros(3))

    def test_pure_heave(self):
        fs = 100.0
        alg = sf.ConingScullingAlg(fs)

        f = np.array([0.0, 0.0, 1.0])  # m/s^2
        w = np.array([0.0, 0.0, 0.0])  # rad/s

        for i in range(int(fs * 1.0)):  # 1 second
            alg.update(f, w)

        dtheta_out, dvel_out = alg.flush()

        dvel_expect = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(dvel_out, dvel_expect)
        np.testing.assert_allclose(dtheta_out, np.zeros(3))

    def test_roll_and_surge(self):
        fs = 100.0
        alg = sf.ConingScullingAlg(fs)

        f = np.array([1.0, 0.0, 0.0])  # m/s^2
        w = np.array([np.radians(90.0), 0.0, 0.0])  # rad/s

        for i in range(int(fs * 1.0)):  # 1 second
            alg.update(f, w)

        dtheta_out, dvel_out = alg.flush()

        dtheta_expect = np.array([np.radians(90.0), 0.0, 0.0])
        dvel_expect = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(dtheta_out, dtheta_expect, atol=1e-8)
        np.testing.assert_allclose(dvel_out, dvel_expect, atol=1e-8)

    def test_pitch_and_sway(self):
        fs = 100.0
        alg = sf.ConingScullingAlg(fs)

        f = np.array([0.0, 1.0, 0.0])  # m/s^2
        w = np.array([0.0, np.radians(90.0), 0.0])  # rad/s

        for i in range(int(fs * 1.0)):  # 1 second
            alg.update(f, w)

        dtheta_out, dvel_out = alg.flush()

        dtheta_expect = np.array([0.0, np.radians(90.0), 0.0])
        dvel_expect = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(dtheta_out, dtheta_expect, atol=1e-8)
        np.testing.assert_allclose(dvel_out, dvel_expect, atol=1e-8)

    def test_yaw_and_heave(self):
        fs = 100.0
        alg = sf.ConingScullingAlg(fs)

        f = np.array([0.0, 0.0, 1.0])  # m/s^2
        w = np.array([0.0, 0.0, np.radians(90.0)])  # rad/s

        for i in range(int(fs * 1.0)):  # 1 second
            alg.update(f, w)

        dtheta_out, dvel_out = alg.flush()

        dtheta_expect = np.array([0.0, 0.0, np.radians(90.0)])
        dvel_expect = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(dtheta_out, dtheta_expect, atol=1e-8)
        np.testing.assert_allclose(dvel_out, dvel_expect, atol=1e-8)
