from pathlib import Path

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

    @pytest.mark.parametrize(
        "algorithm", [sf.ConingScullingAlg, sf.ConingScullingAlgCalibrated]
    )
    def test_update(self, data_ag, data_dtheta_dvel, algorithm):
        f, w = data_ag
        dvel_ref, dtheta_ref = data_dtheta_dvel

        fs_highfreq = 200.0
        fs_lowfreq = 10.0
        step = int(fs_highfreq / fs_lowfreq)
        alg = algorithm(200.0)

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

    def test_roll_pitch_yaw(self):
        fs = 100.0
        alg = sf.ConingScullingAlg(fs)

        f = np.array([0.0, 0.0, 0.0])  # m/s^2
        w = np.array([np.radians(30.0), -np.radians(45.0), np.radians(60.0)])  # rad/s

        for i in range(int(fs * 1.0)):  # 1 second
            alg.update(f, w)

        dtheta_out, dvel_out = alg.flush()

        dtheta_expect = np.array(
            [np.radians(30.0), -np.radians(45.0), np.radians(60.0)]
        )
        np.testing.assert_allclose(dtheta_out, dtheta_expect, atol=1e-8)
        np.testing.assert_allclose(dvel_out, np.zeros(3))

    def test_surge_sway_heave(self):
        fs = 100.0
        alg = sf.ConingScullingAlg(fs)

        f = np.array([1.0, -2.0, 3.0])  # m/s^2
        w = np.array([0.0, 0.0, 0.0])  # rad/s

        for i in range(int(fs * 1.0)):  # 1 second
            alg.update(f, w)

        dtheta_out, dvel_out = alg.flush()

        dvel_expect = np.array([1.0, -2.0, 3.0])
        np.testing.assert_allclose(dvel_out, dvel_expect)
        np.testing.assert_allclose(dtheta_out, np.zeros(3))

    def test_flush(self):
        fs = 100.0
        alg = sf.ConingScullingAlg(fs)

        f = np.array([1.0, 2.0, 3.0])  # m/s^2
        w = np.array([np.radians(30.0), -np.radians(45.0), np.radians(60.0)])  # rad/s

        for i in range(int(fs * 1.0)):  # 1 second
            alg.update(f, w)

        dtheta_out, dvel_out = alg.flush()
        # Check that flush returns non-zero values
        assert np.all(np.abs(dtheta_out) > 0.1)
        assert np.all(np.abs(dvel_out) > 0.1)

        # Flushing again should yield all zeros
        dtheta_out, dvel_out = alg.flush()
        np.testing.assert_allclose(dtheta_out, np.zeros(3))
        np.testing.assert_allclose(dvel_out, np.zeros(3))

    def test_verify_calibration(self, data_ag, data_dtheta_dvel):
        """Tests that the algorithm with built-in calibration produces the same results as the
        uncalibrated algorithm with manual calibration applied to the inputs.
        """
        rng = np.random.default_rng(seed=42)
        FS_HIGH = 200.0
        FS_LOW = 10.0
        downsample_factor = int(FS_HIGH / FS_LOW)
        # Calibration matrices and biases
        A1, A2 = rng.random((3, 3)), rng.random((3, 3))
        b_w, b_f = rng.random(3) * np.radians(0.1), rng.random(3)
        W_w = A1 @ A1.T + np.eye(3)  # Positive semi-definite
        W_f = A2 @ A2.T + np.eye(3)  # Positive semi-definite
        W_f_inv, W_w_inv = np.linalg.inv(W_f), np.linalg.inv(W_w)

        f_true, w_true = data_ag
        dvel_true, dtheta_true = data_dtheta_dvel

        # Generate measurements with scaling, misalignment and bias
        f_meas = np.empty_like(f_true)
        w_meas = np.empty_like(w_true)
        for i in range(len(f_true)):
            f_meas[i] = W_f_inv @ (f_true[i] - b_f)
            w_meas[i] = W_w_inv @ (w_true[i] - b_w)

        # Generate measurements with scaling, misalignment and bias - alternative bias method
        f_meas_alt = np.empty_like(f_true)
        w_meas_alt = np.empty_like(w_true)
        for i in range(len(f_true)):
            f_meas_alt[i] = W_f_inv @ f_true[i] - b_f
            w_meas_alt[i] = W_w_inv @ w_true[i] - b_w

        # Calculate dtheta and dvel when calibrating each measurement manually before feeding to the uncalibrated algorithm
        alg_naive_calibration = sf.ConingScullingAlg(FS_HIGH)
        dtheta_naive, dvel_naive = [], []
        for i, (f_i, w_i) in enumerate(zip(f_meas, w_meas)):
            # Apply calibration to measurements
            f_i_naive = W_f @ (f_i) + b_f
            w_i_naive = W_w @ (w_i) + b_w

            alg_naive_calibration.update(f_i_naive, w_i_naive)
            if (i != 0) and (i % downsample_factor == 0):
                dtheta_i, dvel_i = alg_naive_calibration.flush()
                dtheta_naive.append(dtheta_i)
                dvel_naive.append(dvel_i)
        dtheta_naive = np.array(dtheta_naive)
        dvel_naive = np.array(dvel_naive)

        # Calculate dtheta and dvel using the built-in calibration algorithm
        alg_calibrated = sf.ConingScullingAlgCalibrated(
            FS_HIGH, W_w=W_w, W_f=W_f, b_w=b_w, b_f=b_f
        )
        dtheta_calibrated, dvel_calibrated = [], []
        for i, (f_i, w_i) in enumerate(zip(f_meas, w_meas)):
            alg_calibrated.update(f_i, w_i)
            if (i != 0) and (i % downsample_factor == 0):
                dtheta_i, dvel_i = alg_calibrated.flush()
                dtheta_calibrated.append(dtheta_i)
                dvel_calibrated.append(dvel_i)
        dtheta_calibrated = np.array(dtheta_calibrated)
        dvel_calibrated = np.array(dvel_calibrated)

        # Calculate dtheta and dvel using the built-in calibration algorithm - alternative bias method
        alg_calibrated_alt = sf.ConingScullingAlgCalibrated(
            FS_HIGH, W_w=W_w, W_f=W_f, b_w=b_w, b_f=b_f, bias_alt=True
        )
        dtheta_calibrated_alt, dvel_calibrated_alt = [], []
        for i, (f_i, w_i) in enumerate(zip(f_meas_alt, w_meas_alt)):
            alg_calibrated_alt.update(f_i, w_i)
            if (i != 0) and (i % downsample_factor == 0):
                dtheta_i, dvel_i = alg_calibrated_alt.flush()
                dtheta_calibrated_alt.append(dtheta_i)
                dvel_calibrated_alt.append(dvel_i)
        dtheta_calibrated_alt = np.array(dtheta_calibrated_alt)
        dvel_calibrated_alt = np.array(dvel_calibrated_alt)

        # Test that the different methods match
        np.testing.assert_allclose(dtheta_calibrated, dtheta_true, atol=1e-8)
        np.testing.assert_allclose(dvel_calibrated, dvel_true, atol=1e-8)
        np.testing.assert_allclose(dtheta_calibrated_alt, dtheta_true, atol=1e-8)
        np.testing.assert_allclose(dvel_calibrated_alt, dvel_true, atol=1e-8)
        np.testing.assert_allclose(dtheta_naive, dtheta_true, atol=1e-8)
        np.testing.assert_allclose(dvel_naive, dvel_true, atol=1e-8)

    @pytest.mark.parametrize(
        "w_singular, f_singular", [(True, False), (False, True), (True, True)]
    )
    def test_raises_singular_matrix(self, w_singular, f_singular):
        fs = 100.0
        if w_singular:
            W_w = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        else:
            W_w = np.eye(3)
        if f_singular:
            W_f = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        else:
            W_f = np.eye(3)
        b_w = np.zeros(3)
        b_f = np.zeros(3)
        with pytest.raises(ValueError, match="must be invertible"):
            sf.ConingScullingAlgCalibrated(
                fs, W_w=W_w, W_f=W_f, b_w=b_w, b_f=b_f, bias_alt=False
            )
        if w_singular:
            with pytest.raises(ValueError, match="must be invertible"):
                sf.ConingScullingAlgCalibrated(
                    fs, W_w=W_w, W_f=W_f, b_w=b_w, b_f=b_f, bias_alt=True
                )
