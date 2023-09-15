import numpy as np
import pytest

from smsfusion import _ahrs


class Test_AHRS:
    def test_init(self):
        fs = 10.24
        Kp = 0.5
        Ki = 0.1
        q_init = np.array([1.0, 2.0, 3.0, 4.0]) / np.sqrt(30.0)
        bias_init = np.array([0.1, 0.2, 0.3])

        alg = _ahrs.AHRS(fs, Kp, Ki, q_init=q_init, bias_init=bias_init)

        assert isinstance(alg, _ahrs.AHRS)
        assert alg._fs == fs
        assert alg._dt == 1.0 / fs
        assert alg._Kp == Kp
        assert alg._Ki == Ki
        np.testing.assert_array_almost_equal(alg.q, q_init)
        np.testing.assert_array_almost_equal(alg.bias, bias_init)
        np.testing.assert_array_almost_equal(alg.error, np.array([0.0, 0.0, 0.0]))

    def test_q_init(self):
        fs = 10.24
        Kp = 0.5
        Ki = 0.1
        q_init = np.array([0.96591925, -0.25882081, 0.0, 0.0], dtype=float)

        alg = _ahrs.AHRS(fs, Kp, Ki, q_init=q_init)

        q_expect = q_init
        np.testing.assert_array_almost_equal(alg.q, q_expect)

    def test_q_init_wrong_len(self):
        fs = 10.24
        Kp = 0.5
        Ki = 0.1
        q_init = np.array([0.96591925, -0.25882081, 0.0, 0.0, 0.0], dtype=float)

        with pytest.raises(ValueError):
            _ahrs.AHRS(fs, Kp, Ki, q_init=q_init)

    def test_q_init_notunity(self):
        fs = 10.24
        Kp = 0.5
        Ki = 0.1
        q_init = np.array([2, -1, 0.0, 0.0], dtype=float)

        with pytest.warns():
            _ahrs.AHRS(fs, Kp, Ki, q_init=q_init)

    def test_q_init_none(self):
        fs = 10.24
        Kp = 0.5
        Ki = 0.1
        q_init = None

        alg = _ahrs.AHRS(fs, Kp, Ki, q_init=q_init)

        q_expect = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(alg.q, q_expect)

    def test_bias_init(self):
        fs = 10.24
        Kp = 0.5
        Ki = 0.1
        bias_init = np.array([0.96591925, -0.25882081, 0.0], dtype=float)

        alg = _ahrs.AHRS(fs, Kp, Ki, bias_init=bias_init)

        bias_expect = bias_init
        np.testing.assert_array_almost_equal(alg.bias, bias_expect)

    def test_bias_init_none(self):
        fs = 10.24
        Kp = 0.5
        Ki = 0.1
        bias_init = None

        alg = _ahrs.AHRS(fs, Kp, Ki, bias_init=bias_init)

        bias_expect = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(alg.bias, bias_expect)

    def test_bias_init_wrong_len(self):
        fs = 10.24
        Kp = 0.5
        Ki = 0.1
        bias_init = np.array([0.96591925, -0.25882081, 0.0, 0.0], dtype=float)

        with pytest.raises(ValueError):
            _ahrs.AHRS(fs, Kp, Ki, bias_init=bias_init)

    def test_attitude(self):
        fs = 10.24
        Kp = 0.5
        Ki = 0.1
        q_init = np.array([0.96591925, -0.25882081, 0.0, 0.0], dtype=float)
        alg = _ahrs.AHRS(fs, Kp, Ki, q_init=q_init)

        alpha_beta_gamma = np.array([-30.0, 0.0, 0.0], dtype=float)
        np.testing.assert_array_almost_equal(
            alg.attitude(), alpha_beta_gamma, decimal=3
        )
        np.testing.assert_array_almost_equal(
            alg.attitude(degrees=True), alpha_beta_gamma, decimal=3
        )
        np.testing.assert_array_almost_equal(
            alg.attitude(degrees=False), np.radians(alpha_beta_gamma), decimal=3
        )

    def test__update_Ki(self):
        dt = 0.1
        q = np.array([1.0, 0.0, 0.0, 0.0])
        bias = np.array([0.5, 0.0, -0.5])
        omega_gyro = np.array([1.0, 2.0, 3.0])
        omega_corr = np.array([0.1, 0.2, -0.1])
        Ki = 0.5
        Kp = 0.0

        q, bias, error = _ahrs.AHRS._update(dt, q, bias, omega_gyro, omega_corr, Kp, Ki)

        q_expect = np.array([0.979986, 0.0246221, 0.0982436, 0.171375])
        error_expect = omega_corr
        bias_expect = np.array([0.4975, -0.005, -0.4975])

        np.testing.assert_almost_equal(bias, bias_expect)
        np.testing.assert_almost_equal(error, error_expect)
        np.testing.assert_almost_equal(q, q_expect)

    def test__update_Kp(self):
        dt = 0.1
        q = np.array([1.0, 0.0, 0.0, 0.0])
        bias = np.array([-0.5, 0.0, 0.5])
        omega_gyro = np.array([1.0, 2.0, 3.0])
        omega_corr = np.array([0.1, 0.2, -0.1])
        Ki = 0.0
        Kp = 0.5

        q, bias, error = _ahrs.AHRS._update(dt, q, bias, omega_gyro, omega_corr, Kp, Ki)

        q_expect = np.array([0.9843562, 0.0762876, 0.1033574, 0.1205836])
        error_expect = omega_corr
        bias_expect = np.array([-0.5, 0.0, 0.5])

        np.testing.assert_almost_equal(bias, bias_expect)
        np.testing.assert_almost_equal(error, error_expect)
        np.testing.assert_almost_equal(q, q_expect)

    @pytest.mark.parametrize(
        "degrees, head_degrees", [[True, True], [False, True], [True, False]]
    )
    def test_update(self, degrees, head_degrees):
        fs = 10.24
        Kp = 0.5
        Ki = 0.1
        q_init = np.array([1.0, 0.0, 0.0, 0.0])
        bias_init = np.array([0.0, 0.0, 0.0])

        alg = _ahrs.AHRS(fs, Kp, Ki, q_init=q_init, bias_init=bias_init)

        f_imu = np.array([0.3422471493375811, -0.0, -9.800676053786814])
        w_imu = np.array([1.0, 0.0, 0.0])
        head = 30.0

        if not degrees:
            w_imu = np.radians(w_imu)
        if not head_degrees:
            head = np.radians(head)

        alg.update(f_imu, w_imu, head, degrees=degrees, head_degrees=head_degrees)

        q_expect = np.array([9.999233e-01, 8.521462e-04, 8.602932e-04, 1.232530e-02])
        error_expect = np.array([0.0, 0.034899, 0.5])
        bias_expect = np.array([0.0, -0.000170, -0.002441])

        np.testing.assert_array_almost_equal(alg.q, q_expect)
        np.testing.assert_array_almost_equal(alg.error, error_expect)
        np.testing.assert_array_almost_equal(alg.bias, bias_expect)

    @pytest.mark.parametrize(
        "f_imu, w_imu, head",
        [
            [np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 0]), 0.0],
            [np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 0]), None],
            [np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 0]), np.array([0.0])],
            [[0.0, 0.0, -1.0], [0.0, 0.0, 0], 0.0],
            [
                np.array([[0.0], [0.0], [-1.0]]),
                np.array([[0.0], [0.0], [0]]),
                np.array([[0.0]]),
            ],
        ],
    )
    def test_update_input_valid(self, f_imu, w_imu, head):
        fs = 10.24
        Kp = 0.5
        Ki = 0.1
        q_init = np.array([1.0, 0.0, 0.0, 0.0])
        bias_init = np.array([0.0, 0.0, 0.0])

        alg = _ahrs.AHRS(fs, Kp, Ki, q_init=q_init, bias_init=bias_init)
        alg.update(f_imu, w_imu, head)

        q_expect = np.array([1.0, 0.0, 0.0, 0.0])
        error_expect = np.array([0.0, 0.0, 0.0])
        bias_expect = np.array([0.0, 0.0, 0.0])

        np.testing.assert_array_almost_equal(alg.q, q_expect)
        np.testing.assert_array_almost_equal(alg.error, error_expect)
        np.testing.assert_array_almost_equal(alg.bias, bias_expect)

    @pytest.mark.parametrize(
        "f_imu, w_imu, head",
        [
            [np.array([0.0, 0.0, -1.0, 0.0]), np.array([0.0, 0.0, 0.0]), 0.0],
            [
                np.array([0.0, 0.0, -1.0]),
                np.array([0.0, 0.0, 0.0, 0.0]),
                np.array([0.0]),
            ],
            [[0.0, 0.0, -1.0], [0.0, 0.0, 0], [0.0, 0.0]],
        ],
    )
    def test_update_input_invalid(self, f_imu, w_imu, head):
        fs = 10.24
        Kp = 0.5
        Ki = 0.1
        q_init = np.array([1.0, 0.0, 0.0, 0.0])
        bias_init = np.array([0.0, 0.0, 0.0])

        alg = _ahrs.AHRS(fs, Kp, Ki, q_init=q_init, bias_init=bias_init)
        with pytest.raises((TypeError, ValueError)):
            alg.update(f_imu, w_imu, head)

    def test_update_succesive_calls(self):
        """Test that succesive calls goes through"""
        fs = 10
        Kp = 0.5
        Ki = 0.1

        f_imu = np.random.random((100, 3))
        w_imu = np.random.random((100, 3))
        head = np.random.random(100)
        head[::10] = None

        alg = _ahrs.AHRS(fs, Kp, Ki)
        _ = [
            alg.update(f_imu_i, w_imu_i, head_i).q
            for f_imu_i, w_imu_i, head_i in zip(f_imu, w_imu, head)
        ]
