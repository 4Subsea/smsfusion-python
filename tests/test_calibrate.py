import numpy as np
import pytest

from smsfusion.calibrate import calibrate, decompose


class Test_calibrate:

    @pytest.mark.parametrize(
        "xyz_ref",
        [
            np.random.default_rng().random(size=(4, 3)),
            np.random.default_rng().random(size=(40, 3)),
        ],
    )
    def test_exact(self, xyz_ref):
        xyz_ref = xyz_ref / np.sqrt(np.sum(xyz_ref**2, axis=1))[:, np.newaxis]

        # Some typical calibration values
        W_expected = np.array(
            [
                [0.99304605, -0.00175537, -0.00554466],
                [0.00111272, 1.00574815, -0.01341676],
                [-0.00113683, 0.01124392, 0.983339],
            ]
        )
        bias_expected = np.array([-0.03336607, 0.00441664, -0.00619647])

        xyz = (np.linalg.inv(W_expected) @ (xyz_ref - bias_expected).T).T

        W_out, bias_out = calibrate(xyz_ref, xyz)
        np.testing.assert_almost_equal(W_expected, W_out)
        np.testing.assert_almost_equal(bias_expected, bias_out)

    @pytest.mark.parametrize(
        "xyz_ref",
        [
            np.random.default_rng().random(size=(4, 3)),
            np.random.default_rng().random(size=(40, 3)),
        ],
    )
    def test_exact_alternate(self, xyz_ref):
        xyz_ref = xyz_ref / np.sqrt(np.sum(xyz_ref**2, axis=1))[:, np.newaxis]

        # Some typical calibration values
        W_expected = np.array(
            [
                [0.99304605, -0.00175537, -0.00554466],
                [0.00111272, 1.00574815, -0.01341676],
                [-0.00113683, 0.01124392, 0.983339],
            ]
        )
        bias_expected = np.array([-0.03336607, 0.00441664, -0.00619647])

        xyz = (np.linalg.inv(W_expected) @ (xyz_ref - bias_expected).T).T

        W_out, bias_alt_out = calibrate(xyz_ref, xyz, bias_pre=True)
        np.testing.assert_almost_equal(W_expected, W_out)
        np.testing.assert_almost_equal(bias_expected, W_out @ bias_alt_out)

    @pytest.mark.parametrize(
        "xyz_ref",
        [
            np.random.default_rng().random(size=(400, 3)),
            np.random.default_rng().random(size=(4000, 3)),
        ],
    )
    def test_noisy(self, xyz_ref):
        xyz_ref = xyz_ref / np.sqrt(np.sum(xyz_ref**2, axis=1))[:, np.newaxis]

        # Some typical calibration values
        W_expected = np.array(
            [
                [0.99304605, -0.00175537, -0.00554466],
                [0.00111272, 1.00574815, -0.01341676],
                [-0.00113683, 0.01124392, 0.983339],
            ]
        )
        bias_expected = np.array([-0.03336607, 0.00441664, -0.00619647])

        xyz = (
            np.linalg.inv(W_expected) @ (xyz_ref - bias_expected).T
        ).T + np.random.default_rng().normal(size=xyz_ref.shape, scale=0.001)

        W_out, bias_out = calibrate(xyz_ref, xyz)
        np.testing.assert_almost_equal(W_expected, W_out, decimal=3)
        np.testing.assert_almost_equal(bias_expected, bias_out, decimal=3)

    @pytest.mark.parametrize(
        "xyz_ref",
        [
            np.random.default_rng().random(size=(4, 4)),
            np.random.default_rng().random(size=(40, 2)),
        ],
    )
    def test_shape_wrong(self, xyz_ref):
        xyz = np.ones_like(xyz_ref)
        with pytest.raises(ValueError):
            _ = calibrate(xyz_ref, xyz)

    def test_too_few_data_points(self):
        xyz_ref = np.random.default_rng().random(size=(3, 3))
        # Some typical calibration values
        W_expected = np.array(
            [
                [0.99304605, -0.00175537, -0.00554466],
                [0.00111272, 1.00574815, -0.01341676],
                [-0.00113683, 0.01124392, 0.983339],
            ]
        )
        bias_expected = np.array([-0.03336607, 0.00441664, -0.00619647])

        xyz = (np.linalg.inv(W_expected) @ (xyz_ref - bias_expected).T).T

        with pytest.raises(ValueError):
            _ = calibrate(xyz_ref, xyz)

    @pytest.mark.parametrize(
        "xyz_ref",
        [
            np.random.default_rng().random(size=(20, 3)),
            np.random.default_rng().random(size=(40, 3)),
        ],
    )
    def test_shape_mismatch(self, xyz_ref):
        xyz = np.ones_like(xyz_ref)[:-1]
        with pytest.raises(ValueError):
            _ = calibrate(xyz_ref, xyz)


class Test_decompose:
    def test_decompose(self):

        # Some typical calibration values
        W_expected = np.array(
            [
                [0.99304605, -0.00175537, -0.00554466],
                [0.00111272, 1.00574815, -0.01341676],
                [-0.00113683, 0.01124392, 0.983339],
            ]
        )

        R, S = decompose(W_expected)
        W_out = R @ S

        assert R.shape == S.shape
        assert R.shape == (3, 3)
        np.testing.assert_almost_equal(W_expected, W_out)

    def test_wrong_shape(self):

        # Some typical calibration values
        W_expected = np.array(
            [
                [0.99304605, -0.00175537],
                [0.00111272, 1.00574815],
                [-0.00113683, 0.01124392],
            ]
        )

        with pytest.raises(ValueError):
            _ = decompose(W_expected)
