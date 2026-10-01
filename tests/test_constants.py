from pytest import approx

from smsfusion import constants


def test_imu_noise_params():
    assert constants.ACC_NOISE_DENSITY == approx(0.0007)
    assert constants.ACC_BIAS_STABILITY == approx(0.0005)
    assert constants.ACC_BIAS_CORR_TIME == approx(50.0)
    assert constants.GYRO_NOISE_DENSITY == approx(0.00005)
    assert constants.GYRO_BIAS_STABILITY == approx(0.00005)
    assert constants.GYRO_BIAS_CORR_TIME == approx(50.0)
