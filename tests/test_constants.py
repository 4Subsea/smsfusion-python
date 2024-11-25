from pytest import approx

from smsfusion import constants


def test_ERR_ACC_MOTION1():
    err_expect = {
        "N": 0.004,
        "B": 0.0007,
        "tau_cb": 50.0,
    }

    err_out = constants.ERR_ACC_MOTION1

    assert set(err_out.keys()) == set(err_expect.keys())

    for key_i in err_expect:
        assert err_out[key_i] == approx(err_expect[key_i])


def test_ERR_GYRO_MOTION1():
    err_expect = {
        "N": 0.00009,
        "B": 0.00003,
        "tau_cb": 50.0,
    }

    err_out = constants.ERR_GYRO_MOTION1

    assert set(err_out.keys()) == set(err_expect.keys())

    for key_i in err_expect:
        assert err_out[key_i] == approx(err_expect[key_i])


def test_ERR_ACC_MOTION2():
    err_expect = {
        "N": 0.0007,
        "B": 0.0005,
        "tau_cb": 50.0,
    }

    err_out = constants.ERR_ACC_MOTION2

    assert set(err_out.keys()) == set(err_expect.keys())

    for key_i in err_expect:
        assert err_out[key_i] == approx(err_expect[key_i])


def test_ERR_GYRO_MOTION2():
    err_expect = {
        "N": 0.00005,
        "B": 0.00003,
        "tau_cb": 50.0,
    }

    err_out = constants.ERR_GYRO_MOTION2

    assert set(err_out.keys()) == set(err_expect.keys())

    for key_i in err_expect:
        assert err_out[key_i] == approx(err_expect[key_i])
