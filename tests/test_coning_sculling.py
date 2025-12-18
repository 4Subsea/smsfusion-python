import numpy as np
import smsfusion as sf


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
