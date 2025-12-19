import pytest
import numpy as np

from smsfusion.simulate import ConstantDOF
from smsfusion.simulate._simulate import DOF


class Test_DOF:
    @pytest.fixture
    def some_dof(self):
        class SomeDOF(DOF):

            def _y(self, t):
                return np.ones_like(t)

            def _dydt(self, t):
                return 2 * np.ones_like(t)

            def _d2ydt2(self, t):
                return 3 *np.ones_like(t)
            
        return SomeDOF()

    def test_y(self, some_dof):
        t = np.linspace(0, 10, 100)
        y = some_dof.y(t)
        np.testing.assert_allclose(y, np.ones(100))

    def test_dydt(self, some_dof):
        t = np.linspace(0, 10, 100)
        dydt = some_dof.dydt(t)
        np.testing.assert_allclose(dydt, 2 * np.ones(100))

    def test_d2ydt2(self, some_dof):
        t = np.linspace(0, 10, 100)
        d2ydt2 = some_dof.d2ydt2(t)
        np.testing.assert_allclose(d2ydt2, 3 * np.ones(100))

    def test__call__(self, some_dof):
        t = np.linspace(0, 10, 100)
        y, dydt, dy2dt2 = some_dof(t)
        np.testing.assert_allclose(y, np.ones(100))
        np.testing.assert_allclose(dydt, 2 * np.ones(100))
        np.testing.assert_allclose(dy2dt2, 3 * np.ones(100))

class Test_ConstantDOF:
    @pytest.fixture
    def constant_dof(self):
        return ConstantDOF(value=5.0)

    def test_y(self, constant_dof):
        t = np.linspace(0, 10, 100)
        y = constant_dof.y(t)
        np.testing.assert_allclose(y, 5.0 * np.ones(100))

    def test_dydt(self, constant_dof):
        t = np.linspace(0, 10, 100)
        dydt = constant_dof.dydt(t)
        np.testing.assert_allclose(dydt, np.zeros(100))

    def test_d2ydt2(self, constant_dof):
        t = np.linspace(0, 10, 100)
        d2ydt2 = constant_dof.d2ydt2(t)
        np.testing.assert_allclose(d2ydt2, np.zeros(100))

    def test__call__(self, constant_dof):
        t = np.linspace(0, 10, 100)
        y, dydt, dy2dt2 = constant_dof(t)
        np.testing.assert_allclose(y, 5.0 * np.ones(100))
        np.testing.assert_allclose(dydt, np.zeros(100))
        np.testing.assert_allclose(dy2dt2, np.zeros(100))
