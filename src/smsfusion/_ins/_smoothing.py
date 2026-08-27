from ._amekf import AMEKF
from ._pvamekf import PVAMEKF
from ._vamekf import VAMEKF


class FixedIntervalSmoother:
    def __init__(self, mekf: AMEKF | VAMEKF | PVAMEKF):
        self._mekf = mekf
