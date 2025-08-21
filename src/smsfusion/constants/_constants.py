import numpy as np
from numpy.typing import NDArray

# Default initial state vector
X0: NDArray[np.float64] = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)

# Default initial covariance matrix
P0: NDArray[np.float64] = np.eye(12) * 1e-6

# Noise and bias parameters for SMS Motion Gen 2
ERR_ACC_MOTION2: dict[str, float] = {
    "N": 0.0007,  # (m/s^2)/sqrt(Hz)
    "B": 0.0005,  # m/s^2
    "tau_cb": 50.0,  # s
}

ERR_GYRO_MOTION2: dict[str, float] = {
    "N": 0.00005,  # (rad/s)/sqrt(Hz)
    "B": 0.00005,  # rad/s
    "tau_cb": 50.0,  # s
}
