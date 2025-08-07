import numpy as np

# Default initial state vector
X0 = np.zeros(16, dtype=np.float64)
X0[6] = 1.0

# Default initial covariance matrix
P0 = np.eye(12) * 1e-6

# Noise and bias parameters for SMS Motion Gen 2
ERR_ACC_MOTION2 = {
    "N": 0.0007,  # (m/s^2)/sqrt(Hz)
    "B": 0.0005,  # m/s^2
    "tau_cb": 50.0,  # s
}

ERR_GYRO_MOTION2 = {
    "N": 0.00005,  # (rad/s)/sqrt(Hz)
    "B": 0.00005,  # rad/s
    "tau_cb": 50.0,  # s
}
