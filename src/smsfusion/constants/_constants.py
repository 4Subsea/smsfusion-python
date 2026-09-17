# Noise parameters for SMS Motion Gen 2
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
