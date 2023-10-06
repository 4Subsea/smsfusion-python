# Reference data for AINS / AHRS testing

Data can be generated from:
git clone --single-branch --branch ains-ahrs-reference-case https://github.com/4Subsea/datalab-workspace.git ains-ahrs-reference-case

Filename: ains_ahrs_imu.parquet
Data type: Pandas DataFrame
Columns: 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'Alpha', 'Beta', 'Gamma', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Ax_meas', 'Ay_meas', 'Az_meas', 'Gx_meas', 'Gy_meas', 'Gz_meas', 'X_meas', 'Y_meas', 'Z_meas', 'Gamma_meas'
Coordinate system: NED
Sampling frequency: 10.24 Hz

Data set is extracted from an OrcaFlex simulation of a vessel.
Original simulation is 60 second ramp-up + 3600 seconds, but data set is truncated to 1720 - 1960 seconds (~6 mins)

Representative noise is added to '*_meas' columns. Details given below:

- 'Ax_meas', 'Ay_meas', 'Az_meas', 'Gx_meas', 'Gy_meas', 'Gz_meas' -> Representative IMU noise.
- 'X_meas', 'Y_meas', 'Z_meas' -> Representative GNSS noise, white noise with 5 cm standard deviation.
- 'Gamma' -> Representative compass noise, white noise with 0.1 degrees standard deviation.
