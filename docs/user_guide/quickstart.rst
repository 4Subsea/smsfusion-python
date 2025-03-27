Quickstart
==========

Inertial navigation primer
--------------------------
Measurement data from an inertial measurement unit (IMU) is the backbone of an inertial
navigation system (INS). The measurements from this IMU sensor are integrated to estimate
the position, velocity and attitude (PVA) of the moving object to which the IMU is attached.
To avoid integration drift, these integrated IMU measurements must be aided by other
sensors; typically, a GNSS is used to provide absolute position and/or velocity
measurements, and a compass is used to provide absolute heading measurements. Otherwise,
if such aiding sensors are not available, the INS must rely soley on the IMU's measurements
to provide estimates of the body's motions. In such aiding denied scenarios, only
the roll and pitch degrees of freedom are observable, and an assumtion of stationarity
must be incorporated to ensure convergence of these states.

Measurement data
----------------
This quickstart guide assumes that you have access to accelerometer and gyroscope
data from an IMU, as well as position and heading data from aiding sensors. If
you do not have access to such data, you can generate synthetic measurements using
the code given below.

Using the ``benchmark`` module, you can generate synthetic 3D motion data with ``smsfusion``.
For example, you can generate beating signals representing position, velocity and
attitude (PVA) degrees of freedom using :func:`~smsfusion.benchmark.benchmark_full_pva_beat_202311A`:

.. code-block:: python

    from smsfusion.benchmark import benchmark_full_pva_beat_202311A


    fs = 10.24  # sampling rate in Hz
    t, pos, vel, euler, acc, gyro = benchmark_full_pva_beat_202311A(fs)
    head = euler[:, 2]

To emulate real sensor recordings, these reference signals must be polluted with noise.
The ``noise`` module that comes with ``smsfusion`` provides a variety of noise models
that can be used to corrupt the reference signals. For example, the :func:`~smsfusion.noise.IMUNoise`
class can be used to add IMU-like noise to accelerometer and gyroscope signals:

.. code-block:: python

    from smsfusion.noise import IMUNoise


    fs = 10.24
    err_acc = sf.constants.ERR_ACC_MOTION2  # m/s^2
    err_gyro = sf.constants.ERR_GYRO_MOTION2  # rad/s
    imu_noise = sf.noise.IMUNoise(err_acc, err_gyro)(fs, len(acc))
    acc_imu = acc + imu_noise[:, :3]
    gyro_imu = gyro + imu_noise[:, 3:]

Similarly, white noise can be added to the position and heading measurements using
``NumPy``'s random number generator:

.. code-block:: python

    import numpy as np


    gnss_noise_std = 0.1  # m
    compass_noise_std = 0.01  # rad
    rng = np.random.default_rng()
    pos_aid = pos + gnss_noise_std * rng.standard_normal(pos.shape)
    head_aid = head + compass_noise_std * rng.standard_normal(head.shape)
