Quickstart
==========

Inertial navigation primer
--------------------------
Measurement data from an `inertial measurement unit` (IMU) forms the backbone of an
`inertial navigation system` (INS). These measurements are integrated to estimate the
position, velocity, and attitude (PVA) of the moving object to which the IMU is attached.
Since the IMU's measurements are subject to noise and bias, the PVA estimates will drift
over time if they are not corrected. Thus, Aided INS (AINS) systems incorporate additional
long-term stable aiding measurements to ensure convergence and stability of the INS.
The aiding measuruments are typically provided by a `global navigation satellite system`
(GNSS) or a compass, providing absolute position, velocity, and heading information.

In scenarios where aiding measurements are not available, the INS must rely solely
on the IMU's measurements to estimate the body's motions. In such scenarios, only the roll
and pitch degrees of freedom are observable, as they can still be corrected using
the IMU's accelerometer measurements and the known direction of the gravitational field.
When the AINS is operated in this mode, we call it a `Vertical Reference Unit` (VRU).

``smsfusion`` provides Python implementations of a few INS algorithms, including:

* :class:`~smsfusion.benchmark.StrapdownINS`: Simple strapdown INS algorithm, where the
  IMU measurements are integrated without incorporating any additional aiding measurements.
  The PVA estimates will therefore drift over time and quickly diverge from their true values.
  This class is primarily used for PVA propagation in other aided INS algorithms.
* :class:`~smsfusion.benchmark.AidedINS`: Aided INS algorithm based on the `multiplicative extended Kalman filter` (MEKF).

In this quickstart guide, we will demonstrate how to use the AINS algorithm privided
by ``smsfusion`` to estimate PVA of a moving body using IMU measurements and aiding
measurements.



Measurement data
----------------
This quickstart guide assumes that you have access to accelerometer and gyroscope
data from an IMU sensor, as well as position and heading data from other aiding sensors.
If you do not have access to such data, you can generate synthetic measurements using
the code provided here.

Using the ``benchmark`` module, you can generate synthetic 3D motion data with ``smsfusion``.
For example, you can generate beating signals representing position, velocity and
attitude (PVA) degrees of freedom using :func:`~smsfusion.benchmark.benchmark_full_pva_beat_202311A`:

.. code-block:: python

    from smsfusion.benchmark import benchmark_full_pva_beat_202311A


    fs = 10.24  # sampling rate in Hz
    t, pos, vel, euler, acc, gyro = benchmark_full_pva_beat_202311A(fs)
    head = euler[:, 2]

Note that the generated position signals are in meters (m), velocity signals are in meters
per second (m/s), and attitude signals are in radians (rad). The accelerometer signals
are in meters per second squared (m/s^2), and the gyroscope signals are in radians
per second (rad/s). If your measurement data is given in other units, you must account
for that in other sections of this quickstart guide.

To emulate real sensor recordings, these reference signals must be polluted with noise.
The ``noise`` module that comes with ``smsfusion`` provides a variety of noise models
that can be used to corrupt the reference signals. For example, the :func:`~smsfusion.noise.IMUNoise`
class can be used to add IMU-like noise to accelerometer and gyroscope signals:

.. code-block:: python

    import smsfusion as sf


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


    pos_noise_std = 0.1  # m
    head_noise_std = 0.01  # rad
    rng = np.random.default_rng()
    pos_aid = pos + pos_noise_std * rng.standard_normal(pos.shape)
    head_aid = head + head_noise_std * rng.standard_normal(head.shape)

Estimate position, velocity and attitude (PVA)
----------------------------------------------
If you have access to accelerometer and gyroscope data from an IMU sensor, as well
as position and heading data from other aiding sensors, you can estimate the position,
velocity and attitude (PVA) of a moving body using the :func:`~smsfusion.AidedINS` class
provided by ``smsfusion``:

.. code-block:: python

    import numpy as np
    import smsfusion as sf
    from smsfusion._transforms import _quaternion_from_euler


    # Initial (a priori) state
    p0 = pos[0]  # position [m]
    v0 = vel[0]  # velocity [m/s]
    q0 = _quaternion_from_euler(euler[0])  # attitude as unit quaternion
    ba0 = np.zeros(3)  # accelerometer bias [m/s^2]
    bg0 = np.zeros(3)  # gyroscope bias [rad/s]
    x0 = np.concatenate((p0, v0, q0, ba0, bg0))

    # Initial (a priori) error covariance matrix
    P0 = np.eye(12) * 1e-3

    # IMU noise characteristics
    err_acc = sf.constants.ERR_ACC_MOTION2  # m/s^2
    err_gyro = sf.constants.ERR_GYRO_MOTION2  # rad/s

    # Initialize AINS
    ains = sf.AidedINS(fs, x0, P0, err_acc, err_gyro)

    # Estimate PVA states sequentially using AINS
    pos_est, vel_est, euler_est = [], [], []
    for acc_i, gyro_i, pos_i, head_i in zip(acc_imu, gyro_imu, pos_aid, head_aid):
        ains.update(
            acc_i,
            gyro_i,
            degrees=False,
            pos=pos_i,
            pos_var=pos_noise_std**2 * np.ones(3),
            head=head_i,
            head_var=head_noise_std**2,
            head_degrees=False,
        )
        pos_est.append(ains.position())
        vel_est.append(ains.velocity())
        euler_est.append(ains.euler(degrees=False))

    pos_est = np.array(pos_est)
    vel_est = np.array(vel_est)
    euler_est = np.array(euler_est)

Estimate attitude in aiding-denied scenarios
--------------------------------------------
In aiding-denied scenarios, where you don't have access to long-term stable aiding
sensors like GNSS or compass, you must rely soley on the IMU's measurements to estimate
the body's motions. Only the roll and pitch degrees of freedom are observable in these
scenarios, as they can still be corrected using accelerometer measurements and the
known direction of the gravitational field. When the AINS is operated in this mode,
we call it a Vertical Reference Unit (VRU).

To limit integration drift in VRU mode, we must assume that the sensor on average
is stationary. The static assumtion is incorporated as so-called psedo aiding measurements
of zero with corresponding variances. For most applications, the following pseudo
aiding is sufficient:

* Position: 0 m with 1000 m standard deviation
* Velocity: 0 m/s with 10 m/s standard deviation

If you have access to accelerometer and gyroscope data from an IMU sensor, you can
estimate the roll and pitch degrees of freedom of a moving body using the :func:`~smsfusion.AidedINS`
class provided by ``smsfusion`` operated in VRU mode:

.. code-block:: python

    import numpy as np
    import smsfusion as sf
    from smsfusion._transforms import _quaternion_from_euler


    # Initial (a priori) state
    p0 = pos[0]  # position [m]
    v0 = vel[0]  # velocity [m/s]
    q0 = _quaternion_from_euler(euler[0])  # attitude as unit quaternion
    ba0 = np.zeros(3)  # accelerometer bias [m/s^2]
    bg0 = np.zeros(3)  # gyroscope bias [rad/s]
    x0 = np.concatenate((p0, v0, q0, ba0, bg0))

    # Initial (a priori) error covariance matrix
    P0 = np.eye(12) * 1e-3

    # IMU noise characteristics
    err_acc = sf.constants.ERR_ACC_MOTION2  # m/s^2
    err_gyro = sf.constants.ERR_GYRO_MOTION2  # rad/s

    # Initialize AINS
    ains = sf.AidedINS(fs, x0, P0, err_acc, err_gyro)

    # Estimate PVA states sequentially using AINS
    euler_est = []
    for acc_i, gyro_i in zip(acc_imu, gyro_imu):
        ains.update(
            acc_i,
            gyro_i,
            degrees=False,
            pos=np.zeros(3),
            pos_var=1000.0**2 * np.ones(3),
            vel=np.zeros(3),
            vel_var=10.0**2 * np.ones(3),
        )
        euler_est.append(ains.euler(degrees=False))

    euler_est = np.array(euler_est)
