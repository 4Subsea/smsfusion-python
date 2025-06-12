Quickstart
==========
This is a quick introduction to the `SMS Fusion` Python package. The package provides
Python implementations of INS algorithms as presented below.

Inertial navigation primer
--------------------------
Measurement data from an `inertial measurement unit` (IMU) forms the backbone of an
`inertial navigation system` (INS). These measurements are integrated to estimate the
position, velocity, and attitude (PVA) of the moving object to which the IMU is attached.
Since the IMU's measurements are subject to noise and bias, the PVA estimates will drift
over time if they are not corrected. Thus, `Aided INS` (AINS) systems incorporate additional
long-term stable aiding measurements to ensure convergence and stability of the INS.
The aiding measurements are typically provided by a `global navigation satellite system`
(GNSS) and a compass, providing absolute position, velocity, and heading information.

In scenarios where only compass aiding (but no GNSS) is available, the INS cannot provide
reliable position and velocity information but still deliver stable attitude estimates.
When the AINS is operated in this mode, we call it an `Attitude and Heading Reference System`
(AHRS).

In aiding-denied scenarios, where no aiding measurements are available, the INS
must rely solely on the IMU's measurements to estimate the body's motion. In such
scenarios, only the roll and pitch degrees of freedom are observable, as they can
still be corrected using the IMU's accelerometer data and the known direction of
the gravitational field. When operated in this mode, the AINS is referred to as
a `Vertical Reference Unit` (VRU).

``smsfusion`` provides Python implementations of a few INS algorithms, including:

* :class:`~smsfusion.AidedINS`: Aided INS (AINS) algorithm. Used to estimate position,
  velocity and attitude (PVA) using IMU data, GNSS data and compass data.
* :class:`~smsfusion.AHRS`: AHRS wrapper around :class:`~smsfusion.AidedINS` with sane defaults.
  Used to estimate attitude only using IMU data and compass data.
* :class:`~smsfusion.VRU`: VRU wrapper around :class:`~smsfusion.AidedINS` with sane defaults.
  Used to estimate roll and pitch only using IMU data.
* :class:`~smsfusion.StrapdownINS`: Simple strapdown INS algorithm, where the
  IMU measurements are integrated without incorporating any additional aiding measurements.
  The state estimates will therefore drift over time and quickly diverge from their true values.
  This class is primarily used for PVA propagation in other aided INS algorithms.

All AINS algorithms in ``smsfusion`` are based on a fusion filtering technique known
as the `multiplicative extended Kalman filter` (MEKF).

In this quickstart guide, we will demonstrate how to use the AINS algorithms
available in ``smsfusion`` to estimate PVA of a moving body using IMU measurements
and aiding measurements.

Measurement data
----------------
This quickstart guide assumes that you have access to accelerometer and gyroscope
data from an IMU sensor, and maybe position and heading data from other aiding
sensors. If you do not have access to such data, you can generate synthetic
measurements using the code provided here.

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


    fs = 10.24  # sampling rate in Hz
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


For simpler cases where only compass or no aiding is available, consider using
:func:`~smsfusion.benchmark.benchmark_pure_attitude_beat_202311A` instead to
generate synthetic data.

Aided INS: Estimate position, velocity and attitude (PVA)
---------------------------------------------------------
If you have access to accelerometer and gyroscope data from an IMU sensor, as well
as position and heading data from other aiding sensors, you can estimate the position,
velocity and attitude (PVA) of a moving body using the :func:`~smsfusion.AidedINS` class:

.. code-block:: python

    import numpy as np
    import smsfusion as sf


    # Initial (a priori) state
    p0 = pos[0]  # position [m]
    v0 = vel[0]  # velocity [m/s]
    q0 = sf.quaternion_from_euler(euler[0], degrees=False)  # attitude as unit quaternion
    ba0 = np.zeros(3)  # accelerometer bias [m/s^2]
    bg0 = np.zeros(3)  # gyroscope bias [rad/s]
    x0 = np.concatenate((p0, v0, q0, ba0, bg0))

    # Initialize AINS
    ains = sf.AidedINS(fs, x0)

    # Estimate PVA states sequentially using AINS
    pos_est, vel_est, euler_est = [], [], []
    for f_i, w_i, p_i, h_i in zip(acc_imu, gyro_imu, pos_aid, head_aid):
        ains.update(
            f_i,
            w_i,
            degrees=False,
            pos=p_i,
            pos_var=pos_noise_std**2 * np.ones(3),
            head=h_i,
            head_var=head_noise_std**2,
            head_degrees=False,
        )
        pos_est.append(ains.position())
        vel_est.append(ains.velocity())
        euler_est.append(ains.euler(degrees=False))

    pos_est = np.array(pos_est)
    vel_est = np.array(vel_est)
    euler_est = np.array(euler_est)

AHRS: Estimate attitude with compass-aiding
-------------------------------------------
To limit integration drift in AHRS mode, we must assume that the sensor on average
is stationary. The static assumtion is incorporated as so-called pseudo aiding measurements
of zero with corresponding error variances. For most applications, the following pseudo
aiding is sufficient:

* Position: 0 m with 1000 m standard deviation
* Velocity: 0 m/s with 10 m/s standard deviation

If you have access to accelerometer and gyroscope data from an IMU sensor and
compass measurements, you can estimate the attitude of a moving body using
the :func:`~smsfusion.AHRS` class:

.. code-block:: python

    import numpy as np
    import smsfusion as sf


    # Initial (a priori) state
    p0 = np.zeros(3)  # position [m]
    v0 = np.zeros(3)  # velocity [m/s]
    q0 = sf.quaternion_from_euler(euler[0], degrees=False)  # attitude as unit quaternion
    ba0 = np.zeros(3)  # accelerometer bias [m/s^2]
    bg0 = np.zeros(3)  # gyroscope bias [rad/s]
    x0 = np.concatenate((p0, v0, q0, ba0, bg0))

    # Initialize AHRS
    ahrs = sf.AHRS(fs, x0)

    # Estimate attitude sequentially using AHRS
    euler_est = []
    for f_i, w_i, h_i in zip(acc_imu, gyro_imu, head_aid):
        ahrs.update(
            f_i,
            w_i,
            degrees=False,
            head=h_i,
            head_var=head_noise_std**2,
            head_degrees=False,
        )
        euler_est.append(ahrs.euler(degrees=False))

    euler_est = np.array(euler_est)

VRU: Estimate partial attitude in aiding-denied scenarios
---------------------------------------------------------
To limit integration drift in VRU mode, we must assume that the sensor on average
is stationary. The static assumption is incorporated as so-called pseudo aiding measurements
of zero with corresponding error variances. For most applications, the following pseudo
aiding is sufficient:

* Position: 0 m with 1000 m standard deviation
* Velocity: 0 m/s with 10 m/s standard deviation

Note that the heading is not corrected in VRU mode, and the yaw degree of freedom
will thus drift arbitrarily.

If you have access to accelerometer and gyroscope data from an IMU sensor, you can
estimate the roll and pitch degrees of freedom of a moving body using the
:func:`~smsfusion.VRU` class:

.. code-block:: python

    import numpy as np
    import smsfusion as sf


    # Initial (a priori) state
    p0 = np.zeros(3)  # position [m]
    v0 = np.zeros(3)  # velocity [m/s]
    q0 = sf.quaternion_from_euler(euler[0], degrees=False)  # attitude as unit quaternion
    ba0 = np.zeros(3)  # accelerometer bias [m/s^2]
    bg0 = np.zeros(3)  # gyroscope bias [rad/s]
    x0 = np.concatenate((p0, v0, q0, ba0, bg0))

    # Initialize VRU
    vru = sf.VRU(fs, x0)

    # Estimate roll and pitch sequentially using VRU
    roll_pitch_est = []
    for f_i, w_i in zip(acc_imu, gyro_imu):
        vru.update(
            f_i,
            w_i,
            degrees=False
        )
        roll_pitch_est.append(vru.euler(degrees=False)[:2])

    roll_pitch_est = np.array(roll_pitch_est)


Smoothing
---------
Smoothing refers to post-processing techniques that enhance the accuracy of a Kalman
filter's state and covariance estimates by incorporating both past and future measurements.
In contrast, standard forward filtering (as implemented in :class:`~smsfusion.AidedINS`)
relies only on past and current measurements, leading to suboptimal estimates when
future data is available.

Fixed-interval smoothing
........................
The :class:`~smsfusion.FixedIntervalSmoother` class implements fixed-interval smoothing
for an :class:`~smsfusion.AidedINS` instance or one of its subclasses (:class:`~smsfusion.AHRS`
or :class:`~smsfusion.VRU`). After a complete forward pass using the AINS algorithm,
a backward sweep with a smoothing algorithm is performed to refine the state
and covariance estimates. Fixed-interval smoothing is particularly useful
when the entire measurement sequence is available, as it allows for optimal state
estimation by considering all measurements in the sequence.

The following example demonstrates how to refine a :class:`~smsfusion.VRU`'s roll
and pitch estimates using :class:`~smsfusion.FixedIntervalSmoother`. The same
workflow applies if the underlying AINS instance is an :class:`~smsfusion.AidedINS`
or an :class:`~smsfusion.AHRS` instead. However, note that the ``update()`` method may take
additional aiding parameters depending on the type of AINS instance used.

.. code-block:: python

    import smsfusion as sf


    vru_smoother = sf.FixedIntervalSmoother(vru)

    for f_i, w_i in zip(acc_imu, gyro_imu):
        vru_smoother.update(
            f_i,
            w_i,
            degrees=False
        )

    roll_pitch_est = vru_smoother.euler(degrees=False)[:, :2]
