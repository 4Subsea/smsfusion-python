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

``smsfusion`` provides Python implementations of a few AINS algorithms, including
:class:`~smsfusion.AidedINS`, :class:`~smsfusion.AHRS` and :class:`~smsfusion.VRU`.
In this quickstart guide we will demonstrate how to use these AINS algorithms to
estimate PVA of a moving body using IMU measurements and aiding measurements.

Measurement data
----------------
This quickstart guide assumes that you have access to accelerometer and gyroscope
data from an IMU sensor, and ideally position and heading data from other aiding
sensors. If you do not have access to such data, you can generate synthetic
measurements using the code provided here.

Using the :class:`~smsfusion.simulate.IMUSimulator` class, you can generate synthetic
3D motion data and corresponding IMU accelerometer and gyroscope measurements..
For example, you can simulate beating motion:


.. code-block:: python

    from smsfusion.simulate import BeatDOF, IMUSimulator


    fs = 10.24  # sampling rate in Hz
    n = 10_000  # number of samples
    sim = IMUSimulator(
        pos_x=BeatDOF(0.5, phase=0.0, phase_degrees=True),
        pos_y=BeatDOF(0.5, phase=45.0, phase_degrees=True),
        pos_z=BeatDOF(0.5, phase=90.0, phase_degrees=True),
        alpha=BeatDOF(0.1, phase=135.0, phase_degrees=True),
        beta=BeatDOF(0.1, phase=180.0, phase_degrees=True),
        gamma=BeatDOF(0.1, phase=225.0, phase_degrees=True),
        degrees=False,
    )

    t, pos, vel, euler, acc, gyro = sim(fs, 10_000, degrees=False)
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

INS algorithms
--------------
The following INS algorithms are provided by ``smsfusion``:

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

All AINS algorithms provided by ``smsfusion`` are based on a fusion filtering technique
known as the `multiplicative extended Kalman filter` (MEKF).

AidedINS - IMU + heading and position aiding
............................................
If you have access to accelerometer and gyroscope data from an IMU sensor, as well
as position and heading data from other aiding sensors, you can estimate the position,
velocity and attitude (PVA) of a moving body using the :func:`~smsfusion.AidedINS` class:

.. code-block:: python

    import numpy as np
    import smsfusion as sf


    # Initialize AINS
    fs = 10.24  # sampling rate in Hz
    ains = sf.AidedINS(fs)

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

AHRS - IMU + heading aiding
...........................
In scenarios where only compass aiding is available (i.e., no GNSS), the INS is
unable to provide reliable position and velocity information, but it can still
deliver stable attitude estimates. When the AINS is operated in this mode, we call
it an `Attitude and Heading Reference System` (AHRS).

To limit integration drift in AHRS mode, we must assume that the sensor on average
is stationary. The static assumtion is incorporated as so-called pseudo aiding measurements
of zero with corresponding error variances. For most applications, the following pseudo
aiding is sufficient:

* Position: 0 m with 1000 m standard deviation
* Velocity: 0 m/s with 10 m/s standard deviation

If you have access to accelerometer and gyroscope data from an IMU sensor and
heading measurements from a compass, you can estimate the attitude of a moving body
using the :func:`~smsfusion.AHRS` class:

.. code-block:: python

    import numpy as np
    import smsfusion as sf


    # Initialize AHRS
    fs = 10.24  # sampling rate in Hz
    ahrs = sf.AHRS(fs)

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

VRU - IMU only (aiding-denied)
..............................
In aiding-denied scenarios, where no aiding measurements are available, the INS
must rely solely on the IMU's measurements to estimate the body's motion. In such
scenarios only the roll and pitch degrees of freedom are observable, as they can
still be corrected using the IMU's accelerometer data and the known direction of
the gravitational field. When operated in this mode, the AINS is referred to as
a `Vertical Reference Unit` (VRU).

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


    # Initialize VRU
    fs = 10.24  # sampling rate in Hz
    vru = sf.VRU(fs)

    # Estimate roll and pitch sequentially using VRU
    roll_pitch_est = []
    for f_i, w_i in zip(acc_imu, gyro_imu):
        vru.update(f_i, w_i, degrees=False)
        roll_pitch_est.append(vru.euler(degrees=False)[:2])

    roll_pitch_est = np.array(roll_pitch_est)


Smoothing
---------
Smoothing refers to post-processing techniques that enhance the accuracy of a Kalman
filter's state and covariance estimates by incorporating both past and future measurements.
In contrast, standard forward filtering (as provided by :class:`~smsfusion.AidedINS`,
:class:`~smsfusion.AHRS` and :class:`~smsfusion.VRU`) relies only on past and current
measurements, leading to suboptimal estimates when future data is available.

Fixed-interval smoothing
........................
The :class:`~smsfusion.FixedIntervalSmoother` class implements fixed-interval smoothing
for an :class:`~smsfusion.AidedINS` instance or one of its subclasses (:class:`~smsfusion.AHRS`
or :class:`~smsfusion.VRU`). After a complete forward pass using the given AINS
algorithm, a backward sweep with a smoothing algorithm is performed to refine the
state and covariance estimates. Fixed-interval smoothing is particularly useful
when the entire measurement sequence is available, as it allows for optimal state
estimation by considering all measurements in the sequence.

The following example demonstrates how to refine a :class:`~smsfusion.VRU`'s roll
and pitch estimates using :class:`~smsfusion.FixedIntervalSmoother`. The same
workflow applies if the underlying AINS instance is an :class:`~smsfusion.AidedINS`
or an :class:`~smsfusion.AHRS` instead. However, note that the ``update()`` method may take
additional aiding parameters depending on the type of AINS instance used.

.. code-block:: python

    import smsfusion as sf


    # Initialize VRU-based fixed-interval smoother
    fs = 10.24  # sampling rate in Hz
    smoother = sf.FixedIntervalSmoother(sf.VRU(fs))

    # Update with accelerometer and gyroscope measurements
    for f_i, w_i in zip(acc_imu, gyro_imu):
        smoother.update(f_i, w_i, degrees=False)

    # Get smoothed roll and pitch estimates
    roll_pitch_est = smoother.euler(degrees=False)[:, :2]
