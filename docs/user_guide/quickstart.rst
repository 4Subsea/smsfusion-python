Quickstart
==========
This is a short introduction to `SMS Fusion`: a Python library with inertial navigation
algorithms for `SMS Motion`. Although the primary purpose is to complement the SMS Motion
hardware, the algorithms provided by ``smsfusion`` can be used with any IMU sensor.

Inertial navigation primer
--------------------------
Measurement data from an `inertial measurement units` (IMU) form the backbone of an
`inertial navigation system` (INS). These measurements are integrated to estimate the
position, velocity, and/or attitude (PVA) of the moving object to which the IMU is attached.
Since the IMU's measurements are subject to noise and bias, the PVA estimates will drift
over time if they are not corrected. Thus, `aided INS` (AINS) systems incorporate additional
long-term stable aiding measurements to ensure convergence and stability of the INS.
The aiding measurements are typically provided by a `global navigation satellite system`
(GNSS) and a compass, providing absolute position, velocity, and heading information.

The algorithms used to combine IMU measurements with aiding measurements to estimate
the INS' states are commonly known as `fusion filters`. One of the most widely used
fusion filters for PVA estimation is the `multiplicative extended Kalman filter` (MEKF).


Multiplicative extended Kalman filter (MEKF)
--------------------------------------------

The core algorithms in ``smsfusion`` are a family of MEKF filters:

- :class:`~smsfusion.AMEKF`: estimates attitude and gyroscope bias.
- :class:`~smsfusion.VAMEKF`: estimates velocity, attitude and gyroscope bias.
- :class:`~smsfusion.PVAMEKF`: estimates position, velocity, attitude and gyroscope bias.

The three flavors of the MEKF differ only in the number of states they estimate, and hence also
the type of external aiding they support. :class:`~smsfusion.PVAMEKF` is the most
comprehensive filter, estimating all degrees of freedom (12 states). By leveraging external
aiding measurements, this filter can achieve the highest accuracy in all state estimates.
Thus, if you have access to full external aiding (GNSS and compass), this is the
filter to use. The :class:`~smsfusion.VAMEKF` (9 states) and :class:`~smsfusion.AMEKF` (6 states) are
degenerated versions of the this filter, where some of the states are removed,
resulting in simpler filters with reduced computational complexity at the expense
of potentially lower accuracy. In aiding denied scenarios, where external aiding
is not feasible or simply not available, the simpler filters are more suitable.

The table below lists which MEKF filter to use for different aiding scenarios:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - External aiding
     - Filter options
     - State estimates
   * - No external aiding
     - | :class:`~smsfusion.AMEKF` (w/ gravity reference),
       | :class:`~smsfusion.VAMEKF` (w/ zero-velocity),
     - | Roll,
       | Pitch
   * - Compass (heading)
     - | :class:`~smsfusion.AMEKF` (w/ gravity reference),
       | :class:`~smsfusion.VAMEKF` (w/ zero-velocity),
     - | Roll,
       | Pitch,
       | Yaw
   * - | GNSS (position),
       | Compass (heading)
     - :class:`~smsfusion.PVAMEKF`
     - | Position,
       | Velocity,
       | Roll,
       | Pitch,
       | Yaw

In the following sections, we will demonstrate how to apply the MEKF filters in
these different aiding scenarios.


Measurement data
................
The examples given in this quickstart assume that you have access to measurement
data from an IMU sensor and, depending on the scenario, other external aiding sensors.
If you don't have access to such data, you can generate synthetic measurements using
the :mod:`~smsfusion.benchmark` module in ``smsfusion``:

.. code-block:: python

    import numpy as np
    import smsfusion as sf
    from smsfusion.benchmark import benchmark_full_pva_beat_202311A


    # IMU and PVA reference signals
    fs = 10.24  # sampling rate in Hz
    t, pos, vel, euler, f, w = benchmark_full_pva_beat_202311A(fs)
    head = euler[:, 2]

    # Add measurement noise
    rng = np.random.default_rng(0)
    imu_noise = sf.noise.IMUNoise(seed=1)(fs, len(f))
    f_meas = f + imu_noise[:, :3]  # m/s^2
    w_meas = w + imu_noise[:, 3:]  # rad/s
    pos_meas = pos + 0.1 * rng.standard_normal(pos.shape)  # m
    head_meas = head + 0.01 * rng.standard_normal(head.shape)  # rad

Note that the generated position signals are in meters (m), the velocity signals
are in meters per second (m/s), the attitude signals are in radians (rad), the
accelerometer signals are in meters per second squared (m/s^2), and the gyroscope
signals are in radians per second (rad/s). If your measurement data is given in
other units, you must account for that when using the examples provided.

IMU only (no external aiding) - estimate roll and pitch
.......................................................
In aiding denied scenarios, where you don't have access to long-term stable aiding
measurements, only the roll and pitch degrees of freedom are observable. These states
can still be corrected using the IMU's accelerometer measurements and the known
direction of gravity. In this scenario, two filter options are commonly used:

* :class:`~smsfusion.AMEKF` w/ gravity reference aiding
* :class:`~smsfusion.VAMEKF` w/ zero-velocity update (ZUPT)

:class:`~smsfusion.AMEKF` with gravity reference aiding is the most lightweight
and robust choice for estimating roll and pitch in the absence of external aiding.
The following example demonstrates how to apply the filter:

.. code-block:: python

    import smsfusion as sf


    # Initialize MEKF
    q0 = sf.quaternion_from_euler(euler[0], degrees=False)
    mekf = sf.AMEKF(fs, q0=q0)

    # Gravity reference aiding
    gref = {"gref": True, "gref_var": (0.0001, 0.0001, 0.0001)}

    # Update with IMU measurements
    roll_pitch_est = []
    for f_i, w_i in zip(f_meas, w_meas):
        mekf.update(f_i / fs, w_i / fs, **gref)
        roll_pitch_est.append(mekf.euler()[:2])

    # State estimates
    roll_pitch_est = np.array(roll_pitch_est)

The downside of using accelerometer measurements and the direction of gravity as
aiding, is that it is sensitive to errors from sustained linear accelerations; this
is because we must assume that the body is stationary such that the accelerometer
measures only the gravitational acceleration.

An alternative filter option for this scenario is to use the :class:`~smsfusion.VAMEKF`
with zero-velocity update (ZUPT); i.e., we assume that the body is stationary with
zero velocity. This approach has shown better accuracy compared to the gravity
reference aiding, although it still degrades under sustained linear accelerations.
The following example demonstrates how to apply the filter:

.. code-block:: python

    import smsfusion as sf


    # Initialize MEKF
    lat = 60.0  # latitude
    q0 = sf.quaternion_from_euler(euler[0], degrees=False)
    mekf = sf.VAMEKF(fs, q0=q0, g=sf.gravity(lat))

    # Zero-velocity update (ZUPT) with 10 m/s standard deviation
    zupt = {"vel": (0.0, 0.0, 0.0), "vel_var": (100.0, 100.0, 100.0)}

    # Update with IMU and aiding measurements
    roll_pitch_est = []
    for f_i, w_i in zip(f_meas, w_meas):
        mekf.update(f_i / fs, w_i / fs, **zupt)
        roll_pitch_est.append(mekf.euler()[:2])

    # State estimates
    roll_pitch_est = np.array(roll_pitch_est)

Using the :class:`~smsfusion.VAMEKF` with zero-velocity update requires a well
calibrated accelerometer and a correctly set local gravitational acceleration to
avoid instability. The :class:`~smsfusion.AMEKF` with gravity reference aiding
is thus considered a more robust option.


IMU + compass - estimate roll, pitch and yaw
............................................
If compass (i.e., heading) aiding measurements are available, also the yaw degree
of freedom can be estimated. Roll and pitch should still be corrected either with
gravity reference aiding or zero-velocity aiding, as described above.

The following example demonstrates how to apply the :class:`~smsfusion.AMEKF` with
gravity reference and heading aiding:

.. code-block:: python

    import smsfusion as sf


    # Initialize MEKF
    q0 = sf.quaternion_from_euler(euler[0], degrees=False)
    mekf = sf.AMEKF(fs, q0=q0)

    # Gravity reference aiding
    gref = {"gref": True, "gref_var": (0.0001, 0.0001, 0.0001)}

    # Update with IMU and aiding measurements
    euler_est = []
    for f_i, w_i, h_i in zip(f_meas, w_meas, head_meas):
        mekf.update(f_i / fs, w_i / fs, head=h_i, head_var=0.01**2, **gref)
        euler_est.append(mekf.euler())

    # State estimates
    euler_est = np.array(euler_est)

The following example demonstrates how to apply the :class:`~smsfusion.VAMEKF`
with zero-velocity update (ZUPT) and heading aiding:

.. code-block:: python

    import smsfusion as sf


    # Initialize MEKF
    lat = 60.0  # latitude
    q0 = sf.quaternion_from_euler(euler[0], degrees=False)
    mekf = sf.VAMEKF(fs, q0=q0, g=sf.gravity(lat))

    # Zero-velocity update (ZUPT) with 10 m/s standard deviation
    zupt = {"vel": (0.0, 0.0, 0.0), "vel_var": (100.0, 100.0, 100.0)}

    # Update with IMU and aiding measurements
    euler_est = []
    for f_i, w_i, h_i in zip(f_meas, w_meas, head_meas):
        mekf.update(f_i / fs, w_i / fs, head=h_i, head_var=0.01**2, **zupt)
        euler_est.append(mekf.euler())

    # State estimates
    euler_est = np.array(euler_est)


IMU + GNSS and compass - estimate position, velocity and attitude
.................................................................
With GNSS and compass aiding, we can estimate all degrees of freedom. The attitude
estimates will also become more accurate since we avoid errors caused by linear
acceleration under assumed stationary conditions.

The following example demonstrates how to apply the :class:`~smsfusion.PVAMEKF`
with position and heading aiding:

.. code-block:: python

    import smsfusion as sf


    # Initialize MEKF
    lat = 60.0  # latitude
    p0 = pos_meas[0]
    v0 = vel_meas[0]
    q0 = sf.quaternion_from_euler(euler[0], degrees=False)
    mekf = sf.PVAMEKF(fs, q0=q0, g=sf.gravity(lat))

    # Update with IMU and aiding measurements
    pos_est, vel_est, euler_est = [], [], []
    for f_i, w_i, h_i, p_i in zip(df_meas, w_meas, head_meas, pos_meas):
        mekf.update(
            f_i / fs,
            w_i / fs,
            head=h_i,
            head_var=0.01**2,
            pos=p_i,
            pos_var=(0.1**2, 0.1**2, 0.1**2),
        )
        pos_est.append(mekf.position())
        vel_est.append(mekf.velocity())
        euler_est.append(mekf.euler())

    # State estimates
    pos_est = np.array(pos_est)
    vel_est = np.array(vel_est)
    euler_est = np.array(euler_est)


Smoothing
---------
Smoothing refers to post-processing techniques that enhance the accuracy of a Kalman
filter's state and covariance estimates by incorporating both past and future measurements.
In contrast, standard forward filtering (as provided by the MEKF) relies only on past and current
measurements, leading to suboptimal estimates when future data is available.

Fixed-interval smoothing
........................

The :class:`~smsfusion.FixedIntervalSmoother` class implements fixed-interval smoothing
for a :class:`~smsfusion.PVAMEKF` instance. After a complete forward pass with the MEKF,
a backward sweep with a smoothing algorithm is performed to refine the state and
covariance estimates. Fixed-interval smoothing is particularly useful when the entire
measurement sequence is available, as it allows for optimal state estimation by
considering all measurements in the sequence.

The following example demonstrates how to refine a :class:`~smsfusion.PVAMEKF`
filter's roll and pitch estimates using :class:`~smsfusion.FixedIntervalSmoother`:

.. code-block:: python

    import smsfusion as sf


    # Initialize smoother
    lat = 60.0  # latitude
    p0 = pos_meas[0]
    v0 = vel_meas[0]
    q0 = sf.quaternion_from_euler(euler[0], degrees=False)
    mekf = sf.PVAMEKF(fs, p0=p0, v0=v0, q0=q0, g=sf.gravity(lat))
    smoother = sf.FixedIntervalSmoother(mekf)

    # Update with IMU and aiding measurements
    for f_i, w_i, h_i, p_i in zip(df_meas, w_meas, head_meas, pos_meas):
        smoother.update(
            f_i / fs,
            w_i / fs,
            head=h_i,
            head_var=0.01**2,
            pos=p_i,
            pos_var=(0.1**2, 0.1**2, 0.1**2),
        )

    # Smoothed state estimates
    pos_est = smoother.position()
    vel_est = smoother.velocity()
    euler_est = smoother.euler()


Kladd
-----

.. list-table::
   :header-rows: 1
   :widths: 10 15 35 40

   * - Filter
     - States
     - Aiding
     - When to use
   * - :class:`~smsfusion.PVAMEKF`
     - | Position,
       | Velocity,
       | Attitude,
       | Gyro bias
     - | GNSS
       | Compass
     - Estimate all DOFs when full aiding (GNSS and compass) is available.
   * - :class:`~smsfusion.VAMEKF`
     - | Roll,
       | Pitch,
       | Yaw*,
       | Gyro bias
     - | Zero-velocity,
       | Compass* (optional)
     - Estimate roll and pitch using zero-velocity pseudo aiding. If heading (compass)
       measurements are available, yaw can also be estimated.
   * - :class:`~smsfusion.AMEKF`
     - | Attitude,
       | Gyro bias
     - | Gravity reference,
       | Compass (optional)
     - Estimate roll and pitch using gravity reference aiding. If heading (compass)
       measurements are available, yaw can also be estimated.

