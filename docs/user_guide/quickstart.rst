Quickstart
==========
This is a short introduction to `SMS Fusion`: a Python library with inertial navigation
algorithms for `SMS Motion`. Although the primary purpose is to complement the SMS Motion
hardware, the algorithms can be used with any IMU sensor.

The core algorithms in ``smsfusion`` are a family of fusion filters known as
`multiplicative extended Kalman filters` (MEKF). Three flavors of the MEKF filter
are available:

- :class:`~smsfusion.AMEKF`: estimates attitude and gyroscope bias.
- :class:`~smsfusion.VAMEKF`: estimates velocity, attitude and gyroscope bias.
- :class:`~smsfusion.PVAMEKF`: estimates position, velocity, attitude and gyroscope bias.

The three filters differ only in the number of states they estimate, and hence also
the type of external aiding they support. :class:`~smsfusion.PVAMEKF` is the most
comprehensive filter, estimating all degrees of freedom (DOFs). By leveraging external
aiding measurements, this filter can achieve the highest accuracy in all state estimates.
Thus, if you have access to full external aiding (GNSS and compass), this is the
filter to use. The :class:`~smsfusion.VAMEKF` and :class:`~smsfusion.AMEKF` are
degenerated versions of the this filter, where some of the states are removed,
resulting in simpler filters with reduced computational complexity at the expense
of potentially lower accuracy. In aiding denied scenarios, where external aiding
is not feasible or simply not available, the simpler filters are more suitable.

The following table lists which MEKF filter and aiding configuration to use for
different scenarios:

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

In the following sections, we will demonstrate how to use the MEKF filters in
different aiding scenarios.


Measurement data
----------------
The next sections assume that you have access to accelerometer and gyroscope
measurements from an IMU sensor, and (depending on the scenario) position, velocity
and heading measurements from other, external aiding sensors. If you don't have
access to such data, you can generate synthetic measurements using the
:mod:`~smsfusion.benchmark` module in ``smsfusion``:

.. code-block:: python

    import smsfusion as sf
    from smsfusion.benchmark import benchmark_full_pva_beat_202311A


    # Position, velocity and attitude (PVA) reference signals
    fs = 10.24  # sampling rate in Hz
    t, pos, vel, euler, f, w = benchmark_full_pva_beat_202311A(fs)
    head = euler[:, 2]

    # Add measurement noise
    imu_noise = sf.noise.IMUNoise(seed=0)(fs, len(f))
    f_meas = f + imu_noise[:, :3]
    w_meas = w + imu_noise[:, 3:]
    rng = np.random.default_rng(1)
    pos_aid = pos + 0.1 * rng.standard_normal(pos.shape)
    head_aid = head + 0.01 * rng.standard_normal(head.shape)

    # Velocity and attitude increments
    dvel_meas = f_meas / fs
    dtheta_meas = w_meas / fs

Note that the generated position signals are in meters (m), velocity signals are in meters
per second (m/s), and attitude signals are in radians (rad). The accelerometer signals
are in meters per second squared (m/s^2), and the gyroscope signals are in radians
per second (rad/s). If your measurement data is given in other units, you must account
for that in other sections of this quickstart guide.

IMU only (no external aiding) - estimate roll and pitch
-------------------------------------------------------
In aiding denied scenarios, where you don't have access to long-term stable aiding
measurements, only the roll and pitch degrees of freedom can be estimated since
these are still observable through accelerometer measurements and the known direction
of gravity.

Using :class:`~smsfusion.AMEKF` with gravity reference aiding is the most lightweight
and robust choice for estimating roll and pitch without external aiding:

.. code-block:: python

    import smsfusion as sf


    mekf = sf.AMEKF(fs)

    roll_pitch_est = []
    for dvel_i, dtheta_i in zip(dvel_meas, dtheta_meas):
        mekf.update(dvel_i, dtheta_i)
        roll_pitch_est.append(mekf.euler()[:2])
    roll_pitch_est = np.array(roll_pitch_est)

Alternatively, :class:`~smsfusion.VAMEKF` with zero-velocity aiding can be used:

.. code-block:: python

    import smsfusion as sf


    lat = 59.0  # latitude
    mekf = sf.VAMEKF(fs, g=sf.gravity(lat))

    # Zero-velocity update (ZUPT) with 10 m/s standard deviation
    zupt = {"vel": (0.0, 0.0, 0.0), "vel_var": (100.0, 100.0, 100.0)}

    roll_pitch_est = []
    for dvel_i, dtheta_i in zip(dvel_meas, dtheta_meas):
        mekf.update(dvel_i, dtheta_i, **zupt)
        roll_pitch_est.append(mekf.euler()[:2])
    roll_pitch_est = np.array(roll_pitch_est)

.. note::

    The :class:`~smsfusion.VAMEKF` with zero-velocity aiding has shown higher accuracy
    compared to the :class:`~smsfusion.AMEKF` with gravity reference aiding. However,
    to ensure stability of the filter, a calibrated accelerometer is then required,
    and the correct local gravitational acceleration must be set.


IMU + compass aiding - estimate roll, pitch and yaw
---------------------------------------------------
If compass (heading) aiding is available, yaw can also be estimated along with
roll and pitch.

Using :class:`~smsfusion.AMEKF` with gravity reference aiding and heading aiding
is the most lightweight and robust choice for estimating roll, pitch, and yaw:

.. code-block:: python

    import smsfusion as sf


    mekf = sf.AMEKF(fs)

    euler_est = []
    for dvel_i, dtheta_i, head_i in zip(dvel_meas, dtheta_meas, head_meas):
        mekf.update(dvel_i, dtheta_i, head=head_i, head_var=0.01**2)
        euler_est.append(mekf.euler())
    euler_est = np.array(euler_est)

Alternatively, :class:`~smsfusion.VAMEKF` with zero-velocity and heading aiding
can be used:

.. code-block:: python

    import smsfusion as sf


    lat = 59.0  # latitude
    mekf = sf.VAMEKF(fs, g=sf.gravity(lat))

    # Zero-velocity update (ZUPT) with 10 m/s standard deviation
    zupt = {"vel": (0.0, 0.0, 0.0), "vel_var": (100.0, 100.0, 100.0)}

    euler_est = []
    for dvel_i, dtheta_i, head_i in zip(dvel_meas, dtheta_meas, head_meas):
        mekf.update(dvel_i, dtheta_i, head=head_i, head_var=0.01**2, **zupt)
        euler_est.append(mekf.euler())
    euler_est = np.array(euler_est)


IMU + GNSS and compass aiding - estimate position, velocity and attitude
------------------------------------------------------------------------

