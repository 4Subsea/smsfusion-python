Quickstart
==========
This is a quick introduction to the `SMS Fusion` Python package. ``smsfusion``
provides Python implementations of INS algorithms complementing the `SMS Motion`
hardware.


Measurement data
----------------
This quickstart guide assumes that you have access to accelerometer and gyroscope
data from an IMU sensor, and ideally position and heading data from other aiding
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
    imu_noise = sf.noise.IMUNoise()(fs, len(acc))
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


Choosing a filter
-----------------
Three versions of the `multiplicative extended Kalman filter` (MEKF) are available:

- :class:`~smsfusion.AMEKF`: Estimates attitude and gyroscope bias.
- :class:`~smsfusion.VAMEKF`: Estimates velocity, attitude and gyroscope bias.
- :class:`~smsfusion.PVAMEKF`: Estimates position, velocity, attitude and gyroscope bias.

The filters differ only in the states they estimate (see the above table), and hence
the type of external aiding they support.








Kladd
-----

The following table provides a summary:


.. list-table::
   :header-rows: 1
   :widths: 15 15 25 45

   * - Filter
     - States
     - Aiding
     - When to use
   * - :class:`~smsfusion.PVAMEKF`
     - | Position,
       | Velocity,
       | Attitude,
       | Gyro bias
     - | Position (GNSS),
       | Velocity (GNSS),
       | Heading (compass)
     - Estimate all degrees of freedom when full aiding is available.
   * - :class:`~smsfusion.VAMEKF`
     - | Velocity,
       | Attitude,
       | Gyro bias
     - | Velocity (GNSS)
       | Heading (compass)
     - Estimate velocity and attitude when velocity and heading aiding is available.
   * - :class:`~smsfusion.AMEKF`
     - | Attitude,
       | Gyro bias
     - Heading (compass)
     - Estimate attitude when heading aiding is available.


Inertial navigation primer
--------------------------
Measurement data from an `inertial measurement unit` (IMU) forms the backbone of an
`inertial navigation system` (INS). These measurements are integrated to estimate the
position, velocity, and attitude (PVA) of the moving object to which the IMU is attached.
Since the IMU's measurements are subject to noise and bias, the PVA estimates will drift
over time if they are not corrected. Thus, aided INS (AINS) systems incorporate additional
long-term stable aiding measurements to ensure convergence and stability of the INS.
The aiding measurements are typically provided by a `global navigation satellite system`
(GNSS) and a compass, providing absolute position, velocity, and heading information.

Internally, an AINS uses a fusion filter to estimate its states. ``smsfusion`` provides
Python implementations of a type of fusion filter known as the `multiplicative extended
Kalman filter` (MEKF). Three versions of the MEKF filter are available: :class:`~smsfusion.AMEKF`,
:class:`~smsfusion.VAMEKF`, and :class:`~smsfusion.PVAMEKF`. In this quickstart guide
we will demonstrate how to use these MEKF filters to estimate position, velocity
and/or attitude of a moving body using IMU measurements and optional external aiding
measurements.


Choosing a filter
-----------------
The choice of which MEKF filter to use depends on the states you need to estimate,
and the availability of external aiding. The table below summarizes the different
MEKF filters and when to use them.

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Filter
     - States
     - When to use
   * - :class:`~smsfusion.PVAMEKF`
     - Position, velocity, attitude, gyroscope bias
     - Estimation of all states when full aiding is available.
   * - :class:`~smsfusion.VAMEKF`
     - Velocity, attitude, gyroscope bias
     - Estimation of velocity and attitude when velocity aiding is available.
   * - :class:`~smsfusion.AMEKF`
     - Attitude, gyroscope bias
     - Attitude-only estimation when no external aiding is available.

Measurement data
----------------
The filters are updated with velocity increments, ``dvel`` (m/s), and attitude
increments, ``dtheta`` (rad), rather than with specific force and angular rate
directly. For most applications, the simple approximations ``dvel = f * dt``
and ``dtheta = w * dt`` are sufficient, where ``f`` is the specific force in
m/s^2, ``w`` is the angular rate in rad/s, and ``dt = 1 / fs`` is the time step.
For higher accuracy, e.g., when downsampling high-rate IMU data, use
:class:`~smsfusion.ConingScullingAlg` to compute coning and sculling corrected
increments.

If you do not have access to real measurements, synthetic data can be generated
with the ``benchmark`` and ``noise`` modules:

.. code-block:: python

    import numpy as np
    import smsfusion as sf
    from smsfusion.benchmark import benchmark_full_pva_beat_202311A


    fs = 10.24  # sampling rate in Hz
    dt = 1.0 / fs
    t, pos, vel, euler, acc, gyro = benchmark_full_pva_beat_202311A(fs)

    # Add IMU noise
    err_acc = sf.constants.ERR_ACC_MOTION2
    err_gyro = sf.constants.ERR_GYRO_MOTION2
    imu_noise = sf.noise.IMUNoise(err_acc, err_gyro)(fs, len(t))
    dvel = (acc + imu_noise[:, :3]) * dt
    dtheta = (gyro + imu_noise[:, 3:]) * dt

    # Add aiding noise
    rng = np.random.default_rng()
    pos_noise_std = 0.1  # m
    vel_noise_std = 0.05  # m/s
    head_noise_std = 0.01  # rad
    pos_aid = pos + pos_noise_std * rng.standard_normal(pos.shape)
    vel_aid = vel + vel_noise_std * rng.standard_normal(vel.shape)
    head_aid = euler[:, 2] + head_noise_std * rng.standard_normal(len(t))

AMEKF - attitude only
---------------------
:class:`~smsfusion.AMEKF` is the lightest filter. It uses the accelerometer and
the known direction of gravity to correct roll and pitch, and optionally a
heading measurement to correct yaw:

.. code-block:: python

    amekf = sf.AMEKF(fs)

    euler_est = []
    for dvel_i, dtheta_i, head_i in zip(dvel, dtheta, head_aid):
        amekf.update(dvel_i, dtheta_i, head=head_i, head_var=head_noise_std**2)
        euler_est.append(amekf.euler())
    euler_est = np.array(euler_est)

Omit ``head`` to run in VRU mode, in which case yaw will drift.

VAMEKF - attitude with zero-velocity aiding
-------------------------------------------
:class:`~smsfusion.VAMEKF` also estimates velocity, which allows it to separate
gravity from the body's own acceleration. By default, it applies zero-velocity
pseudo aiding (``vel=(0, 0, 0)`` with a large variance), which assumes that the
body is stationary on average. This gives more accurate attitude estimates than
:class:`~smsfusion.AMEKF` for bodies in oscillating motion, e.g., on a vessel.

The gravitational acceleration must be known. Use :func:`~smsfusion.gravity`
to compute it from the latitude:

.. code-block:: python

    vamekf = sf.VAMEKF(fs, g=sf.gravity(lat=60.0))

    euler_est = []
    for dvel_i, dtheta_i, head_i in zip(dvel, dtheta, head_aid):
        vamekf.update(dvel_i, dtheta_i, head=head_i, head_var=head_noise_std**2)
        euler_est.append(vamekf.euler())
    euler_est = np.array(euler_est)

The zero-velocity aiding can be tuned through the ``vel_var`` argument of
:meth:`~smsfusion.VAMEKF.update`.

PVAMEKF - full position, velocity and attitude
----------------------------------------------
:class:`~smsfusion.PVAMEKF` estimates all degrees of freedom, and should be used
when position, velocity and heading aiding are available. If the position aiding
sensor (e.g., a GNSS antenna) is offset from the IMU, specify the offset with
the ``lever_arm`` argument:

.. code-block:: python

    pvamekf = sf.PVAMEKF(fs, g=sf.gravity(lat=60.0))

    pos_est, vel_est, euler_est = [], [], []
    for dvel_i, dtheta_i, p_i, v_i, head_i in zip(dvel, dtheta, pos_aid, vel_aid, head_aid):
        pvamekf.update(
            dvel_i,
            dtheta_i,
            pos=p_i,
            pos_var=pos_noise_std**2 * np.ones(3),
            vel=v_i,
            vel_var=vel_noise_std**2 * np.ones(3),
            head=head_i,
            head_var=head_noise_std**2,
        )
        pos_est.append(pvamekf.position())
        vel_est.append(pvamekf.velocity())
        euler_est.append(pvamekf.euler())
    pos_est = np.array(pos_est)
    vel_est = np.array(vel_est)
    euler_est = np.array(euler_est)

Pass ``None`` for any aiding measurement that is unavailable at a given time
step. Without position and velocity aiding, the filter falls back to zero-position
and zero-velocity pseudo aiding.

Smoothing
---------
When the full measurement sequence is available (post-processing), the
:class:`~smsfusion.FixedIntervalSmoother` refines the estimates of a
:class:`~smsfusion.PVAMEKF` by performing a backward sweep with the
Rauch-Tung-Striebel (RTS) algorithm after the forward pass. Its ``update()``
method takes the same arguments as :meth:`~smsfusion.PVAMEKF.update`, and the
smoothed estimates are returned for all time steps:

.. code-block:: python

    smoother = sf.FixedIntervalSmoother(sf.PVAMEKF(fs, g=sf.gravity(lat=60.0)))

    for dvel_i, dtheta_i, p_i, head_i in zip(dvel, dtheta, pos_aid, head_aid):
        smoother.update(
            dvel_i,
            dtheta_i,
            pos=p_i,
            pos_var=pos_noise_std**2 * np.ones(3),
            head=head_i,
            head_var=head_noise_std**2,
        )

    pos_est = smoother.position()  # shape (n, 3)
    euler_est = smoother.euler()  # shape (n, 3)
