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


    fs = 10.24  # Sampling rate in Hz
    t, pos, vel, euler, acc, gyro = sf.benchmark_full_pva_beat_202311A(fs)

This will
