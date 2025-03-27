Quickstart
==========
TODO.

Get measurement data
---------------------
Using the ``benchmark`` module, you can generate synthetic 3D motion data with ``smsfusion``.
For example, you can generate beating signals representing position, velocity and
attitude (PVA) degrees of freedom using :func:`~smsfusion.benchmark.benchmark_full_pva_beat_202311A`:

.. code-block:: python

    from smsfusion.benchmark import benchmark_full_pva_beat_202311A


    fs = 10.24  # Sampling rate in Hz
    t, pos, vel, euler, acc, gyro = sf.benchmark_full_pva_beat_202311A(fs)
