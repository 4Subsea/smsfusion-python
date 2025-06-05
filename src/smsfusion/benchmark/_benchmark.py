from __future__ import annotations

import abc

import numpy as np
from numpy.typing import ArrayLike, NDArray

from smsfusion._ins import gravity
from smsfusion._transforms import _inv_angular_matrix_from_euler, _rot_matrix_from_euler


class _Signal(abc.ABC):
    """
    Abstract class for benchmark signals.
    """

    def __call__(
        self,
        t: ArrayLike,
        phase: float = 0.0,
        phase_degrees: float = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Generate a beating signal.

        Parameters
        ----------
        t : array-like, shape (N,)
            A time array in seconds.
        phase : float, defualt 0.0
            The phase of the main sinusiodal signal.
        phase_degrees : bool, default True
            Whether ``phase`` is in degrees or radians.

        Return
        ------
        y : numpy.ndarray, shape (N,)
            The generated signal.
        dydt : numpy.ndarray, shape (N,)
            The time derivative of the signal.
        d2ydt2 : numpy.ndarray, shape (N,)
            The second time derivative of the signal.
        """
        t = np.asarray_chkfinite(t)
        if phase_degrees:
            phase = np.radians(phase)
        return self._y(t, phase), self._dydt(t, phase), self._d2ydt2(t, phase)

    @abc.abstractmethod
    def _y(self, t: NDArray[np.float64], phase: float) -> NDArray[np.float64]:
        raise NotImplementedError

    @abc.abstractmethod
    def _dydt(self, t: NDArray[np.float64], phase: float) -> NDArray[np.float64]:
        raise NotImplementedError

    @abc.abstractmethod
    def _d2ydt2(self, t: NDArray[np.float64], phase: float) -> NDArray[np.float64]:
        raise NotImplementedError


class BeatSignal(_Signal):
    """
    Generate a unit amplitude beating sinusoidal signal, and its first and second
    time derivatives.

    This function creates a signal with a main frequency and a beating
    frequency, resulting in a wave that appears to "beat" or vary in
    amplitude over time.

    Parameters
    ----------
    f_main : float
        The main frequency of the sinusoidal signal.
    f_beat : float
        The beating frequency, which controls the
        variation in amplitude.
    freq_hz : bool, default True.
        Whether the frequencies, ``f_main`` and ``f_beat``, are in Hz or rad/s.

    Notes
    -----
    The signal may be expressed as::

        y = sin(f_beat / 2.0 * t) * cos(f_main * t + phase)
    """

    def __init__(self, f_main: float, f_beat: float, freq_hz: bool = True) -> None:
        self._f_main = f_main
        self._f_beat = f_beat

        if freq_hz:
            self._f_main = self._f_main * 2.0 * np.pi
            self._f_beat = self._f_beat * 2.0 * np.pi

    def _y(self, t: NDArray[np.float64], phase: float) -> NDArray[np.float64]:
        """
        Generate a unit amplitude beating signal.
        """
        main = np.cos(self._f_main * t + phase)
        beat = np.sin(self._f_beat / 2.0 * t)
        y = beat * main
        return y  # type: ignore[no-any-return]

    def _dydt(self, t: NDArray[np.float64], phase: float) -> NDArray[np.float64]:
        """
        Generate the first time derivative of a unit amplitue beating signal.
        """
        main = np.cos(self._f_main * t + phase)
        beat = np.sin(self._f_beat / 2.0 * t)
        dmain = -self._f_main * np.sin(self._f_main * t + phase)
        dbeat = self._f_beat / 2.0 * np.cos(self._f_beat / 2.0 * t)

        dydt = dbeat * main + beat * dmain
        return dydt  # type: ignore[no-any-return]

    def _d2ydt2(self, t: NDArray[np.float64], phase: float) -> NDArray[np.float64]:
        """
        Generate the second time derivative of a unit amplitue beating signal.
        """
        main = np.cos(self._f_main * t + phase)
        beat = np.sin(self._f_beat / 2.0 * t)
        dmain = -self._f_main * np.sin(self._f_main * t + phase)
        dbeat = self._f_beat / 2.0 * np.cos(self._f_beat / 2.0 * t)
        d2main = -((self._f_main) ** 2) * np.cos(self._f_main * t + phase)
        d2beat = -((self._f_beat / 2.0) ** 2) * np.sin(self._f_beat / 2.0 * t)

        d2ydt2 = dbeat * dmain + d2beat * main + beat * d2main + dbeat * dmain

        return d2ydt2  # type: ignore[no-any-return]


class ChirpSignal(_Signal):
    """
    Generate a unit amplitude chirp sinusoidal signal, and its first and second
    time derivative.

    This function creates a signal with a frequency that appears to vary in time.
    The frequency oscillates between 0. and a maximum frequency at a specific
    rate.

    Parameters
    ----------
    f_max : float
        The max frequency the signal ramps up to, before ramping down to zero.
    f_os : float
        The frequency oscillation rate.
    freq_hz : bool, default True.
        Whether the frequencies, ``f_max`` and ``f_os``, are in Hz or radians.

    Notes
    -----
    The signal may be expressed as::

        phi = 2 * f_max / f_os * sin(f_os * t)
        y = sin(phi + phase)
    """

    def __init__(self, f_max: float, f_os: float, freq_hz: bool = True) -> None:
        self._f_max = f_max
        self._f_os = f_os

        if freq_hz:
            self._f_max *= 2.0 * np.pi
            self._f_os *= 2.0 * np.pi

    def _y(self, t: NDArray[np.float64], phase: float) -> NDArray[np.float64]:
        """
        Generate a unit amplitude chirp signal with oscillating frequency.
        """
        phi = 2.0 * self._f_max / self._f_os * np.sin(self._f_os / 2.0 * t)
        y = np.sin(phi + phase)
        return y  # type: ignore[no-any-return]

    def _dydt(self, t: NDArray[np.float64], phase: float) -> NDArray[np.float64]:
        """
        Generate the time derivative of a unit amplitude chirp signal with
        oscillating frequency.
        """
        phi = 2.0 * self._f_max / self._f_os * np.sin(self._f_os / 2.0 * t)
        dphi = self._f_max * np.cos(self._f_os / 2.0 * t)
        dydt = dphi * np.cos(phi + phase)
        return dydt  # type: ignore[no-any-return]

    def _d2ydt2(self, t: NDArray[np.float64], phase: float) -> NDArray[np.float64]:
        """
        Generate the second time derivative of a unit amplitude chirp signal with
        oscillating frequency.
        """
        phi = 2.0 * self._f_max / self._f_os * np.sin(self._f_os / 2.0 * t)
        dphi = self._f_max * np.cos(self._f_os / 2.0 * t)
        d2phi = -self._f_max * self._f_os / 2.0 * np.sin(self._f_os / 2.0 * t)
        d2ydt2 = -(dphi**2) * np.sin(phi + phase) + d2phi * np.cos(phi + phase)
        return d2ydt2  # type: ignore[no-any-return]


def _benchmark_helper(
    duration: float,
    amplitude: tuple[float, float, float, float, float, float],
    mean: tuple[float, float, float, float, float, float],
    phase: tuple[float, float, float, float, float, float],
    signal_family: _Signal,
    fs: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Generate benchmark scenarios for INS sensor fusion performance evaluation.

    The general helper function works by first generating position and attitude
    based on the provided input, and then derives the remaining variables. The
    final output consists of:

    * Time [s]
    * Position [m]
    * Velocity [m/s]
    * Attitude [rad] (Euler angles, see Notes)
    * Accelerations [m/s**2] in body frame (corresponding to accelerometer measurements)
    * Angular rates [rad/s] in body frame (corresponding to gyroscope measurements)

    Position, velocity, and attitude are relative to the *NED* frame and expressed
    in the *NED* frame. The position origin is at (0., 0., 0.).

    Parameters
    ----------
    duration : float
        Duration of the generated signals in seconds.
    amplitude : tuple of (float, float, float, float, float, float)
        Amplitude of the signals generated by the generator defined in ``signal_family``.
        The values represent the amplitude of postions in x, y, and z directions in
        meters, and Euler angles as roll, pitch, and yaw in radians (see Notes).
        The order is given as [pos_x, pos_y, pos_z, roll, pitch, yaw].
    mean : tuple of (float, float, float, float, float, float)
        Mean values of the generated signals. Follows the same conventions as
        ``amplitude``.
    phase : tuple of (float, float, float, float, float, float)
        Phase values in radians passed on to the signal generator in
        ``signal_family``. Otherwise, follows the same conventions as
        ``amplitude``.
    signal_family : {:clas:`smsfusion.benchmark.BeatSignal`, :clas:`smsfusion.benchmark.BeatSignal`}
        Instance of a class used to generate the signals.
    fs : float
        Sampling rate of the outputs in hertz.

    Returns
    -------
    t : numpy.ndarray, shape (N,)
        Time in seconds.
    position : numpy.ndarray, shape (N, 3)
        Position [m] of the body.
    velocity : numpy.ndarray, shape (N, 3)
        Velocity [m/s] of the body.
    euler : numpy.ndarray, shape (N, 3)
        Attitude of the body in Euler angles [rad], see Notes.
    acc : numpy.ndarray, shape (N, 3)
        Accelerations [m/s**2] in body frame (corresponding to accelerometer measurements).
    gyro : numpy.ndarray, shape (N, 3)
        Angular rates [rad/s] in body frame (corresponding to gyroscope measurements).

    Notes
    -----
    The length of the signals are determined by ``np.arange(0.0, duration, 1.0 / fs)``.

    The Euler angles describe how to transition from the 'NED' frame to the 'body'
    frame through three consecutive intrinsic and passive rotations in the ZYX order:

    #. A rotation by an angle gamma (often called yaw) about the z-axis.
    #. A subsequent rotation by an angle beta (often called pitch) about the y-axis.
    #. A final rotation by an angle alpha (often called roll) about the x-axis.

    This sequence of rotations is used to describe the orientation of the 'body' frame
    relative to the 'NED' frame in 3D space.

    Intrinsic rotations mean that the rotations are with respect to the changing
    coordinate system; as one rotation is applied, the next is about the axis of
    the newly rotated system.

    Passive rotations mean that the frame itself is rotating, not the object
    within the frame.
    """
    t = np.arange(0.0, duration, 1.0 / fs)

    amp_x, amp_y, amp_z, amp_roll, amp_pitch, amp_yaw = amplitude
    mean_x, mean_y, mean_z, mean_roll, mean_pitch, mean_yaw = mean
    phase_x, phase_y, phase_z, phase_roll, phase_pitch, phase_yaw = phase

    pos_x_, vel_x_, acc_x_ = signal_family(t, phase_x, phase_degrees=False)
    pos_x = mean_x + amp_x * pos_x_
    vel_x = amp_x * vel_x_
    acc_x = amp_x * acc_x_

    pos_y_, vel_y_, acc_y_ = signal_family(t, phase_y, phase_degrees=False)
    pos_y = mean_y + amp_y * pos_y_
    vel_y = amp_y * vel_y_
    acc_y = amp_y * acc_y_

    pos_z_, vel_z_, acc_z_ = signal_family(t, phase_z, phase_degrees=False)
    pos_z = mean_z + amp_z * pos_z_
    vel_z = amp_z * vel_z_
    acc_z = amp_z * acc_z_

    roll_, droll_, _ = signal_family(t, phase_roll, phase_degrees=False)
    roll = mean_roll + amp_roll * roll_
    droll = amp_roll * droll_

    pitch_, dpitch_, _ = signal_family(t, phase_pitch, phase_degrees=False)
    pitch = mean_pitch + amp_pitch * pitch_
    dpitch = amp_pitch * dpitch_

    yaw_, dyaw_, _ = signal_family(t, phase_yaw, phase_degrees=False)
    yaw = mean_yaw + amp_yaw * yaw_
    dyaw = amp_yaw * dyaw_

    position = np.column_stack((pos_x, pos_y, pos_z))
    velocity = np.column_stack((vel_x, vel_y, vel_z))
    acceleration = np.column_stack((acc_x, acc_y, acc_z))
    attitude = np.column_stack((roll, pitch, yaw))
    dattitude = np.column_stack((droll, dpitch, dyaw))

    accelerometer = []
    gyroscope = []
    for euler_i, deuler_i, acc_i in zip(attitude, dattitude, acceleration):
        gyroscope.append(_inv_angular_matrix_from_euler(euler_i).dot(deuler_i))
        accelerometer.append(
            _rot_matrix_from_euler(euler_i).T.dot(
                acc_i + np.array([0.0, 0.0, -gravity()])
            )
        )

    return (
        t,
        position,
        velocity,
        attitude,
        np.asarray(accelerometer),
        np.asarray(gyroscope),
    )


def benchmark_pure_attitude_beat_202311A(
    fs: float = 10.24,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """
    Generate a benchmark with pure attitude for performance testing of INS/AHRS/VRU
    sensor fusion algorithms.

    The benchmark scenario is 30 minutes long. It generates Euler angles
    (roll, pitch, and yaw), and the corresponding accelerometer and
    gyroscope signals. Note that the generated motion is pure rotations.

    The generated signals are "beating" with maximum amplitudes corresponding to
    5 degrees. The main signal frequency is 0.1 Hz while the "beating" frequency
    is 0.01 Hz. The phases for roll, pitch, and yaw are 0.0, 45.0, and 90.0 degrees
    respectively. See :class:`smsfusion.benchmark.BeatSignal` for details.

    Parameters
    ----------
    fs : float, default 10.24
        Sampling frequency in hertz of the generated signals.

    Returns
    -------
    t : numpy.ndarray, shape (N,)
        Time in seconds.
    euler : numpy.ndarray, shape (N, 3)
        Attitude of the body in Euler angles [rad], see Notes.
    acc : numpy.ndarray, shape (N, 3)
        Accelerations [m/s**2] in body frame (corresponding to accelerometer measurements).
    gyro : numpy.ndarray, shape (N, 3)
        Angular rates [rad/s] in body frame (corresponding to gyroscope measurements).

    Notes
    -----
    The Euler angles describe how to transition from the 'NED' frame to the 'body'
    frame through three consecutive intrinsic and passive rotations in the ZYX order:

    #. A rotation by an angle gamma (often called yaw) about the z-axis.
    #. A subsequent rotation by an angle beta (often called pitch) about the y-axis.
    #. A final rotation by an angle alpha (often called roll) about the x-axis.

    This sequence of rotations is used to describe the orientation of the 'body' frame
    relative to the 'NED' frame in 3D space.

    Intrinsic rotations mean that the rotations are with respect to the changing
    coordinate system; as one rotation is applied, the next is about the axis of
    the newly rotated system.

    Passive rotations mean that the frame itself is rotating, not the object
    within the frame.
    """
    duration = 1800.0  # 30 minutes
    amplitude = np.radians(np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0]))
    mean = np.radians(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    phase = np.radians(np.array([0.0, 0.0, 0.0, 0.0, 45.0, 90.0]))

    f_main = 0.1
    f_beat = 0.01

    t, _, _, euler, acc, gyro = _benchmark_helper(
        duration, amplitude, mean, phase, BeatSignal(f_main, f_beat), fs
    )
    return t, euler, acc, gyro


def benchmark_pure_attitude_chirp_202311A(
    fs: float = 10.24,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """
    Generate a benchmark with pure attitude for performance testing of INS/AHRS/VRU
    sensor fusion algorithms.

    The benchmark scenario is 30 minutes long. It generates Euler angles
    (roll, pitch, and yaw), and the corresponding accelerometer and
    gyroscope signals. Note that the generated motion is pure rotations.

    The generated signals have a frequency that appears to vary in time and
    amplitudes corresponding to 5 degrees. The signal frequency oscillates
    between 0.0 Hz and 0.25 Hz every 100 seconds. The phases for roll, pitch,
    and yaw are 0.0, 45.0, and 90.0 degrees respectively. See
    :class:`smsfusion.benchmark.ChirpSignal` for details.

    Parameters
    ----------
    fs : float, default 10.24
        Sampling frequency in hertz of the generated signals.

    Returns
    -------
    t : numpy.ndarray, shape (N,)
        Time in seconds.
    euler : numpy.ndarray, shape (N, 3)
        Attitude of the body in Euler angles [rad], see Notes.
    acc : numpy.ndarray, shape (N, 3)
        Accelerations [m/s**2] in body frame (corresponding to accelerometer measurements).
    gyro : numpy.ndarray, shape (N, 3)
        Angular rates [rad/s] in body frame (corresponding to gyroscope measurements).

    Notes
    -----
    The Euler angles describe how to transition from the 'NED' frame to the 'body'
    frame through three consecutive intrinsic and passive rotations in the ZYX order:

    #. A rotation by an angle gamma (often called yaw) about the z-axis.
    #. A subsequent rotation by an angle beta (often called pitch) about the y-axis.
    #. A final rotation by an angle alpha (often called roll) about the x-axis.

    This sequence of rotations is used to describe the orientation of the 'body' frame
    relative to the 'NED' frame in 3D space.

    Intrinsic rotations mean that the rotations are with respect to the changing
    coordinate system; as one rotation is applied, the next is about the axis of
    the newly rotated system.

    Passive rotations mean that the frame itself is rotating, not the object
    within the frame.
    """
    duration = 1800.0  # 30 minutes
    amplitude = np.radians(np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0]))
    mean = np.radians(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    phase = np.radians(np.array([0.0, 0.0, 0.0, 0.0, 45.0, 90.0]))

    f_max = 0.25
    f_os = 0.01

    t, _, _, euler, acc, gyro = _benchmark_helper(
        duration, amplitude, mean, phase, ChirpSignal(f_max, f_os), fs
    )
    return t, euler, acc, gyro


def benchmark_full_pva_beat_202311A(
    fs: float = 10.24,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Generate a benchmark with full position, velocity, and attitude (PVA) for
    performance testing of INS/AHRS/VRU sensor fusion algorithms.

    The benchmark scenario is 30 minutes long. It generates full position, velocity
    and attitude (PVA), and the corresponding accelerometer and gyroscope signals.

    The generated position reference signals are characterized by:

    * "Beating" signal. See ``BeatSignal`` for details.
    * Maximum amplitude of 0.5 m.
    * The phases for x-, y-, and z-axis are 0.0, 30.0, and 60.0 degrees respectively.

    The generated Euler reference signals are characterized by:

    * "Beating" signal. See :class:`smsfusion.benchmark.BeatSignal` for details.
    * Maximum amplitude of 5 degrees.
    * The phases for roll, pitch, and yaw are 90.0, 120.0, and 150.0 degrees respectively.

    The other reference signals will be exact analythical derivatives of these signals.

    Parameters
    ----------
    fs : float, default 10.24
        Sampling frequency in hertz of the generated signals.

    Returns
    -------
    t : numpy.ndarray, shape (N,)
        Time in seconds.
    pos : numpy.ndarray, shape (N, 3)
        Position [m] of the body relative to the NED frame.
    vel : numpy.ndarray, shape (N, 3)
        Velocity [m/s] of the body relative to the NED frame.
    euler : numpy.ndarray, shape (N, 3)
        Attitude of the body in Euler angles [rad], see Notes.
    acc : numpy.ndarray, shape (N, 3)
        Accelerations [m/s**2] in body frame (corresponding to accelerometer measurements).
    gyro : numpy.ndarray, shape (N, 3)
        Angular rates [rad/s] in body frame (corresponding to gyroscope measurements).

    Notes
    -----
    The Euler angles describe how to transition from the 'NED' frame to the 'body'
    frame through three consecutive intrinsic and passive rotations in the ZYX order:

    #. A rotation by an angle gamma (often called yaw) about the z-axis.
    #. A subsequent rotation by an angle beta (often called pitch) about the y-axis.
    #. A final rotation by an angle alpha (often called roll) about the x-axis.

    This sequence of rotations is used to describe the orientation of the 'body' frame
    relative to the 'NED' frame in 3D space.

    Intrinsic rotations mean that the rotations are with respect to the changing
    coordinate system; as one rotation is applied, the next is about the axis of
    the newly rotated system.

    Passive rotations mean that the frame itself is rotating, not the object
    within the frame.
    """
    duration = 1800.0  # 30 minutes
    amplitude = (0.5, 0.5, 0.5, np.radians(5.0), np.radians(5.0), np.radians(5.0))
    mean = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    phase = np.radians(np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0]))

    f_main = 0.1
    f_beat = 0.01

    t, pos, vel, euler, acc, gyro = _benchmark_helper(
        duration, amplitude, mean, phase, BeatSignal(f_main, f_beat), fs
    )
    return t, pos, vel, euler, acc, gyro


def benchmark_full_pva_chirp_202311A(
    fs: float = 10.24,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Generate a benchmark with full position, velocity, and attitude (PVA) for
    performance testing of INS/AHRS/VRU sensor fusion algorithms.

    The benchmark scenario is 30 minutes long. It generates full position, velocity
    and attitude (PVA), and the corresponding accelerometer and gyroscope signals.

    The generated position reference signals are characterized by:

    * "Chirp" signal. See :class:`smsfusion.benchmark.ChirpSignal` for details.
    * Maximum amplitude of 0.5 m.
    * The phases for x-, y-, and z-axis are 0.0, 30.0, and 60.0 degrees respectively.

    The generated Euler reference signals are characterized by:

    * "Chirp" signal. See :class:`smsfusion.benchmark.ChirpSignal` for details.
    * Maximum amplitude of 5 degrees.
    * The phases for roll, pitch, and yaw are 90.0, 120.0, and 150.0 degrees respectively.

    The other reference signals will be exact analythical derivatives of these signals.

    Parameters
    ----------
    fs : float, default 10.24
        Sampling frequency in hertz of the generated signals.

    Returns
    -------
    t : numpy.ndarray, shape (N,)
        Time in seconds.
    pos : numpy.ndarray, shape (N, 3)
        Position [m] of the body relative to the NED frame.
    vel : numpy.ndarray, shape (N, 3)
        Velocity [m/s] of the body relative to the NED frame.
    euler : numpy.ndarray, shape (N, 3)
        Attitude of the body in Euler angles [rad], see Notes.
    acc : numpy.ndarray, shape (N, 3)
        Accelerations [m/s**2] in body frame (corresponding to accelerometer measurements).
    gyro : numpy.ndarray, shape (N, 3)
        Angular rates [rad/s] in body frame (corresponding to gyroscope measurements).

    Notes
    -----
    The Euler angles describe how to transition from the 'NED' frame to the 'body'
    frame through three consecutive intrinsic and passive rotations in the ZYX order:

    #. A rotation by an angle gamma (often called yaw) about the z-axis.
    #. A subsequent rotation by an angle beta (often called pitch) about the y-axis.
    #. A final rotation by an angle alpha (often called roll) about the x-axis.

    This sequence of rotations is used to describe the orientation of the 'body' frame
    relative to the 'NED' frame in 3D space.

    Intrinsic rotations mean that the rotations are with respect to the changing
    coordinate system; as one rotation is applied, the next is about the axis of
    the newly rotated system.

    Passive rotations mean that the frame itself is rotating, not the object
    within the frame.
    """
    duration = 1800.0  # 30 minutes
    amplitude = (0.5, 0.5, 0.5, np.radians(5.0), np.radians(5.0), np.radians(5.0))
    mean = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    phase = np.radians(np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0]))

    f_max = 0.25
    f_os = 0.01

    t, pos, vel, euler, acc, gyro = _benchmark_helper(
        duration, amplitude, mean, phase, ChirpSignal(f_max, f_os), fs
    )
    return t, pos, vel, euler, acc, gyro
