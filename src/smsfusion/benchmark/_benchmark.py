from __future__ import annotations

import abc
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike, NDArray

from smsfusion._ins import gravity
from smsfusion._transforms import _inv_angular_matrix_from_euler, _rot_matrix_from_euler
from smsfusion.simulate import BeatDOF, ChirpDOF, IMUSimulator


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
        warn("`BeatSignal`` is deprecated, use ``simulate.BeatDOF`` instead.")
        self._f_main = 2.0 * np.pi * f_main if freq_hz else f_main
        self._f_beat = f_beat * 2.0 * np.pi if freq_hz else f_beat

    def _y(self, t: NDArray[np.float64], phase: float) -> NDArray[np.float64]:
        """
        Generate a unit amplitude beating signal.
        """
        y = BeatDOF(1.0, self._f_main, self._f_beat, phase=phase)._y(t)
        return y  # type: ignore[no-any-return]

    def _dydt(self, t: NDArray[np.float64], phase: float) -> NDArray[np.float64]:
        """
        Generate the first time derivative of a unit amplitue beating signal.
        """
        dydt = BeatDOF(1.0, self._f_main, self._f_beat, phase=phase)._dydt(t)
        return dydt  # type: ignore[no-any-return]

    def _d2ydt2(self, t: NDArray[np.float64], phase: float) -> NDArray[np.float64]:
        """
        Generate the second time derivative of a unit amplitue beating signal.
        """
        d2ydt2 = BeatDOF(1.0, self._f_main, self._f_beat, phase=phase)._d2ydt2(t)
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
        warn("`ChirpSignal` is deprecated, use `simulate.ChirpDOF` instead.")
        self._f_max = 2.0 * np.pi * f_max if freq_hz else f_max
        self._f_os = 2.0 * np.pi * f_os if freq_hz else f_os

    def _y(self, t: NDArray[np.float64], phase: float) -> NDArray[np.float64]:
        """
        Generate a unit amplitude chirp signal with oscillating frequency.
        """
        y = ChirpDOF(1.0, self._f_max, self._f_os, phase=phase)._y(t)
        return y  # type: ignore[no-any-return]

    def _dydt(self, t: NDArray[np.float64], phase: float) -> NDArray[np.float64]:
        """
        Generate the time derivative of a unit amplitude chirp signal with
        oscillating frequency.
        """
        dydt = ChirpDOF(1.0, self._f_max, self._f_os, phase=phase)._dydt(t)
        return dydt  # type: ignore[no-any-return]

    def _d2ydt2(self, t: NDArray[np.float64], phase: float) -> NDArray[np.float64]:
        """
        Generate the second time derivative of a unit amplitude chirp signal with
        oscillating frequency.
        """
        d2ydt2 = ChirpDOF(1.0, self._f_max, self._f_os, phase=phase)._d2ydt2(t)
        return d2ydt2  # type: ignore[no-any-return]


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
    respectively. See :class:`smsfusion.simulate.BeatDOF` for details.

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

    f_main = 0.1
    f_beat = 0.01

    amp = np.radians(5.0)
    alpha = BeatDOF(amp, f_main, f_beat, freq_hz=True, phase=0.0)
    beta = BeatDOF(amp, f_main, f_beat, freq_hz=True, phase=45.0, phase_degrees=True)
    gamma = BeatDOF(amp, f_main, f_beat, freq_hz=True, phase=90.0, phase_degrees=True)
    sim = IMUSimulator(alpha=alpha, beta=beta, gamma=gamma)

    n = int(duration * fs)
    t, _, _, euler, f, w = sim(fs, n, degrees=False)

    return t, euler, f, w


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
    :class:`smsfusion.simulate.ChirpDOF` for details.

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

    f_max = 0.25
    f_os = 0.01

    amp = np.radians(5.0)
    alpha = ChirpDOF(amp, f_max, f_os, freq_hz=True, phase=0.0)
    beta = ChirpDOF(amp, f_max, f_os, freq_hz=True, phase=45.0, phase_degrees=True)
    gamma = ChirpDOF(amp, f_max, f_os, freq_hz=True, phase=90.0, phase_degrees=True)
    sim = IMUSimulator(alpha=alpha, beta=beta, gamma=gamma)

    n = int(duration * fs)
    t, _, _, euler, f, w = sim(fs, n, degrees=False)

    return t, euler, f, w


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

    * "Beating" signal. See ``smsfusion.simulate.BeatDOF`` for details.
    * Maximum amplitude of 0.5 m.
    * The phases for x-, y-, and z-axis are 0.0, 30.0, and 60.0 degrees respectively.

    The generated Euler reference signals are characterized by:

    * "Beating" signal. See :class:`smsfusion.simulate.BeatDOF` for details.
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

    f_main = 0.1
    f_beat = 0.01

    amp_p = 0.5
    amp_r = np.radians(5.0)
    pos_x = BeatDOF(amp_p, f_main, f_beat, freq_hz=True, phase=0.0, phase_degrees=True)
    pos_y = BeatDOF(amp_p, f_main, f_beat, freq_hz=True, phase=30.0, phase_degrees=True)
    pos_z = BeatDOF(amp_p, f_main, f_beat, freq_hz=True, phase=60.0, phase_degrees=True)
    alpha = BeatDOF(amp_r, f_main, f_beat, freq_hz=True, phase=90.0, phase_degrees=True)
    beta = BeatDOF(amp_r, f_main, f_beat, freq_hz=True, phase=120.0, phase_degrees=True)
    gamma = BeatDOF(
        amp_r, f_main, f_beat, freq_hz=True, phase=150.0, phase_degrees=True
    )
    sim = IMUSimulator(pos_x, pos_y, pos_z, alpha, beta, gamma)

    n = int(duration * fs)
    t, pos, vel, euler, f, w = sim(fs, n, degrees=False)

    return t, pos, vel, euler, f, w


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

    * "Chirp" signal. See :class:`smsfusion.simulate.ChirpDOF` for details.
    * Maximum amplitude of 0.5 m.
    * The phases for x-, y-, and z-axis are 0.0, 30.0, and 60.0 degrees respectively.

    The generated Euler reference signals are characterized by:

    * "Chirp" signal. See :class:`smsfusion.simulate.ChirpDOF` for details.
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

    f_max = 0.25
    f_os = 0.01

    amp_p = 0.5
    amp_r = np.radians(5.0)
    pos_x = ChirpDOF(amp_p, f_max, f_os, freq_hz=True, phase=0.0, phase_degrees=True)
    pos_y = ChirpDOF(amp_p, f_max, f_os, freq_hz=True, phase=30.0, phase_degrees=True)
    pos_z = ChirpDOF(amp_p, f_max, f_os, freq_hz=True, phase=60.0, phase_degrees=True)
    alpha = ChirpDOF(amp_r, f_max, f_os, freq_hz=True, phase=90.0, phase_degrees=True)
    beta = ChirpDOF(amp_r, f_max, f_os, freq_hz=True, phase=120.0, phase_degrees=True)
    gamma = ChirpDOF(amp_r, f_max, f_os, freq_hz=True, phase=150.0, phase_degrees=True)
    sim = IMUSimulator(pos_x, pos_y, pos_z, alpha, beta, gamma)

    n = int(duration * fs)
    t, pos, vel, euler, f, w = sim(fs, n, degrees=False)

    return t, pos, vel, euler, f, w
