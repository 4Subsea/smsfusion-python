from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class BeatSignal:
    """
    Generate a unit amplitude sinusoidal signal with a beating effect, and
    its time derivative.

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
        If ``True``, ``f_main`` and ``f_beat`` are in Hz. Otherwise,
        rad/s is assumed.

    Notes
    -----
    The signal may be expressed as:

        ```
        y = amp * sin(f_beat / 2.0 * t) * cos(f_main * t + phase)
        ```
    """

    def __init__(self, f_main: float, f_beat: float, freq_hz: bool = True) -> None:
        self._f_main = f_main
        self._f_beat = f_beat

        if freq_hz:
            self._f_main = self._f_main * 2.0 * np.pi
            self._f_beat = self._f_beat * 2.0 * np.pi

    def __call__(
        self,
        t: ArrayLike,
        phase: float = 0.0,
        phase_degrees: float = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Generate a beating signal.

        Parameters
        ----------
        t : array-like
            A time array in seconds.
        phase : float, defualt 0.0
            The phase of the main sinusiodal signal.
        phase_degrees : bool, default True
            If ``True``, ``phase`` is in degrees. Otherwise, rad/s is assumed.

        Return
        ------
        y : numpy.ndarray
            The generated signal.
        dydt : numpy.ndarray
            The time derivative of the signal.
        """
        t = np.asarray_chkfinite(t)
        if phase_degrees:
            phase = np.radians(phase)
        return self._y(t, 1.0, phase), self._dydt(t, 1.0, phase)

    def _y(
        self, t: NDArray[np.float64], amp: float, phase: float
    ) -> NDArray[np.float64]:
        """
        Generate a beating signal with a maximum amplitude given by ``amp``.
        """
        y = amp * np.sin(self._f_beat / 2.0 * t) * np.cos(self._f_main * t + phase)
        return y  # type: ignore[no-any-return]

    def _dydt(
        self, t: NDArray[np.float64], amp: float, phase: float
    ) -> NDArray[np.float64]:
        """
        Generate the time derivative of a beating signal with a maximum
        amplitude given by ``amp``.
        """
        dydt = -amp * (self._f_main) * (
            np.sin(self._f_beat / 2.0 * t) * np.sin(self._f_main * t + phase)
        ) + amp * (self._f_beat / 2.0) * (
            np.cos(self._f_beat / 2.0 * t) * np.cos(self._f_main * t + phase)
        )
        return dydt  # type: ignore[no-any-return]


class ChirpSignal:
    """
    Generate an evenly sampled chirp signal with oscillating frequency by
    defining a maximum frequency and a frequency oscillation rate.

    Parameters
    ----------
    f_max : float
        The max frequency the signal ramps up to, before ramping down to zero.
    f_os : float
        The frequency oscillation rate.
    freq_hz : bool, default True.
        If ``True``, ``f_main`` and ``f_beat`` are in Hz. Otherwise,
        rad/s is assumed.

    Notes
    -----
    The signal may be expressed as:

        ```
        phi = 2 * f_max / f_os * sin(f_os * t)
        y = amp * sin(phi + phase)
        ```
    """

    def __init__(self, f_max: float, f_os: float, freq_hz: bool = True) -> None:
        self._f_max = f_max
        self._f_os = f_os

        if freq_hz:
            self._f_max *= 2.0 * np.pi
            self._f_os *= 2.0 * np.pi

    def __call__(
        self,
        t: ArrayLike,
        phase: float = 0.0,
        phase_degrees: float = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Generate a chirp signal with oscillating frequency.

        Parameters
        ----------
        t : array-like
            A time array in seconds.
        phase : float, defualt 0.0
            The phase of the main sinusiodal signal.
        phase_degrees : bool, default True
            If ``True``, ``phase`` is in degrees. Otherwise, rad/s is assumed.

        Return
        ------
        y : numpy.ndarray
            The generated signal.
        dydt : numpy.ndarray
            The time derivative of the signal.

        """
        t = np.asarray_chkfinite(t)
        if phase_degrees:
            phase = np.radians(phase)
        return self._y(t, 1., phase), self._dydt(t, 1., phase)

    def _y(
        self, t: NDArray[np.float64], amp: float, phase: float
    ) -> NDArray[np.float64]:
        """
        Generate a chirp signal with oscillating frequency given an amplitude
        and phase.
        """
        phi = 2.0 * self._f_max / self._f_os * np.sin(self._f_os / 2.0 * t)
        y = amp * np.sin(phi + phase)
        return y  # type: ignore[no-any-return]

    def _dydt(
        self, t: NDArray[np.float64], amp: float, phase: float
    ) -> NDArray[np.float64]:
        """
        Generate the time derivative of a chirp signal with oscillating frequency
        given an amplitude and phase.
        """
        phi = 2.0 * self._f_max / self._f_os * np.sin(self._f_os / 2.0 * t)
        dydt = amp * self._f_max * np.cos(phi + phase) * np.cos(self._f_os / 2.0 * t)
        return dydt  # type: ignore[no-any-return]