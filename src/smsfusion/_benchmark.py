import numpy as np


class BeatSignal:
    """
    Generate an evenly sampled beating sinusoidal signal by defining a main frequency
    and a beating frequency.

    Parameters
    ----------
    f_main : float
        The main sinusoidal signal frequency.
    f_beat : float
        The beating signal frequency.
    freq_hz : bool, default True.
        If ``True``, ``f_main`` and ``f_beat`` are in Hz. Otherwise,
        rad/s is assumed.

    """
    def __init__(self, f_main, f_beat, freq_hz=True):
        self._f_main = f_main
        self._f_beat = f_beat

        if freq_hz:
            self._f_main *= 2.0 * np.pi
            self._f_beat *= 2.0 * np.pi

    def __call__(self, fs, duration, amp=5.0, phase=0.0, phase_degrees=True):
        """
        Generate a beating signal.

        Parameters
        ----------
        fs : float
            Sampling frequency (in Hz) of the generated signal.
        duration : float
            Duration of the generated signal in seconds.
        amp : float, default 5.0
            The maximum amplitude of the generated signal.
        phase : float, defualt 0.0
            The phase of the main sinusiodal signal.
        phase_degrees : bool, default True
            If ``True``, ``phase`` is in degrees. Otherwise, rad/s is assumed.

        """
        t = np.arange(0.0, duration, 1 / fs)
        if phase_degrees:
            phase = np.radians(phase)
        return t, self._y(t, amp, phase), self._dydt(t, amp, phase)

    def _y(self, t, amp, phase):
        """
        Generate a beating signal with a maximum amplitude given by ``amp``.
        """
        y = amp * np.sin(self._f_beat / 2.0 * t) * np.cos(self._f_main * t + phase)
        return y

    def _dydt(self, t, amp, phase):
        """
        Generate the time derivative of a beating signal with a maximum
        amplitude given by ``amp``.
        """
        dydt = -amp * (self._f_main) * (
            np.sin(self._f_beat / 2.0 * t) * np.sin(self._f_main * t + phase)
        ) + amp * (self._f_beat / 2.0) * (
            np.cos(self._f_beat / 2.0 * t) * np.cos(self._f_main * t + phase)
        )
        return dydt
