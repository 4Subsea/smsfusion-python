import numpy as np
from numpy.typing import NDArray


def _standard_normal(n: int, seed: int | None = None) -> NDArray[np.float64]:
    """
    Draw i.i.d. samples from a standard Normal distribution (mean=0, stdev=1).

    Parameters
    ----------
    n : int
        Number of samples to generate.
    seed : int, optional
        A seed used to initialize a random number generator.

    Returns
    -------
    x : numpy.ndarray
        Sequence of i.i.d. samples.
    """

    return np.random.default_rng(seed).standard_normal(n)


def white_noise(
    N: float, fs: float, n: int, seed: int | None = None
) -> NDArray[np.float64]:
    """
    Generates a discrete-time (bandlimited) Gaussian white noise sequence.

    Bandlimited white noise is characterized by a spectral amplitude which is
    constant over the bandwidth, and zero outside that range. I.e.,:

        S(w) = N ** 2,  for ``|w|`` <= 2*pi*W

        S(w) = 0,       for ``|w|`` > 2*pi*W

    where `W = fs / 2` is the bandwidth in hertz, and `N` is the spectral
    density coefficient.

    The returned white sequence will thus have a variance of `N ** 2 * fs`.

    Parameters
    ----------
    N : float
        Spectral density coefficient.
    fs : float
        Sampling frequency in Hz.
    n : int
        Number of samples to generate.

    Returns
    -------
    x : numpy.ndarray
        Discrete time Gaussian white noise sequence.

    See Also
    --------
    smsfusion.gauss_markov, smsfusion.random_walk
    """

    sigma_wn = N * np.sqrt(fs)

    return sigma_wn * _standard_normal(n, seed=seed)  # type: ignore[no-any-return]


def random_walk(
    K: float, fs: float, n: int, seed: int | None = None
) -> NDArray[np.float64]:
    """
    Generates a discrete-time random walk (i.e., Brownian noise) sequence.

    The random walk process is characterized by a power spectrum:

        S(w) = K ** 2 / w ** 2

    where `K` is the spectral density coefficient, and `w` is the angular
    frequency.

    A discrete-time realization of the process is generated by the recursive
    equation:

        X[k+1] = X[k] + W[k]

    with `W[k]` being a zero-mean Gaussian white noise sequence with standard
    deviation:

        sigma_wn = K / sqrt(fs)

    The sequence starts always at 0.

    Parameters
    ----------
    K : float
        Spectral density coefficient.
    fs : float
        Sampling frequency in Hz.
    n : int
        Number of samples to generate.

    Return
    ------
    x : numpy.ndarray
        Discrete-time random walk sequence.

    See Also
    --------
    smsfusion.gauss_markov, smsfusion.white_noise
    """

    sigma_wn = K / np.sqrt(fs)

    x = np.zeros(n)
    epsilon = _standard_normal(n - 1, seed=seed)
    for i in range(1, n):
        x[i] = x[i - 1] + sigma_wn * epsilon[i - 1]

    return x


def gauss_markov(
    sigma: float, tau_c: float, fs: float, n: int, seed: int | None = None
) -> NDArray[np.float64]:
    """
    Generates a discrete-time first-order Gauss-Markov sequence.

    The first-order Gauss-Markov process is characterized by a power spectrum:

        S(w) = 2 * sigma**2 * beta / (w**2 + beta**2)

    where `sigma` is the long-term standard deviation of the process,
    `tau_c = 1 / beta` is the correlation time, and `w` is the angular
    frequency.

    A discrete-time realization of the process is generated by the recursive
    equation (see reference [1]_):

        X[k+1] = exp(-beta * dt) * X[k] + W[k]

    with `W[k]` being a zero-mean Gaussian white noise sequence with standard
    deviation:

        sigma_wn = sigma * sqrt(1 - exp(-2 * beta * dt))

    The sequence starts always at 0.

    Parameters
    ----------
    sigma : float
        Standard deviation (i.e., root-mean-square value) of the process.
    tau_c : float
        Correlation time in seconds.
    fs : float
        Sampling frequency in Hz.
    n : int
        Number of samples to generate.

    Return
    ------
    x : numpy.ndarray
        Discrete-time first-order Gauss-Markov sequence.

    See Also
    --------
    smsfusion.random_walk, smsfusion.white_noise

    References
    ----------
    .. [1] Brown R.G. & Hwang P.Y.C. (2012) "Random Signals and Applied Kalman
       Filtering". (4th ed., p. 78, 79 and 100). Wiley.
    """

    dt = 1.0 / fs
    beta = 1.0 / tau_c

    phi = np.exp(-beta * dt)
    sigma_wn = sigma * np.sqrt(1.0 - np.exp(-2 * beta * dt))

    x = np.zeros(n)
    epsilon = _standard_normal(n - 1, seed=seed)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + sigma_wn * epsilon[i - 1]

    return x


def _gen_seeds(seed: int, num: int | None) -> NDArray[np.uint64]:
    """
    Generates a list of seeds based on one seed.

    Parameters
    ----------
    seed : int or None
        A seed used to generate a list of new seeds.
    num : int
        Number of new seeds to generate.

    Returns
    -------
    seeds : numpy.ndarray
        List of seeds.
    """
    return np.random.SeedSequence(seed).generate_state(num, "uint64")


class NoiseModel:
    """
    Provides an interface for generating random sensor noise.

    The noise is assumed to have four contributions:
        * Constant bias (optional)
        * White noise:
            Generated by drawing samples from a standard Normal distribution
            with standard deviation `N * sqrt(fs)`.
        * Flicker noise / pink noise:
            Generated by a first-order Gauss-Markov (GM) process with long-term
            standard deviation `B` and correlation time `tau_cb`.
        * Random walk / Brownian noise:
            Generated by a first-order Gauss-Markov (GM) process with long-term
            standard deviation `K * sqrt(tau_ck / 2)` and correlation time
            `tau_ck`. Or, if no correlation time is provided (default), then
            this noise term is modelled by a random walk (RW) process with
            spectral density coefficient, `K`.

    The total noise is obtained by adding together the four noise
    contributions. I.e.:

        error(t) = bc + WN(t; N) + GM(t; B, tau_cb) + RW(t; K, tau_ck)

    Or,

        error(t) = bc + WN(t; N) + GM(t; B, tau_cb) + GM(t; K*sqrt(tau_ck/2), tau_ck)

    The GM and RW sequences starts always at 0.

    Parameters
    ----------
    N : float
        White noise spectral density given in units `V/sqrt(Hz)`, where `V`
        represents the unit of the output noise.
    B : float
        Bias stability / pink noise power spectral density coefficient given in
        the same units as the output noise.
    K : float
        Brownian noise power spectral density coefficient given in units
        `V*sqrt(Hz)` where `V` represents the unit of the output noise.
    tau_cb : float
        Correlation time in seconds for the pink noise (i.e., flicker noise).
    tau_ck : float, optional
        Correlation time in seconds for the Brownian noise. If `None`, the
        Brownian noise is modeled as a random walk (RW) process. Otherwise, it
        is modelled as a first-order Gauss-Markov (GM) process.
    bc : float, optional
        Constant bias given in the same units as the output noise.
    seed : int, optional
        A seed used to initialize the random number generator.

    See Also
    --------
    smsfusion.gauss_markov, smsfusion.random_walk, smsfusion.white_noise
    """

    def __init__(
        self,
        N: float,
        B: float,
        K: float,
        tau_cb: float,
        tau_ck: float | None = None,
        bc: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self._N = N
        self._B = B
        self._K = K
        self._tau_cb = tau_cb
        self._tau_ck = tau_ck
        self._bc = bc
        self._seed = seed

    def _gen_white_noise(
        self, fs: float, n: int, seed: int | None = None
    ) -> NDArray[np.float64]:
        """Generate white noise with spectral density `N`."""
        return white_noise(self._N, fs, n, seed=seed)

    def _gen_pink_noise(
        self, fs: float, n: int, seed: int | None = None
    ) -> NDArray[np.float64]:
        """Generate pink noise with spectral density coefficient `B`"""
        return gauss_markov(self._B, self._tau_cb, fs, n, seed=seed)

    def _gen_brown_noise(
        self, fs: float, n: int, seed: int | None = None
    ) -> NDArray[np.float64]:
        """Generate Brownian noise with spectral density coefficient `K`"""
        if self._tau_ck:
            sigma = self._K * np.sqrt(self._tau_ck / 2.0)
            return gauss_markov(sigma, self._tau_ck, fs, n, seed=seed)
        else:
            return random_walk(self._K, fs, n, seed=seed)

    def __call__(self, fs: float, n: int) -> NDArray[np.float64]:
        """
        Generates a discrete-time random noise sequence.

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz.
        n : int
            Number of samples to generate.

        Returns
        -------
        x : numpy.ndarray
            Noise sequence.

        Notes
        -----
        The generated noise will have units corresponding to the noise
        parameters given during initialization.
        """
        seeds = _gen_seeds(self._seed, 3)
        x = (
            self._bc
            + self._gen_white_noise(fs, n, seed=seeds[0])
            + self._gen_pink_noise(fs, n, seed=seeds[1])
            + self._gen_brown_noise(fs, n, seed=seeds[2])
        )
        return x
