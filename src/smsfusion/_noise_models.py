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
    x : numpy.ndarray, shape (n,)
        Sequence of i.i.d. samples.
    """

    return np.random.default_rng(seed).standard_normal(n)


def white_noise(
    N: float, fs: float, n: int, seed: int | None = None
) -> NDArray[np.float64]:
    """
    Generates a (bandlimited) Gaussian white noise sequence.

    Bandlimited white noise is described by a spectral amplitude which is
    constant over the bandwidth, and zero outside that range. I.e.,:

        ``S(f) = N ** 2,  for |f| <= W``

        ``S(f) = 0,       for |f| > W``

    where `W = fs / 2` is the bandwidth in hertz, and N is the spectral density
    coefficient.

    The returned white sequence will thus have a variance of:

        ``Var = N ** 2 * fs``

    Parameters
    ----------
    N : float
        Spectral density coefficient.
    fs : float
        Sampling frequency in Hz.
    n : int
        Number of samples to generate.

    Return
    ------
    x : numpy.ndarray, shape (n,)
        Discrete time white noise sequence.
    """

    sigma_wn = N * np.sqrt(fs)

    return sigma_wn * _standard_normal(n, seed=seed)


def random_walk(K, fs, n, seed=None):
    """
    Generates a discrete time random walk (i.e., Brown noise) sequence. The
    sequence starts always at 0.

    The generated signal will have a power spectrum,

        S(f) = K ** 2 / (2*pi*f) ** 2

    where K is the spectral density coefficient.

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
    array :
        Discrete time random walk sequence.
    """

    sigma_k = K / np.sqrt(fs)

    x = np.zeros(n)
    epsilon = _standard_normal(n - 1, seed=seed)
    for i in range(1, n):
        x[i] = x[i - 1] + sigma_k * epsilon[i - 1]

    return x


def gauss_markov(G, tau_c, fs, n, seed=None):
    """
    Generates a discrete time first-order Gauss-Markov sequence. The sequence
    starts always at 0.

    Implemented according to sensor_4s Theory Manual.

    Parameters
    ----------
    G : float
        Spectral density coefficient of the driving white noise.
    tau_c : float
        Correlation time in seconds.
    fs : float
        Sampling frequency in Hz.
    n : int
        Number of samples to generate.
    """

    sigma_g = G / np.sqrt(fs)

    x = np.zeros(n)
    epsilon = _standard_normal(n - 1, seed=seed)
    phi = 1.0 - 1.0 / (tau_c * fs)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + sigma_g * epsilon[i - 1]

    return x