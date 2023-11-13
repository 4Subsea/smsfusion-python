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
