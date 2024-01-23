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
    numpy.ndarray, shape (n,)
        Sequence of i.i.d. samples.
    """
    return np.random.default_rng(seed).standard_normal(n)


def white_noise(
    N: float, fs: float, n: int, seed: int | None = None
) -> NDArray[np.float64]:
    """
    Generates a discrete-time (bandlimited) Gaussian white noise sequence.

    Bandlimited white noise is characterized by a spectral amplitude which is
    constant over the bandwidth, and zero outside that range. I.e.,::

        S(w) = N ** 2,  for ``|w|`` <= 2*pi*W

        S(w) = 0,       for ``|w|`` > 2*pi*W

    where ``W = fs / 2`` is the bandwidth in hertz, and ``N`` is the spectral
    density coefficient.

    The returned white sequence will thus have a variance of ``N ** 2 * fs``.

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
    numpy.ndarray, shape (n,)
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

    The random walk process is characterized by a power spectrum::

        S(w) = K ** 2 / w ** 2

    where ``K`` is the spectral density coefficient, and ``w`` is the angular
    frequency.

    A discrete-time realization of the process is generated by the recursive
    equation::

        X[k+1] = X[k] + W[k]

    with ``W[k]`` being a zero-mean Gaussian white noise sequence with standard
    deviation::

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
    numpy.ndarray, shape (n,)
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

    The first-order Gauss-Markov process is characterized by a power spectrum::

        S(w) = 2 * sigma**2 * beta / (w**2 + beta**2)

    where ``sigma`` is the long-term standard deviation of the process,
    ``tau_c = 1 / beta`` is the correlation time, and ``w`` is the angular
    frequency.

    A discrete-time realization of the process is generated by the recursive
    equation (see reference [1]_)::

        X[k+1] = exp(-beta * dt) * X[k] + W[k]

    with ``W[k]`` being a zero-mean Gaussian white noise sequence with standard
    deviation::

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
    numpy.ndarray, shape (n,)
        Discrete-time first-order Gauss-Markov sequence.

    References
    ----------
    .. [1] Brown R.G. & Hwang P.Y.C., "Random Signals and Applied Kalman
       Filtering", 4th ed., p. 78-79 and 100, Wiley, 2012.

    See Also
    --------
    smsfusion.random_walk, smsfusion.white_noise
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


def _gen_seeds(seed: int | None, num: int) -> NDArray[np.uint64]:
    """
    Generates a sequence of seeds based on one seed.

    Parameters
    ----------
    seed : int or None
        A seed used to generate a sequence of new seeds.
    num : int
        Number of new seeds to generate.

    Returns
    -------
    seeds : numpy.ndarray, shape (num,)
        Sequence of seeds.
    """
    return np.random.SeedSequence(seed).generate_state(num, "uint64")  # type: ignore[return-value]


class NoiseModel:
    """
    Provides an interface for generating random sensor noise.

    The noise is assumed to have four contributions:

        * Constant bias (optional)
        * White noise:
            Generated by drawing samples from a standard Normal distribution
            with standard deviation ``N * sqrt(fs)``.
        * Flicker noise / pink noise:
            Generated by a first-order Gauss-Markov (GM) process with long-term
            standard deviation ``B`` and correlation time ``tau_cb``.
        * Random walk / Brownian noise:
            Generated by a first-order Gauss-Markov (GM) process with long-term
            standard deviation ``K * sqrt(tau_ck / 2)`` and correlation time
            ``tau_ck``. Or, if no correlation time is provided (default), then
            this noise term is modelled by a random walk (RW) process with
            spectral density coefficient, ``K``.

    The total noise is obtained by adding together the four noise
    contributions. I.e.::

        error(t) = bc + WN(t; N) + GM(t; B, tau_cb) + RW(t; K, tau_ck)

    Or::

        error(t) = bc + WN(t; N) + GM(t; B, tau_cb) + GM(t; K*sqrt(tau_ck/2), tau_ck)

    The GM and RW sequences starts always at 0.

    Parameters
    ----------
    N : float
        White noise spectral density given in units ``V/sqrt(Hz)``, where ``V``
        represents the unit of the output noise.
    B : float
        Bias stability / pink noise power spectral density coefficient given in
        the same units as the output noise.
    K : float
        Brownian noise power spectral density coefficient given in units
        ``V*sqrt(Hz)`` where ``V`` represents the unit of the output noise.
    tau_cb : float
        Correlation time in seconds for the pink noise (i.e., flicker noise).
    tau_ck : float, optional
        Correlation time in seconds for the Brownian noise. If ``None``, the
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
        """
        Generate white noise with spectral density ``N``.
        """
        return white_noise(self._N, fs, n, seed=seed)

    def _gen_pink_noise(
        self, fs: float, n: int, seed: int | None = None
    ) -> NDArray[np.float64]:
        """
        Generate pink noise with spectral density coefficient ``B``.
        """
        return gauss_markov(self._B, self._tau_cb, fs, n, seed=seed)

    def _gen_brown_noise(
        self, fs: float, n: int, seed: int | None = None
    ) -> NDArray[np.float64]:
        """
        Generate Brownian noise with spectral density coefficient ``K``.
        """
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
        numpy.ndarray, shape (n,)
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


class IMUNoise:
    """
    Provides an interface for adding random measurement noise to IMU sensor
    signals.

    Parameters
    ----------
    err_acc : dict of {str : tuple of (float, float, float)}, optional
        Accelerometer noise parameters (see Notes).
    err_gyro : dict of {str : tuple of (float, float, float)}, optional
        Gyroscope noise parameters (see Notes).
    seed : int, optional
        A seed used to initialize a random number generator.

    Notes
    -----
    The noise parameters are given as dictionaries where the key represents the
    parameter name and the value represents the corresponding noise parameter
    values. The values are given as lists with values for the x-, y- and z-axis
    (in that order).

    The noise parameters should include:

        * ``N`` : float
            White noise spectral density given in units ``V/sqrt(Hz)``, where
            ``V`` represents the unit of the output noise.
        * ``B`` : float
            Bias stability / pink noise power spectral density coefficient
            given in the same units as the output noise.
        * ``K`` : float
            Brownian noise power spectral density coefficient given in units
            ``V*sqrt(Hz)`` where ``V`` represents the unit of the output noise.
        * ``tau_cb`` : float
            Correlation time in seconds for the pink noise (i.e., flicker
            noise).
        * ``tau_ck`` : float or None
            Correlation time in seconds for the Brownian noise. If ``None``, the
            Brownian noise is modeled as a random walk (RW) process. Otherwise,
            it is modelled as a first-order Gauss-Markov (GM) process.

    Default accelerometer and gyroscope noise parameters are::

        err_acc = {
            'bc': (0.0, 0.0, 0.0),
            'N': (4.0e-4, 4.0e-4, 4.5e-4),
            'B': (1.5e-4, 1.5e-4, 3.0e-4),
            'K': (4.5e-6, 4.5e-6, 1.5e-5),
            'tau_cb': (50, 50, 30),
            'tau_ck': (5e5, 5e5, 5e5),
        }

        err_gyro = {
            'bc': (0.0, 0.0, 0.0),
            'N': (1.9e-3, 1.9e-3, 1.7e-3),
            'B': (7.5e-4, 4.0e-4, 8.8e-4),
            'K': (2.5e-5, 2.5e-5, 4.0e-5),
            'tau_cb': (50, 50, 50),
            'tau_ck': (5e5, 5e5, 5e5),
        }

    The default parameters represent noise with units m/s^2 for the
    accelerometer, and deg/s for the gyroscope. To generate noise with
    different units, the parameters must be scaled accordingly.

    See Also
    --------
    smsfusion.NoiseModel : Used to generate the specific noise for each sensor
                           signal.
    smsfusion.gauss_markov, smsfusion.random_walk, smsfusion.white_noise
    """

    _DEFAULT_ERR_ACC = {
        "bc": (0.0, 0.0, 0.0),
        "N": (4.0e-4, 4.0e-4, 4.5e-4),
        "B": (1.5e-4, 1.5e-4, 3.0e-4),
        "K": (4.5e-6, 4.5e-6, 1.5e-5),
        "tau_cb": (50, 50, 30),
        "tau_ck": (5e5, 5e5, 5e5),
    }

    _DEFAULT_ERR_GYRO = {
        "bc": (0.0, 0.0, 0.0),
        "N": (1.9e-3, 1.9e-3, 1.7e-3),
        "B": (7.5e-4, 4.0e-4, 8.8e-4),
        "K": (2.5e-5, 2.5e-5, 4.0e-5),
        "tau_cb": (50, 50, 50),
        "tau_ck": (5e5, 5e5, 5e5),
    }

    def __init__(
        self,
        err_acc: dict[str, tuple[float, float, float]] | None = None,
        err_gyro: dict[str, tuple[float, float, float]] | None = None,
        seed: int | None = None,
    ) -> None:
        self._err_acc = err_acc or self._DEFAULT_ERR_ACC
        self._err_gyro = err_gyro or self._DEFAULT_ERR_GYRO
        self._seed = seed

        if set(self._err_acc) != {"bc", "N", "B", "K", "tau_cb", "tau_ck"}:
            raise ValueError("Noise parameter names are not valid.")

        if set(self._err_gyro) != {"bc", "N", "B", "K", "tau_cb", "tau_ck"}:
            raise ValueError("Noise parameter names are not valid.")

        self._err_list = self._to_list(self._err_acc) + self._to_list(self._err_gyro)

        if not len(self._err_list) == 6:
            raise ValueError("Not enough noise parameters provided.")

    @staticmethod
    def _to_list(
        dict_of_lists: dict[str, tuple[float, float, float]]
    ) -> list[dict[str, float]]:
        """Convert dict of lists to list of dicts."""
        if len(set(map(len, dict_of_lists.values()))) != 1:
            raise ValueError("lists must be of same length")

        list_of_dicts = [
            {key: val_i for key, val_i in zip(dict_of_lists.keys(), values)}
            for values in zip(*dict_of_lists.values())
        ]

        return list_of_dicts

    def __call__(self, fs: float, n: int) -> NDArray[np.float64]:
        """
        Generates discrete-time random IMU measurement noise.

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz.
        n : int
            Number of samples to generate.

        Returns
        -------
        numpy.ndarray, shape (n, 6)
            Noise sequences for the IMU sensor signals: Ax, Ay, Az, Gx, Gy, Gz
            (in that order).

        Notes
        -----
        The generated noise will have units corresponding to the noise
        parameters given during initialization; the default noise parameters
        yields accelerometer noise in m/s^2, and gyroscope noise in deg/s.
        """
        seeds = _gen_seeds(self._seed, 6)
        x = np.column_stack(
            [NoiseModel(**self._err_list[i], seed=seeds[i])(fs, n) for i in range(6)]
        )
        return x
