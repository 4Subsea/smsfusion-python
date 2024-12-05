import numpy as np
from numpy.typing import ArrayLike, NDArray


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
    Provides an interface for generating random sensor noise with the following
    contributions:

    - **N**: White noise
    - **B**: Flicker noise (sometimes referred to as 'pink noise' or 'bias instability')
    - **K**: Drift / Brownian noise (optional)
    - **bc**: Constant bias (optional)

    A white sequence, ``WN[k]``, with spectral density ``N**2`` is generated by drawing samples
    from a Normal distribution with standard deviation ``N * sqrt(fs)``.

    A flicker noise (or pink noise) sequence, ``PN[k]``, is simulated using a first-order
    Gauss-Markov (GM) process with long-term standard deviation ``B`` and correlation
    time ``tau_cb``.

    A Brownian noise (or drift) sequence, ``BN[k]``, is simulated using one of the
    following processes, depending on whether the correlation time, ``tau_ck``,
    is provided or not:

    - Random walk (RW) process (if no correlation time ``tau_ck`` is provided).
    - First-order Gauss-Markov (GM) process, with a long-term  standard deviation
      ``K * sqrt(tau_ck / 2)`` and correlation time ``tau_ck``.

    The total noise is computed as::

        err[k] = bc + WN[k] + PN[k] + BN[k]

    where ``WN[k]``, ``PN[k]``, and ``BN[k]`` represent the white, flicker, and Brownian
    noise contributions, respectively.

    Parameters
    ----------
    N : float
        White noise spectral density coefficient given in units ``V/sqrt(Hz)``,
        where ``V`` represents the unit of the desired output noise.
    B : float
        Bias stability given in the same units as the desired output noise; represents
        the power spectral density coefficient for the flicker noise (or pink noise).
    tau_cb : float
        Correlation time in seconds for the flicker noise (i.e., pink noise).
    K : float, optional
        Drift rate given in units ``V*sqrt(Hz)``, where ``V`` represents the unit
        of the desired output noise; represents the power spectral density coefficient
        of the Brownian noise component. If ``None``, the Brownian noise contribution
        is excluded.
    tau_ck : float, optional
        Correlation time in seconds for the Brownian noise (drift). If ``None``, the
        Brownian noise is modeled as a random walk (RW) process. Otherwise, it
        is modeled as a first-order Gauss-Markov (GM) process.
    bc : float, optional
        Constant bias given in the same units as the output noise.
    seed : int, optional
        A seed used to initialize the random number generator.

    See Also
    --------
    smsfusion.gauss_markov
    smsfusion.random_walk
    smsfusion.white_noise
    """

    def __init__(
        self,
        N: float,
        B: float,
        tau_cb: float,
        K: float | None = None,
        tau_ck: float | None = None,
        bc: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self._N = N
        self._B = B
        self._tau_cb = tau_cb
        self._K = K
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
        if not self._K:
            return np.zeros(n)
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
        )

        if self._K is not None:
            x += self._gen_brown_noise(fs, n, seed=seeds[2])

        return x


class IMUNoise:
    """
    Provides an interface for generating random IMU sensor measurement noise with
    the following contributions for each sensor axis (x, y, z):

    - **N**: White noise
    - **B**: Flicker noise (sometimes referred to as 'pink noise' or 'bias instability')
    - **K**: Drift / Brownian noise (optional)
    - **bc**: Constant bias (optional)

    A white sequence, ``WN[k]``, with spectral density ``N**2`` is generated by drawing samples
    from a Normal distribution with standard deviation ``N * sqrt(fs)``.

    A flicker noise (or pink noise) sequence, ``PN[k]``, is simulated using a first-order
    Gauss-Markov (GM) process with long-term standard deviation ``B`` and correlation
    time ``tau_cb``.

    A Brownian noise (or drift) sequence, ``BN[k]``, is simulated using one of the
    following processes, depending on whether the correlation time, ``tau_ck``,
    is provided or not:

    - Random walk (RW) process (if no correlation time ``tau_ck`` is provided).
    - First-order Gauss-Markov (GM) process, with a long-term  standard deviation
      ``K * sqrt(tau_ck / 2)`` and correlation time ``tau_ck``.

    The total noise is computed as::

        err[k] = bc + WN[k] + PN[k] + BN[k]

    where ``WN[k]``, ``PN[k]``, and ``BN[k]`` represent the white, flicker, and Brownian
    noise contributions, respectively.

    Parameters
    ----------
    err_acc : dict
        Noise parameters for the accelerometer (see Notes). The dictionary values
        can either be scalar (same noise characteristics for all axes) or per-axis
        (list of values).
    err_gyro : dict
        Noise parameters for the gyroscope (see Notes). The dictionary values
        can either be scalar (same noise characteristics for all axes) or per-axis
        (list of values).
    seed : int, optional
        A seed used to initialize a random number generator.

    Notes
    -----
    The input dictionaries must include the following parameters:

    - **N** (required): White noise spectral density coefficient given in units
      ``V/sqrt(Hz)``, where ``V`` represents the unit of the output noise.
    - **B** (required): Bias stability / pink noise power spectral density coefficient
      given in the same units as the output noise.
    - **tau_cb**: Correlation time in seconds for the pink noise (i.e., flicker noise).

    The following parameters are optional and can be omitted or set to `None`:

    - **bc** (optional): Constant bias given in the same units as the desired output noise.
    - **K** (optional): Brownian noise power spectral density coefficient given in units
      ``V*sqrt(Hz)``, where ``V`` represents the unit of the output noise. If ``None``,
      Brownian noise is excluded.
    - **tau_ck** (optional): Correlation time in seconds for the Brownian noise. If ``None``, the
      Brownian noise is modeled as a random walk (RW) process. Otherwise, it
      is modeled as a first-order Gauss-Markov (GM) process.

    The value for each key can be:

    - **Scalar value**: A single value applied to all axes (x, y, z).
    - **Per-axis values**: List of length 3 with values for each axis (x, y, z).

    Examples
    --------

    Full example with different noise characteristics for all axes:

    .. code-block:: python

        err_acc = {
            "bc": [0.1, 0.2, 0.3],
            "N": [1.0e-4, 2.0e-4, 3e-4],
            "B": [1.0e-5, 2.0e-5, 3.0e-5],
            "K": [1.0e-6, 2.0e-6, 3.0e-6],
            "tau_cb": [10.0, 20.0, 30.0],
            "tau_ck": [1_000.0, 2_000.0, 3_000.0],
        }

        err_gyro = {
            "bc": [0.4, 0.5, 0.6],
            "N": [4.0e-4, 5.0e-4, 6e-4],
            "B": [4.0e-5, 5.0e-5, 6.0e-5],
            "K": [4.0e-6, 5.0e-6, 6.0e-6],
            "tau_cb": [40.0, 50.0, 60.0],
            "tau_ck": [4_000.0, 5_000.0, 6_000.0],
        }

        imu_noise = IMUNoise(err_acc, err_gyro)

    Minimal example with the same noise characteristics for all axes, and excluding
    Brownian noise and constant bias:

    .. code-block:: python

        err_acc = {
            "N": 1.0e-4,
            "B": 1.0e-5,
            "tau_cb": 10.0,
        }

        err_gyro = {
            "N": 4.0e-4,
            "B": 4.0e-5,
            "tau_cb": 40.0,
        }

        imu_noise = IMUNoise(err_acc, err_gyro)

    See Also
    --------
    smsfusion.constants.ERR_ACC_MOTION1
    smsfusion.constants.ERR_GYRO_MOTION1
    smsfusion.constants.ERR_ACC_MOTION2
    smsfusion.constants.ERR_GYRO_MOTION2
    smsfusion.NoiseModel : Generates the specific noise for one single sensor axis.

    """

    _REQUIRED_KEYS = {"N", "B", "tau_cb"}
    _ALLOWED_KEYS = _REQUIRED_KEYS | {"K", "tau_ck", "bc"}
    _ERR_DEFAULT = {"K": None, "tau_ck": 5e5, "bc": 0.0}

    def __init__(
        self,
        err_acc,
        err_gyro,
        seed: int | None = None,
    ) -> None:
        self._err_acc = err_acc
        self._err_gyro = err_gyro
        self._seed = seed

        if not self._REQUIRED_KEYS.issubset(self._err_acc.keys()):
            raise ValueError("Missing required keys in accelerometer noise parameters.")
        if not self._REQUIRED_KEYS.issubset(self._err_gyro.keys()):
            raise ValueError("Missing required keys in gyroscope noise parameters.")

        if not self._ALLOWED_KEYS.issuperset(self._err_acc.keys()):
            raise ValueError("Invalid keys in accelerometer noise parameters.")
        if not self._ALLOWED_KEYS.issuperset(self._err_gyro.keys()):
            raise ValueError("Invalid keys in gyroscope noise parameters.")

        self._err_acc = self._ERR_DEFAULT | self._err_acc
        self._err_gyro = self._ERR_DEFAULT | self._err_gyro

        self._err_list = self._to_list(self._err_acc) + self._to_list(self._err_gyro)

        if not len(self._err_list) == 6:
            raise ValueError("Not enough noise parameters provided.")

    @staticmethod
    def _full(value):
        value = np.asarray_chkfinite(value)
        if value.size == 1:
            return np.full(3, value.item())
        elif value.size == 3:
            return value
        else:
            raise ValueError(
                "Parameter values must be a scalar or an array-like of size 3."
            )

    def _to_list(self, dict_of_lists: dict[str, list[float]]) -> list[dict[str, float]]:
        """Convert dict of lists to list of dicts."""
        dict_of_lists = {key: self._full(val) for key, val in dict_of_lists.items()}
        list_of_dicts = [
            {key_i: val_i for key_i, val_i in zip(dict_of_lists.keys(), values_j)}
            for values_j in zip(*dict_of_lists.values())
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
