import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm, svd


def van_loan(dt, F, G, W):
    """
    Calculate the state transition matrix, ``phi``, and the process noise covariance
    matrix, ``Q``, using the 'Van Loan method' (see ref [1]_.).

    Parameters
    ----------
    F : array-like, shape (N, N)
        State matrix.
    G : array-like, shape (N, M)
        White noise input matrix.
    W : array-like, shape (M, M)
        White noise power spectral density matrix.

    Returns
    -------
    phi : numpy.ndarray, shape (N, N)
        State transition matrix.
    Q : numpy.ndarray, shape (N, N)
        Process noise covariance matrix.

    Notes
    -----
    The Van Loan method assumes that the random process is described by a continuous-time,
    stochastic differential equation (SDE) on the form::

        dx(t)/dt = Fx(t) + Gu(t)

    where ``x(t)`` is the state vector and ``u(t)`` is the white noise input vector with
    power spectral density matrix ``W``.

    Let the discrete-time version of the above SDE be given by::

        x[k+1] = phi * x[k] + w[k]

    where ``phi`` is the state transition matrix and ``w[k]`` is the process noise vector
    with covariance matrix ``Q``.

    The Van Loan method is used to calculate ``phi`` and ``Q`` from the continuous-time
    state-space model.

    References
    ----------
    .. [1] Brown, R.G. and Hwang P.Y.C, "Random Signals and Applied Kalman Filtering
    with MATLAB Exercises", 4th Edition, John Wiley & Sons, 2012.
    """
    F = np.asarray_chkfinite(F)
    G = np.asarray_chkfinite(G)
    W = np.asarray_chkfinite(W)

    n_states = F.shape[0]
    A = np.zeros((2 * n_states, 2 * n_states))
    A[:n_states, :n_states] = -F
    A[:n_states, n_states:] = G @ W @ G.T
    A[n_states:, n_states:] = F.T
    A = dt * A

    B = expm(A)

    phi = B[n_states:, n_states:].T
    Q = phi @ B[:n_states, n_states:]

    return phi, Q


def _standard_normal(size, seed: int | None = None) -> NDArray[np.float64]:
    """
    Draw i.i.d. samples from a standard Normal distribution (mean=0, stdev=1).

    Parameters
    ----------
    size : tuple[int]
        Shape of the output array.
    seed : int, optional
        A seed used to initialize a random number generator.

    Returns
    -------
    numpy.ndarray
        Sequence(s) of i.i.d. samples.
    """
    return np.random.default_rng(seed).standard_normal(size)


class MonteCarlo:
    """
    Provides an interface for doing Monte Carlo simulations of a random process.

    The random process should be described by a continuous-time, stochastic differential
    equation (SDE) on the form::

        dx(t)/dt = Fx(t) + Gu(t)

    where ``x(t)`` is the state vector and ``u(t)`` is the white noise input vector.
    ``F`` and ``G`` are the state matrix and the white noise input matrix, respectively.

    Monte Carlo simulations of the random process is done according to the method
    described by ref [1]_. Specifically, the continuous-time state-space model above is
    discretized using the 'Van Loan method' and then simulated using the discrete-time
    state-space model::

        x[k+1] = phi * x[k] + w[k]

    Parameters
    ----------
    F : array-like, shape (N, N)
        State matrix.
    G : array-like, shape (N, M)
        White noise input matrix.
    W : array-like, shape (M, M)
        White noise power spectral density matrix.

    Note
    ----
    Even though the system matrices describe a continuous-time process, the simulation
    is performed in discrete-time. I.e., the continuous-time state-space model is
    discretized using the 'Van Loan method' (see ref [1]_) and then simulated using the
    discrete-time state-space model::

        x[k+1] = phi * x[k] + w[k]

    References
    ----------
    .. [1] Brown, R.G. and Hwang P.Y.C, "Random Signals and Applied Kalman Filtering
    with MATLAB Exercises", 4th Edition, John Wiley & Sons, 2012.
    """

    def __init__(self, F, G, W):
        self._F = np.asarray_chkfinite(F)
        self._G = np.asarray_chkfinite(G)
        self._W = np.asarray_chkfinite(W)
        self._n_states, self._n_noises = self._G.shape

        if F.shape != (self._n_states, self._n_states):
            raise ValueError()
        if W.shape != (self._n_noises, self._n_noises):
            raise ValueError()

    def simulate(self, x0, fs, n, seed=None):
        """
        Simulate the system.

        The system is discretized using the 'Van Loan method' and then simulated using
        the discrete-time state-space model (see ref [1]_)::

            x[k+1] = phi * x[k] + w[k]

        Parameters
        ----------
        x0 : array-like, shape (N,)
            Initial state vector.
        fs : float
            Sampling frequency in hertz.
        n : int
            Number of samples to generate.
        seed : int, optional
            A seed used to initialize a random number generator.

        References
        ----------
        .. [1] Brown, R.G. and Hwang P.Y.C, "Random Signals and Applied Kalman Filtering
        with MATLAB Exercises", 4th Edition, John Wiley & Sons, 2012.
        """
        x0 = np.asarray_chkfinite(x0).reshape(self._n_states)
        dt = 1.0 / fs

        # Discretize the system using the van Loan method
        phi, Q = van_loan(dt, self._F, self._G, self._W)

        # Find C such that w[k] = Cu[k] where u[k] are independent samples form a
        # standard normal population
        U, T, _ = svd(Q, full_matrices=True)
        S = np.sqrt(T)
        C = U @ S

        # Simulate
        x = np.zeros((n, self._n_states))
        x[0, :] = x0
        u = _standard_normal(size=(n - 1, self._n_states), seed=seed)
        for i in range(1, n):
            x[i, :] = phi @ x[i - 1, :] + C * u[i - 1, :]

        return x


class GaussMarkov(MonteCarlo):
    """
    Provides an interface for doing Monte Carlo simulations of a first-order
    Gauss-Markov (GM) process.

    The first-order Gauss-Markov process is described by a continuous-time, stochastic
    differential equation (SDE) on the form::

        dx(t)/dt = -beta * x(t) + w(t)

    where ``x(t)`` is the state, and ``w(t)`` is the white noise input with spectral
    density ``sqrt(2.0 * sigma**2 * beta)``. Furthermore, ``beta = 1.0 / tau_c``
    determines the correlation time (i.e., ``tau_c``).

    Monte Carlo simulations of the random process is done according to the method
    described by ref [1]_. Specifically, the continuous-time state-space model above is
    discretized using the 'Van Loan method' and then simulated using the discrete-time
    state-space model::

        x[k+1] = phi * x[k] + w[k]

    Parameters
    ----------
    sigma_2 : float
        Mean-square value (i.e., variance) of the process.
    tau_c : float
        Correlation time in seconds.
    fs : float
        Sampling frequency in Hz.
    n : int
        Number of samples to generate.

    References
    ----------
    .. [1] Brown R.G. & Hwang P.Y.C., "Random Signals and Applied Kalman
       Filtering", 4th ed., p. 78-79 and 100, Wiley, 2012.
    """

    def __init__(self, sigma_2, tau_c):
        beta = 1.0 / tau_c
        F = np.array([[-beta]])
        G = np.array([[np.sqrt(2.0 * sigma_2 * beta)]])
        W = np.array([[1.0]])
        self._sigma_2 = sigma_2
        self._beta = 1.0 / tau_c
        super().__init__(F, G, W)


class RandomWalk(MonteCarlo):
    """
    Provides an interface for doing Monte Carlo simulations of a random walk (RW)
    process.

    The random walk process is described by a continuous-time, stochastic differential
    equation (SDE) on the form::

        dx(t)/dt = w(t)

    where ``x(t)`` is the state, and ``w(t)`` is the white noise input with spectral
    density ``K**2``.

    Monte Carlo simulations of the random process is done according to the method
    described by ref [1]_. Specifically, the continuous-time state-space model above is
    discretized using the 'Van Loan method' and then simulated using the discrete-time
    state-space model::

        x[k+1] = phi * x[k] + w[k]

    Parameters
    ----------
    K : float
        Spectral density coefficient.
    fs : float
        Sampling frequency in Hz.
    n : int
        Number of samples to generate.

    References
    ----------
    .. [1] Brown R.G. & Hwang P.Y.C., "Random Signals and Applied Kalman
       Filtering", 4th ed., p. 78-79 and 100, Wiley, 2012.
    """

    def __init__(self, K):
        F = np.array([[0.0]])
        G = np.array([[1.0]])
        W = np.array([[K**2]])
        super().__init__(F, G, W)
