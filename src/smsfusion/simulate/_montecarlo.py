import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm


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


def _standard_normal(size=None, seed: int | None = None) -> NDArray[np.float64]:
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
    return np.random.default_rng(seed).standard_normal(size)


class MonteCarlo:
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
        x0 = np.asarray_chkfinite(x0).reshape(self._n_states)
        dt = 1.0 / fs

        # Find discrete system matrices using the van Loan method
        phi, Q = van_loan(dt, self._F, self._G, self._W)

        # Find C such that w = Cu where u are independent samples from a N(0, 1) population
        U, T, Vh = np.linalg.svd(Q, full_matrices=True)
        S = np.sqrt(T)
        C = U @ S

        # Simulate
        x = np.zeros((n, self._n_states))
        x[0, :] = x0
        u = _standard_normal(size=(n - 1, self._n_states), seed=seed)
        for i in range(1, n):
            x[i, :] = phi @ x[i - 1, :] + C * u[i - 1, :]

        return x


def _standard_normal(size=None, seed: int | None = None) -> NDArray[np.float64]:
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
    return np.random.default_rng(seed).standard_normal(size)


def van_loan(dt, F, G, W):
    """
    Calculate the state transition matrix, ``phi``, and the process noise covariance
    matrix, ``Q``, using the 'Van Loan method'.
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


class MonteCarlo:
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
        x0 = np.asarray_chkfinite(x0).reshape(self._n_states)
        dt = 1.0 / fs

        # Find discrete system matrices using the van Loan method
        phi, Q = van_loan(dt, self._F, self._G, self._W)

        # Find C such that w = Cu where u are independent samples from a N(0, 1) population
        U, T, Vh = np.linalg.svd(Q, full_matrices=True)
        S = np.sqrt(T)
        C = U @ S

        # Simulate
        x = np.zeros((n, self._n_states))
        x[0, :] = x0
        u = _standard_normal(size=(n - 1, self._n_states), seed=seed)
        for i in range(1, n):
            x[i, :] = phi @ x[i - 1, :] + C * u[i - 1, :]

        return x
