import numpy as np
from numpy.typing import ArrayLike, NDArray


def allan_var(
    y: ArrayLike, fs: float, num: int = 100, progress: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Estimate the Allan variance of a signal using the overlapping Allan variance method.

    Parameters
    ----------
    y : array-like, shape (N, M)
        Sequence(s) of timeseries signal with shape (N, M), where N is
        the number of samples and M is the number of signals. See Notes regarding
        required signal lengths.
    fs : float
        Sampling frequency in Hz.
    num : int, default 100
        Number of averaging times to estimate. Default is 100. Note that if the signal
        is not long enough, fewer averaging times will be returned.
    progress: bool, optional
        Whether to show a progress bar while calculating. This requires :py:mod:`tqdm`
        to be present.

    Return
    ------
    tau : ndarray, shape (<=num,)
        Averaging times. The number of elements will be less than or equal to ``num``.
    avar : ndarray, shape (<=num, M)
        Allan variance estimate(s). The number of elements will be less than or equal to ``num``.

    Note
    ----
    The Allan variance is estimated for avaraging times up to ``tau_0 * (N - 1) / 32``,
    where ``tau_0`` is the sampling period of the data and N is the number of samples.
    For instance, if you sample at 10 Hz and want to estimate the Allan variance for
    averaging times up to e.g. 1 hour, then you need at least ``N = 3600 * 32 * 10 + 1``
    samples.
    """
    if progress:
        try:
            from tqdm import trange as _range
        except ImportError:
            raise ImportError("tqdm is required for showing a progress bar.") from None
    else:
        _range = range

    y = np.asarray_chkfinite(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    N, M = y.shape

    m_max = (N - 1) / 32
    m = np.unique(np.logspace(0, np.log10(m_max), num, dtype=np.int64))

    tau_0 = 1.0 / fs
    tau = m * tau_0

    x = np.cumsum(y, axis=0) / fs
    avar = np.zeros((len(tau), M))
    for i in _range(len(m)):
        idx = np.arange(N - 2 * m[i])
        tmp = (x[idx + 2 * m[i], :] - 2 * x[idx + m[i], :] + x[idx, :]) ** 2
        avar[i, :] = np.sum(tmp, axis=0) / (2 * tau[i] ** 2.0 * (N - 2 * m[i]))
    return tau, avar
