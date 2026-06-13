"""
Statistics utilities backed by C++ kernels.

Every routine here takes its data as a NumPy array (a 1D sample or a 2D data
matrix with observations in rows) and does the arithmetic in C++. The results
match NumPy / textbook definitions, which the test suite checks directly.
"""

from collections import namedtuple

import numpy as np

from quantcore._core import stat_mean as _stat_mean
from quantcore._core import stat_variance as _stat_variance
from quantcore._core import stat_std as _stat_std
from quantcore._core import covariance_matrix as _covariance_matrix
from quantcore._core import correlation_matrix as _correlation_matrix
from quantcore._core import linear_regression as _linear_regression
from quantcore._core import mean_confidence_interval as _mean_confidence_interval

RegressionResult = namedtuple("RegressionResult", ["intercept", "coefficients", "r_squared"])
ConfidenceInterval = namedtuple("ConfidenceInterval", ["low", "high"])


def _as_1d(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


def _as_2d(X) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def mean(x) -> float:
    """Sample mean of a 1D array."""
    return _stat_mean(_as_1d(x))


def variance(x, ddof: int = 0) -> float:
    """
    Variance of a 1D array.

    ``ddof`` is the delta degrees of freedom: ``ddof=0`` (default) gives the
    population variance and matches ``numpy.var``; ``ddof=1`` gives the
    unbiased sample variance.
    """
    return _stat_variance(_as_1d(x), ddof)


def std(x, ddof: int = 0) -> float:
    """Standard deviation of a 1D array (see :func:`variance` for ``ddof``)."""
    return _stat_std(_as_1d(x), ddof)


def covariance(X, ddof: int = 1) -> np.ndarray:
    """
    Covariance matrix of ``X``.

    Parameters
    ----------
    X : array_like
        Data matrix of shape ``(n_obs, n_vars)`` — observations in rows,
        variables in columns. A 1D input is treated as a single column.
    ddof : int
        Delta degrees of freedom. ``ddof=1`` (default) matches
        ``numpy.cov(X, rowvar=False)``.

    Returns
    -------
    np.ndarray
        Covariance matrix of shape ``(n_vars, n_vars)``.
    """
    return _covariance_matrix(_as_2d(X), ddof)


def correlation(X) -> np.ndarray:
    """
    Pearson correlation matrix of ``X`` (observations in rows).

    Matches ``numpy.corrcoef(X, rowvar=False)``. Returns an
    ``(n_vars, n_vars)`` matrix with unit diagonal.
    """
    return _correlation_matrix(_as_2d(X))


def linear_regression(X, y, fit_intercept: bool = True) -> RegressionResult:
    """
    Ordinary least squares fit of ``y`` on ``X``.

    Parameters
    ----------
    X : array_like
        Design matrix of shape ``(n_obs, n_features)``. A 1D input is treated
        as a single feature.
    y : array_like
        Response vector of shape ``(n_obs,)``.
    fit_intercept : bool
        Whether to include an intercept term. Default True.

    Returns
    -------
    RegressionResult
        Named tuple ``(intercept, coefficients, r_squared)`` where
        ``coefficients`` is a length-``n_features`` array of slopes and
        ``r_squared`` is the coefficient of determination.
    """
    intercept, coefficients, r_squared = _linear_regression(
        _as_2d(X), _as_1d(y), fit_intercept)
    return RegressionResult(intercept=intercept, coefficients=coefficients,
                            r_squared=r_squared)


def confidence_interval(x, confidence: float = 0.95) -> ConfidenceInterval:
    """
    Two-sided confidence interval for the population mean.

    Uses the normal (large-sample) approximation
    ``mean +/- z * s / sqrt(n)`` where ``z`` is the standard-normal quantile
    for the given confidence level and ``s`` is the sample standard deviation.

    Parameters
    ----------
    x : array_like
        1D sample. Needs at least 2 observations.
    confidence : float
        Confidence level in ``(0, 1)``. Default 0.95.

    Returns
    -------
    ConfidenceInterval
        Named tuple ``(low, high)``.
    """
    low, high = _mean_confidence_interval(_as_1d(x), confidence)
    return ConfidenceInterval(low=low, high=high)
