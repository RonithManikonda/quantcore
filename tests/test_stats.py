"""
Tests for the statistics utilities (Phase 7).

Every routine is validated against NumPy's own implementation (or a textbook
formula) on the same inputs, so the C++ kernels are checked against a trusted
reference rather than against themselves.
"""

import numpy as np
import pytest

from quantcore import (
    mean,
    variance,
    std,
    covariance,
    correlation,
    linear_regression,
    confidence_interval,
)


# ---------------------------------------------------------------------------
# Scalar reductions vs NumPy
# ---------------------------------------------------------------------------

def test_mean_matches_numpy():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.5, -2.0])
    assert mean(x) == pytest.approx(np.mean(x))


@pytest.mark.parametrize("ddof", [0, 1])
def test_variance_matches_numpy(ddof):
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    assert variance(x, ddof=ddof) == pytest.approx(np.var(x, ddof=ddof))


@pytest.mark.parametrize("ddof", [0, 1])
def test_std_matches_numpy(ddof):
    rng = np.random.default_rng(1)
    x = rng.normal(size=1000)
    assert std(x, ddof=ddof) == pytest.approx(np.std(x, ddof=ddof))


def test_variance_requires_enough_data():
    with pytest.raises((ValueError, RuntimeError)):
        variance([1.0], ddof=1)


# ---------------------------------------------------------------------------
# Covariance / correlation vs NumPy
# ---------------------------------------------------------------------------

def test_covariance_matches_numpy():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 3))
    np.testing.assert_allclose(covariance(X, ddof=1),
                               np.cov(X, rowvar=False, ddof=1), atol=1e-10)


def test_correlation_matches_numpy():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 4))
    np.testing.assert_allclose(correlation(X),
                               np.corrcoef(X, rowvar=False), atol=1e-10)


def test_correlation_diagonal_is_one():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 3))
    np.testing.assert_allclose(np.diag(correlation(X)), 1.0, atol=1e-12)


def test_correlation_of_perfectly_correlated_columns():
    x = np.linspace(0, 1, 100)
    X = np.column_stack([x, 2 * x + 3])  # perfectly linearly related
    corr = correlation(X)
    assert corr[0, 1] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Linear regression vs numpy.linalg.lstsq
# ---------------------------------------------------------------------------

def test_regression_recovers_known_coefficients():
    rng = np.random.default_rng(5)
    n = 2000
    X = rng.normal(size=(n, 2))
    true_intercept, true_beta = 1.5, np.array([2.0, -3.0])
    y = true_intercept + X @ true_beta + rng.normal(scale=0.01, size=n)

    res = linear_regression(X, y, fit_intercept=True)
    assert res.intercept == pytest.approx(true_intercept, abs=0.01)
    np.testing.assert_allclose(res.coefficients, true_beta, atol=0.01)
    assert res.r_squared > 0.999


def test_regression_matches_lstsq():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(300, 3))
    y = rng.normal(size=300)

    res = linear_regression(X, y, fit_intercept=True)

    # reference fit with an explicit intercept column
    design = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    np.testing.assert_allclose(res.intercept, beta[0], atol=1e-8)
    np.testing.assert_allclose(res.coefficients, beta[1:], atol=1e-8)


def test_regression_no_intercept():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 2))
    y = X @ np.array([1.0, 2.0]) + rng.normal(scale=0.01, size=300)

    res = linear_regression(X, y, fit_intercept=False)
    assert res.intercept == 0.0
    np.testing.assert_allclose(res.coefficients, [1.0, 2.0], atol=0.01)


def test_regression_single_feature_1d_input():
    x = np.linspace(0, 10, 200)
    y = 3.0 + 2.0 * x
    res = linear_regression(x, y, fit_intercept=True)
    assert res.intercept == pytest.approx(3.0, abs=1e-8)
    assert res.coefficients[0] == pytest.approx(2.0, abs=1e-8)
    assert res.r_squared == pytest.approx(1.0, abs=1e-12)


def test_regression_collinear_raises():
    x = np.linspace(0, 1, 50)
    X = np.column_stack([x, x])  # exactly collinear -> singular normal equations
    y = rng = np.random.default_rng(8).normal(size=50)
    with pytest.raises((ValueError, RuntimeError)):
        linear_regression(X, y, fit_intercept=True)


# ---------------------------------------------------------------------------
# Confidence interval
# ---------------------------------------------------------------------------

def test_confidence_interval_formula():
    rng = np.random.default_rng(9)
    x = rng.normal(loc=5.0, scale=2.0, size=1000)

    ci = confidence_interval(x, confidence=0.95)
    m = np.mean(x)
    se = np.std(x, ddof=1) / np.sqrt(len(x))
    z = 1.959963984540054  # standard normal 0.975 quantile
    assert ci.low == pytest.approx(m - z * se, abs=1e-6)
    assert ci.high == pytest.approx(m + z * se, abs=1e-6)
    assert ci.low < m < ci.high


def test_confidence_interval_contains_true_mean_usually():
    # 95% CI should contain the true mean in the vast majority of repetitions
    true_mean = 0.0
    hits = 0
    trials = 200
    for s in range(trials):
        x = np.random.default_rng(s).normal(loc=true_mean, scale=1.0, size=400)
        ci = confidence_interval(x, 0.95)
        if ci.low <= true_mean <= ci.high:
            hits += 1
    assert hits >= int(0.90 * trials)  # comfortably above, well below 100%


def test_confidence_interval_bad_level_raises():
    with pytest.raises((ValueError, RuntimeError)):
        confidence_interval([1.0, 2.0, 3.0], confidence=1.5)
