"""
Tests for geometric Brownian motion (Phase 7).

GBM uses the exact lognormal solution, so two things must hold tightly:
  - paths are strictly positive and start at s0;
  - the terminal distribution matches the closed-form lognormal moments,
    E[S_T] = s0 * exp(mu * T) and median[S_T] = s0 * exp((mu - sigma^2/2) * T).
"""

import numpy as np
import pytest

from quantcore import gbm_paths, gbm_paths_parallel, gbm_terminal


# ---------------------------------------------------------------------------
# Shape, dtype, starting condition
# ---------------------------------------------------------------------------

def test_gbm_paths_shape_and_start():
    p = gbm_paths(n_steps=252, n_paths=10, dt=1 / 252, mu=0.05, sigma=0.2,
                  s0=100.0, seed=42)
    assert p.shape == (10, 253)
    assert p.dtype == np.float64
    np.testing.assert_array_equal(p[:, 0], 100.0)


def test_gbm_paths_strictly_positive():
    p = gbm_paths(n_steps=500, n_paths=200, dt=0.01, mu=-0.5, sigma=0.8,
                  s0=50.0, seed=1)
    assert np.all(p > 0.0)


def test_gbm_terminal_shape():
    s = gbm_terminal(n_paths=1000, T=1.0, mu=0.05, sigma=0.2, s0=100.0, seed=0)
    assert s.shape == (1000,)
    assert np.all(s > 0.0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_gbm_same_seed_reproducible():
    a = gbm_paths(n_steps=100, n_paths=10, dt=0.01, mu=0.1, sigma=0.3, s0=1.0, seed=7)
    b = gbm_paths(n_steps=100, n_paths=10, dt=0.01, mu=0.1, sigma=0.3, s0=1.0, seed=7)
    np.testing.assert_array_equal(a, b)


def test_gbm_parallel_single_thread_matches_serial():
    serial = gbm_paths(n_steps=120, n_paths=40, dt=0.01, mu=0.07, sigma=0.25,
                       s0=20.0, seed=11)
    par = gbm_paths_parallel(n_steps=120, n_paths=40, dt=0.01, mu=0.07,
                             sigma=0.25, s0=20.0, seed=11, n_threads=1)
    np.testing.assert_array_equal(serial, par)


# ---------------------------------------------------------------------------
# Zero-volatility limit: deterministic exponential growth
# ---------------------------------------------------------------------------

def test_gbm_zero_sigma_is_deterministic_growth():
    mu, T, s0, n_steps = 0.1, 1.0, 100.0, 252
    p = gbm_paths(n_steps=n_steps, n_paths=5, dt=T / n_steps, mu=mu,
                  sigma=0.0, s0=s0, seed=0)
    # every path identical and equal to s0 * exp(mu * t)
    expected_terminal = s0 * np.exp(mu * T)
    np.testing.assert_allclose(p[:, -1], expected_terminal, rtol=1e-10)
    np.testing.assert_allclose(p[0], p[1], rtol=1e-12)


# ---------------------------------------------------------------------------
# Closed-form lognormal moments
# ---------------------------------------------------------------------------

def test_gbm_terminal_mean_matches_theory():
    mu, sigma, T, s0 = 0.05, 0.2, 1.0, 100.0
    s = gbm_terminal(n_paths=100_000, T=T, mu=mu, sigma=sigma, s0=s0, seed=42)
    expected = s0 * np.exp(mu * T)            # E[S_T]
    np.testing.assert_allclose(s.mean(), expected, rtol=0.01)


def test_gbm_terminal_median_matches_theory():
    mu, sigma, T, s0 = 0.05, 0.2, 1.0, 100.0
    s = gbm_terminal(n_paths=100_000, T=T, mu=mu, sigma=sigma, s0=s0, seed=42)
    expected_median = s0 * np.exp((mu - 0.5 * sigma ** 2) * T)
    np.testing.assert_allclose(np.median(s), expected_median, rtol=0.02)


def test_gbm_paths_terminal_matches_direct_terminal_in_distribution():
    # full-path final column and the direct terminal draw share the same law
    mu, sigma, s0, n_steps = 0.05, 0.2, 100.0, 252
    dt, T = 1 / 252, 1.0
    p = gbm_paths(n_steps=n_steps, n_paths=100_000, dt=dt, mu=mu, sigma=sigma,
                  s0=s0, seed=1)
    s = gbm_terminal(n_paths=100_000, T=T, mu=mu, sigma=sigma, s0=s0, seed=2)
    np.testing.assert_allclose(p[:, -1].mean(), s.mean(), rtol=0.01)
    np.testing.assert_allclose(p[:, -1].std(), s.std(), rtol=0.03)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("s0", [0.0, -10.0])
def test_gbm_nonpositive_s0_raises(s0):
    with pytest.raises((ValueError, RuntimeError)):
        gbm_paths(n_steps=10, n_paths=1, dt=0.01, sigma=0.2, s0=s0, seed=0)


def test_gbm_negative_sigma_raises():
    with pytest.raises((ValueError, RuntimeError)):
        gbm_paths(n_steps=10, n_paths=1, dt=0.01, sigma=-0.2, s0=1.0, seed=0)


def test_gbm_terminal_bad_T_raises():
    with pytest.raises((ValueError, RuntimeError)):
        gbm_terminal(n_paths=10, T=0.0, sigma=0.2, s0=1.0, seed=0)
