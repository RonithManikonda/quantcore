import pytest
import numpy as np

import quantcore
from quantcore.stochastic import brownian_paths, brownian_increments


# ---------------------------------------------------------------------------
# Import and public API
# ---------------------------------------------------------------------------

def test_module_imports():
    assert hasattr(quantcore, "brownian_paths")
    assert hasattr(quantcore, "brownian_increments")


def test_version_string():
    assert isinstance(quantcore.__version__, str)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_paths_shape_basic():
    paths = brownian_paths(n_steps=100, n_paths=10, dt=0.01, sigma=1.0, seed=42)
    assert paths.shape == (10, 101)  # (n_paths, n_steps + 1)


def test_increments_shape_basic():
    inc = brownian_increments(n_steps=100, n_paths=10, dt=0.01, sigma=1.0, seed=42)
    assert inc.shape == (10, 100)  # (n_paths, n_steps)


def test_single_path_shape():
    paths = brownian_paths(n_steps=50, n_paths=1, dt=0.01, seed=0)
    assert paths.shape == (1, 51)


def test_single_step_shape():
    paths = brownian_paths(n_steps=1, n_paths=5, dt=1.0, seed=0)
    assert paths.shape == (5, 2)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_paths_returns_numpy_float64():
    result = brownian_paths(n_steps=10, n_paths=2, dt=0.1, seed=0)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_increments_returns_numpy_float64():
    result = brownian_increments(n_steps=10, n_paths=2, dt=0.1, seed=0)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# Initial condition: W(0) = 0
# ---------------------------------------------------------------------------

def test_paths_start_at_zero():
    paths = brownian_paths(n_steps=200, n_paths=20, dt=0.01, seed=99)
    np.testing.assert_array_equal(paths[:, 0], 0.0)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_same_paths():
    a = brownian_paths(n_steps=100, n_paths=10, dt=0.01, seed=7)
    b = brownian_paths(n_steps=100, n_paths=10, dt=0.01, seed=7)
    np.testing.assert_array_equal(a, b)


def test_same_seed_same_increments():
    a = brownian_increments(n_steps=100, n_paths=10, dt=0.01, seed=7)
    b = brownian_increments(n_steps=100, n_paths=10, dt=0.01, seed=7)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_different_paths():
    a = brownian_paths(n_steps=100, n_paths=10, dt=0.01, seed=1)
    b = brownian_paths(n_steps=100, n_paths=10, dt=0.01, seed=2)
    assert not np.array_equal(a, b)


def test_paths_and_increments_consistent():
    # cumsum of increments (prepended with 0) must equal the full path
    seed = 42
    paths = brownian_paths(n_steps=50, n_paths=5, dt=0.01, seed=seed)
    inc   = brownian_increments(n_steps=50, n_paths=5, dt=0.01, seed=seed)
    reconstructed = np.concatenate([np.zeros((5, 1)), np.cumsum(inc, axis=1)], axis=1)
    np.testing.assert_allclose(paths, reconstructed, atol=1e-12)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_negative_sigma_raises():
    with pytest.raises((ValueError, RuntimeError)):
        brownian_paths(n_steps=10, n_paths=1, dt=0.01, sigma=-0.1, seed=0)


def test_zero_dt_raises():
    with pytest.raises((ValueError, RuntimeError)):
        brownian_paths(n_steps=10, n_paths=1, dt=0.0, sigma=1.0, seed=0)


def test_negative_dt_raises():
    with pytest.raises((ValueError, RuntimeError)):
        brownian_paths(n_steps=10, n_paths=1, dt=-0.01, sigma=1.0, seed=0)


def test_zero_steps_raises():
    with pytest.raises((ValueError, RuntimeError)):
        brownian_paths(n_steps=0, n_paths=1, dt=0.01, sigma=1.0, seed=0)


def test_negative_steps_raises():
    with pytest.raises((ValueError, RuntimeError)):
        brownian_paths(n_steps=-5, n_paths=1, dt=0.01, sigma=1.0, seed=0)


def test_zero_paths_raises():
    with pytest.raises((ValueError, RuntimeError)):
        brownian_paths(n_steps=10, n_paths=0, dt=0.01, sigma=1.0, seed=0)


def test_negative_paths_raises():
    with pytest.raises((ValueError, RuntimeError)):
        brownian_paths(n_steps=10, n_paths=-1, dt=0.01, sigma=1.0, seed=0)


# ---------------------------------------------------------------------------
# Edge case: zero volatility
# ---------------------------------------------------------------------------

def test_zero_sigma_paths_stay_at_zero():
    paths = brownian_paths(n_steps=100, n_paths=10, dt=0.01, sigma=0.0, seed=0)
    np.testing.assert_array_equal(paths, 0.0)


def test_zero_sigma_increments_are_zero():
    inc = brownian_increments(n_steps=100, n_paths=10, dt=0.01, sigma=0.0, seed=0)
    np.testing.assert_array_equal(inc, 0.0)


# ---------------------------------------------------------------------------
# Statistical sanity checks
# (large samples, so tolerances are intentionally wide)
# ---------------------------------------------------------------------------

def test_increments_mean_near_zero():
    # E[dW] = 0
    inc = brownian_increments(n_steps=1000, n_paths=500, dt=0.01, sigma=1.0, seed=42)
    assert abs(inc.mean()) < 0.05


def test_increment_variance_matches_theory():
    # Var(dW) = sigma^2 * dt
    sigma, dt = 2.0, 0.05
    inc = brownian_increments(n_steps=500, n_paths=1000, dt=dt, sigma=sigma, seed=42)
    expected = sigma ** 2 * dt
    assert abs(inc.var() - expected) < 0.05


def test_path_variance_grows_linearly_with_time():
    # Var(W(t)) = sigma^2 * t  where  t = k * dt
    sigma, dt = 1.0, 0.01
    n_paths = 5000
    paths = brownian_paths(n_steps=100, n_paths=n_paths, dt=dt, sigma=sigma, seed=42)
    for k in [10, 50, 100]:
        measured = paths[:, k].var()
        expected = sigma ** 2 * k * dt
        assert abs(measured - expected) < 0.15, (
            f"At step {k}: measured={measured:.4f}, expected={expected:.4f}"
        )


def test_increments_are_uncorrelated():
    # dW_k and dW_j should be independent for k != j
    inc = brownian_increments(n_steps=200, n_paths=2000, dt=0.01, sigma=1.0, seed=0)
    # correlation between step 0 and step 100 should be near zero
    corr = np.corrcoef(inc[:, 0], inc[:, 100])[0, 1]
    assert abs(corr) < 0.1
