"""
Tests for the multithreaded Brownian motion variants (Phase 6).

The parallel functions split work across path batches, each thread owning a
private RNG stream and a disjoint slice of the output. Two contracts matter:

  1. Correctness — same shapes, same starting condition, and the same
     statistical behaviour as the single-threaded version.
  2. Determinism — output is fully determined by (seed, n_threads). In
     particular, n_threads=1 must reproduce the single-threaded result
     exactly, because thread 0 derives the same RNG as make_seeded_rng(seed).
"""

import numpy as np
import pytest

from quantcore import (
    brownian_paths,
    brownian_increments,
    brownian_paths_parallel,
    brownian_increments_parallel,
)


# ---------------------------------------------------------------------------
# Shape and starting condition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_threads", [1, 2, 4, 8])
def test_parallel_paths_shape(n_threads):
    paths = brownian_paths_parallel(n_steps=100, n_paths=64, dt=0.01,
                                    seed=42, n_threads=n_threads)
    assert paths.shape == (64, 101)
    np.testing.assert_array_equal(paths[:, 0], 0.0)


@pytest.mark.parametrize("n_threads", [1, 2, 4, 8])
def test_parallel_increments_shape(n_threads):
    inc = brownian_increments_parallel(n_steps=100, n_paths=64, dt=0.01,
                                       seed=42, n_threads=n_threads)
    assert inc.shape == (64, 100)


def test_auto_thread_count_runs():
    # n_threads=0 -> auto-detect; should still produce the right shape
    paths = brownian_paths_parallel(n_steps=50, n_paths=200, dt=0.01, seed=1)
    assert paths.shape == (200, 51)


def test_more_threads_than_paths_is_safe():
    # n_threads capped at n_paths internally — must not crash or misshape
    paths = brownian_paths_parallel(n_steps=10, n_paths=3, dt=0.01,
                                    seed=0, n_threads=64)
    assert paths.shape == (3, 11)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_single_thread_matches_serial_paths():
    # n_threads=1 must reproduce the single-threaded result bit-for-bit
    serial = brownian_paths(n_steps=120, n_paths=40, dt=0.01, sigma=0.7, seed=11)
    par    = brownian_paths_parallel(n_steps=120, n_paths=40, dt=0.01,
                                     sigma=0.7, seed=11, n_threads=1)
    np.testing.assert_array_equal(serial, par)


def test_single_thread_matches_serial_increments():
    serial = brownian_increments(n_steps=120, n_paths=40, dt=0.01, seed=11)
    par    = brownian_increments_parallel(n_steps=120, n_paths=40, dt=0.01,
                                          seed=11, n_threads=1)
    np.testing.assert_array_equal(serial, par)


@pytest.mark.parametrize("n_threads", [2, 4])
def test_same_seed_same_threads_reproducible(n_threads):
    a = brownian_paths_parallel(n_steps=80, n_paths=50, dt=0.01,
                                seed=5, n_threads=n_threads)
    b = brownian_paths_parallel(n_steps=80, n_paths=50, dt=0.01,
                                seed=5, n_threads=n_threads)
    np.testing.assert_array_equal(a, b)


def test_parallel_increments_cumsum_consistent():
    seed, nt = 42, 4
    paths = brownian_paths_parallel(n_steps=50, n_paths=20, dt=0.01,
                                    seed=seed, n_threads=nt)
    inc   = brownian_increments_parallel(n_steps=50, n_paths=20, dt=0.01,
                                         seed=seed, n_threads=nt)
    reconstructed = np.concatenate(
        [np.zeros((20, 1)), np.cumsum(inc, axis=1)], axis=1)
    np.testing.assert_allclose(paths, reconstructed, atol=1e-12)


# ---------------------------------------------------------------------------
# Statistical sanity (large samples, wide tolerances)
# ---------------------------------------------------------------------------

def test_parallel_variance_grows_linearly():
    sigma, dt = 1.0, 0.01
    paths = brownian_paths_parallel(n_steps=100, n_paths=5000, dt=dt,
                                    sigma=sigma, seed=42, n_threads=4)
    for k in [10, 50, 100]:
        measured = paths[:, k].var()
        expected = sigma ** 2 * k * dt
        assert abs(measured - expected) < 0.15


# ---------------------------------------------------------------------------
# Validation propagates through the parallel path too
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(n_steps=10, n_paths=1, dt=0.01, sigma=-1.0),
    dict(n_steps=10, n_paths=1, dt=0.0, sigma=1.0),
    dict(n_steps=0, n_paths=1, dt=0.01, sigma=1.0),
    dict(n_steps=10, n_paths=0, dt=0.01, sigma=1.0),
])
def test_parallel_validation_raises(kwargs):
    with pytest.raises((ValueError, RuntimeError)):
        brownian_paths_parallel(seed=0, n_threads=4, **kwargs)
