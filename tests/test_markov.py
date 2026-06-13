"""
Tests for the Markov chain simulator (Phase 7).

Correctness is pinned three ways:
  - structural: shapes, valid state range, reproducibility;
  - deterministic limits: identity and permutation matrices give known paths;
  - statistical: empirical transition frequencies and long-run visit
    frequencies match the transition matrix and its stationary distribution.
"""

import numpy as np
import pytest

from quantcore import markov_chain, stationary_distribution


# A simple, well-mixed two-state chain with a known stationary distribution.
TWO_STATE = np.array([[0.9, 0.1],
                      [0.2, 0.8]])
# pi @ P = pi  =>  pi = [2/3, 1/3]
TWO_STATE_PI = np.array([2 / 3, 1 / 3])


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_markov_shape_and_start():
    chains = markov_chain(TWO_STATE, n_steps=50, n_chains=8, start_state=1, seed=0)
    assert chains.shape == (8, 51)
    np.testing.assert_array_equal(chains[:, 0], 1)


def test_markov_states_in_range():
    chains = markov_chain(TWO_STATE, n_steps=200, n_chains=20, seed=3)
    assert chains.min() >= 0
    assert chains.max() <= 1


def test_markov_reproducible():
    a = markov_chain(TWO_STATE, n_steps=100, n_chains=5, seed=42)
    b = markov_chain(TWO_STATE, n_steps=100, n_chains=5, seed=42)
    np.testing.assert_array_equal(a, b)


def test_markov_integer_dtype():
    chains = markov_chain(TWO_STATE, n_steps=10, n_chains=2, seed=0)
    assert np.issubdtype(chains.dtype, np.integer)


# ---------------------------------------------------------------------------
# Deterministic limits
# ---------------------------------------------------------------------------

def test_identity_matrix_stays_put():
    P = np.eye(3)
    chains = markov_chain(P, n_steps=100, n_chains=4, start_state=2, seed=1)
    assert np.all(chains == 2)


def test_permutation_matrix_cycles():
    # 0 -> 1 -> 2 -> 0 deterministically
    P = np.array([[0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0]])
    chain = markov_chain(P, n_steps=9, n_chains=1, start_state=0, seed=0)[0]
    expected = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    np.testing.assert_array_equal(chain, expected)


def test_absorbing_state_is_absorbing():
    # state 1 absorbs; once entered, never left
    P = np.array([[0.5, 0.5],
                  [0.0, 1.0]])
    chains = markov_chain(P, n_steps=500, n_chains=10, start_state=0, seed=7)
    # for every chain, after the first time it hits state 1 it stays at 1
    for chain in chains:
        hits = np.where(chain == 1)[0]
        if hits.size:
            assert np.all(chain[hits[0]:] == 1)


# ---------------------------------------------------------------------------
# Statistical behaviour
# ---------------------------------------------------------------------------

def test_empirical_transitions_match_matrix():
    chain = markov_chain(TWO_STATE, n_steps=200_000, n_chains=1, seed=123)[0]
    # estimate P[0, 1] from observed transitions out of state 0
    from_0 = chain[:-1] == 0
    to_1_given_0 = (chain[:-1] == 0) & (chain[1:] == 1)
    p01 = to_1_given_0.sum() / from_0.sum()
    assert abs(p01 - 0.1) < 0.01


def test_visit_frequencies_match_stationary():
    chain = markov_chain(TWO_STATE, n_steps=500_000, n_chains=1, seed=999)[0]
    freq0 = np.mean(chain == 0)
    assert abs(freq0 - TWO_STATE_PI[0]) < 0.01


def test_stationary_distribution_matches_known():
    pi = stationary_distribution(TWO_STATE)
    np.testing.assert_allclose(pi, TWO_STATE_PI, atol=1e-9)
    assert pi.sum() == pytest.approx(1.0)


def test_stationary_distribution_three_state_is_valid():
    # A 3-state chain whose Perron eigenvector numpy returns sign-flipped:
    # pi must still come back non-negative, normalised, and fixed by pi @ P.
    P = np.array([[0.70, 0.20, 0.10],
                  [0.30, 0.40, 0.30],
                  [0.20, 0.45, 0.35]])
    pi = stationary_distribution(P)
    assert np.all(pi >= 0.0)
    assert pi.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pi @ P, pi, atol=1e-9)  # stationarity


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_non_square_matrix_raises():
    with pytest.raises((ValueError, RuntimeError)):
        markov_chain(np.array([[0.5, 0.5, 0.0]]), n_steps=10)


def test_rows_not_summing_to_one_raises():
    with pytest.raises((ValueError, RuntimeError)):
        markov_chain(np.array([[0.5, 0.4], [0.2, 0.8]]), n_steps=10)


def test_negative_probability_raises():
    with pytest.raises((ValueError, RuntimeError)):
        markov_chain(np.array([[1.2, -0.2], [0.2, 0.8]]), n_steps=10)


def test_bad_start_state_raises():
    with pytest.raises((ValueError, RuntimeError)):
        markov_chain(TWO_STATE, n_steps=10, start_state=5)
