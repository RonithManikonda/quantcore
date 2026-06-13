"""
Discrete-time, finite-state Markov chain simulation.

The simulator runs in C++ and takes a row-stochastic transition matrix as
input. ``stationary_distribution`` is a small NumPy-based companion that solves
for the long-run state probabilities analytically, which the simulated visit
frequencies should converge to.
"""

import numpy as np

from quantcore._core import simulate_markov as _simulate_markov


def markov_chain(
    P,
    n_steps: int,
    n_chains: int = 1,
    start_state: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """
    Simulate discrete-time Markov chains over a finite state space.

    Parameters
    ----------
    P : array_like
        Row-stochastic transition matrix of shape ``(n_states, n_states)``.
        ``P[i, j]`` is the probability of moving to state ``j`` from state ``i``.
        Must be square, non-negative, with each row summing to 1.
    n_steps : int
        Number of transitions to simulate. Must be > 0.
    n_chains : int
        Number of independent chains. Must be > 0. Default 1.
    start_state : int
        Initial state index, in ``[0, n_states)``. Default 0.
    seed : int
        RNG seed. Same seed + arguments reproduce the output. Default 0.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_chains, n_steps + 1)``. Each row is one
        chain's state trajectory; column 0 is always ``start_state``.
    """
    P = np.ascontiguousarray(P, dtype=np.float64)
    return _simulate_markov(P, n_steps, n_chains, start_state, seed)


def stationary_distribution(P) -> np.ndarray:
    """
    Compute the stationary distribution of a Markov chain.

    Solves ``pi @ P = pi`` with ``sum(pi) = 1`` by taking the left eigenvector
    of ``P`` associated with eigenvalue 1. For an irreducible, aperiodic chain
    this is the unique long-run distribution that ``markov_chain`` visit
    frequencies converge to.

    Parameters
    ----------
    P : array_like
        Row-stochastic transition matrix of shape ``(n_states, n_states)``.

    Returns
    -------
    np.ndarray
        The stationary probability vector of shape ``(n_states,)``, normalised
        to sum to 1 and clipped to be non-negative.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square 2D matrix")

    # Left eigenvectors of P are the (right) eigenvectors of P.T.
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    pi = np.real(eigvecs[:, idx])

    # Eigenvectors are only defined up to a scale, so numpy may hand back the
    # all-negative version. Normalise by the sum first (which fixes the sign),
    # then clip away any tiny negative numerical noise and renormalise.
    total = pi.sum()
    if abs(total) < 1e-12:
        raise ValueError("could not find a valid stationary distribution")
    pi = pi / total
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()
