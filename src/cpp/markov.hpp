#pragma once

#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// Discrete-time, finite-state Markov chain simulation.
//
// Given a row-stochastic transition matrix P (P[i][j] = P(next = j | now = i)),
// simulate independent state trajectories. This is the library's first routine
// that takes a matrix as *input* from Python rather than only returning arrays.
// ---------------------------------------------------------------------------

// Simulate n_chains independent Markov chains for n_steps transitions each.
//
//   P            - transition matrix [n_states][n_states]; square, entries >= 0,
//                  each row summing to 1 (validated, small tolerance allowed)
//   n_steps      - number of transitions (> 0)
//   n_chains     - number of independent chains (> 0)
//   start_state  - initial state index, in [0, n_states)
//   seed         - RNG seed; same args + seed reproduce the output exactly
//
// Returns: [n_chains][n_steps + 1] of state indices. Column 0 is start_state.
std::vector<std::vector<int>> simulate_markov(
    const std::vector<std::vector<double>>& P,
    int n_steps, int n_chains, int start_state, uint64_t seed);
