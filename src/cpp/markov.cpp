#include "markov.hpp"
#include "rng.hpp"
#include "utils.hpp"

#include <random>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>  // std::upper_bound
#include <stdexcept>
#include <string>

namespace {

// Validate that P is square, non-negative, and row-stochastic. Returns the
// number of states. A small tolerance on the row sums absorbs the rounding a
// user gets from, e.g., dividing counts by a total.
int validate_transition_matrix(const std::vector<std::vector<double>>& P) {
    if (P.empty())
        throw std::invalid_argument("P must have at least one state");

    const std::size_t n = P.size();
    constexpr double tol = 1e-9;

    for (std::size_t i = 0; i < n; ++i) {
        if (P[i].size() != n)
            throw std::invalid_argument("P must be square (n_states x n_states)");

        double row_sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            if (P[i][j] < 0.0)
                throw std::invalid_argument(
                    "P has a negative entry; probabilities must be >= 0");
            row_sum += P[i][j];
        }
        if (std::abs(row_sum - 1.0) > tol)
            throw std::invalid_argument(
                "each row of P must sum to 1 (row " + std::to_string(i) +
                " sums to " + std::to_string(row_sum) + ")");
    }
    return static_cast<int>(n);
}

}  // namespace

std::vector<std::vector<int>> simulate_markov(
    const std::vector<std::vector<double>>& P,
    int n_steps, int n_chains, int start_state, uint64_t seed)
{
    require_gt0_int(n_steps, "n_steps");
    require_gt0_int(n_chains, "n_chains");

    const int n_states = validate_transition_matrix(P);
    if (start_state < 0 || start_state >= n_states)
        throw std::invalid_argument("start_state must be in [0, n_states)");

    // Precompute each row's cumulative distribution once so sampling a
    // transition is a single binary search. The last entry is pinned to 1.0 so
    // a uniform draw can never fall past the end due to rounding.
    std::vector<std::vector<double>> cdf(n_states, std::vector<double>(n_states));
    for (int i = 0; i < n_states; ++i) {
        double acc = 0.0;
        for (int j = 0; j < n_states; ++j) {
            acc += P[i][j];
            cdf[i][j] = acc;
        }
        cdf[i][n_states - 1] = 1.0;
    }

    auto rng = make_seeded_rng(seed);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    std::vector<std::vector<int>> result(n_chains, std::vector<int>(n_steps + 1));
    for (int c = 0; c < n_chains; ++c) {
        int state = start_state;
        result[c][0] = state;
        for (int t = 1; t <= n_steps; ++t) {
            const double u = unif(rng);
            const std::vector<double>& row = cdf[state];
            // first state whose cumulative probability exceeds u
            auto it = std::upper_bound(row.begin(), row.end(), u);
            state = static_cast<int>(it - row.begin());
            if (state >= n_states) state = n_states - 1;  // numerical safety
            result[c][t] = state;
        }
    }
    return result;
}
