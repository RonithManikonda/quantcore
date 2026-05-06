#include "brownian.hpp"
#include "rng.hpp"
#include "utils.hpp"

// Standard library headers you will need:
#include <random>    // std::normal_distribution<double>, std::mt19937_64
#include <cmath>     // std::sqrt
#include <cstdint>   // uint64_t
#include <vector>
#include <stdexcept> // std::invalid_argument (thrown by validate_brownian_params)

// ---------------------------------------------------------------------------
// simulate_paths
// ---------------------------------------------------------------------------
// TODO: implement this function.
//
// Steps:
//   1. Call validate_brownian_params(n_steps, n_paths, dt, sigma).
//   2. Create a seeded RNG with make_seeded_rng(seed) from rng.hpp.
//   3. Build a std::normal_distribution<double> with mean=0 and
//      stddev = sigma * std::sqrt(dt).  Each draw is one increment dW.
//   4. Allocate result as a [n_paths][n_steps+1] vector-of-vectors.
//   5. For each path i:
//        result[i][0] = 0.0          (Brownian motion starts at zero)
//        for k = 1 .. n_steps:
//            result[i][k] = result[i][k-1] + dist(rng)
//   6. Return result.
//
// Hint: draw ALL increments from the same rng object in a consistent order
// so the output is fully determined by the seed.

std::vector<std::vector<double>> simulate_paths(
    int n_steps, int n_paths, double dt, double sigma, uint64_t seed)
{
    // TODO: implement
    return {};
}

// ---------------------------------------------------------------------------
// simulate_increments
// ---------------------------------------------------------------------------
// TODO: implement this function.
//
// Same setup as simulate_paths, but instead of building cumulative sums you
// store the raw draws dW ~ N(0, sigma^2 * dt).
//
// Returns a [n_paths][n_steps] array.
//
// Hint: you can share the same validate + rng + distribution setup as above.
// The draw order should match what simulate_paths uses for steps 1..n_steps
// so that: simulate_increments(...).cumsum() + 0 == simulate_paths(...)[1:].

std::vector<std::vector<double>> simulate_increments(
    int n_steps, int n_paths, double dt, double sigma, uint64_t seed)
{
    // TODO: implement
    return {};
}
