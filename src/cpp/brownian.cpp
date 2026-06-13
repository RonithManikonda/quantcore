#include "brownian.hpp"
#include "rng.hpp"
#include "utils.hpp"
#include "parallel.hpp"  // resolve_n_threads

#include <random>    // std::normal_distribution, std::mt19937_64
#include <cmath>     // std::sqrt
#include <cstdint>   // uint64_t
#include <vector>
#include <thread>    // std::thread

std::vector<std::vector<double>> simulate_paths(
    int n_steps, int n_paths, double dt, double sigma, uint64_t seed)
{
    validate_brownian_params(n_steps, n_paths, dt, sigma);

    auto rng = make_seeded_rng(seed);

    std::normal_distribution<double> dist(0.0, sigma * std::sqrt(dt));

    std::vector<std::vector<double>> result(n_paths, std::vector<double>(n_steps + 1));

    for (int i = 0; i < n_paths; i++) {
        result[i][0] = 0.0;

        for (int j = 1; j <= n_steps; j++) {
            result[i][j] = result[i][j - 1] + dist(rng);
        }
    }

    return result;
}

std::vector<std::vector<double>> simulate_increments(
    int n_steps, int n_paths, double dt, double sigma, uint64_t seed)
{
    validate_brownian_params(n_steps, n_paths, dt, sigma);

    auto rng = make_seeded_rng(seed);

    std::normal_distribution<double> dist(0.0, sigma * std::sqrt(dt));

    std::vector<std::vector<double>> result(n_paths, std::vector<double>(n_steps));

    for (int i = 0; i < n_paths; i++) {
        for (int j = 0; j < n_steps; j++) {
            result[i][j] = dist(rng);
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Worker functions (one per parallel variant)
// Each worker owns a private RNG derived from (seed, thread_id) and writes
// only into rows [start, end) of result — never outside that range.
// ---------------------------------------------------------------------------

// Worker for simulate_paths_parallel.
//
// Parameters:
//   result    - the full output array, allocated by the caller before any
//               thread is launched; this thread writes rows [start, end) only
//   start     - first path index this thread is responsible for (inclusive)
//   end       - one past the last path index (exclusive), i.e. [start, end)
//   n_steps   - number of time steps per path
//   dt        - time step size
//   sigma     - diffusion coefficient
//   seed      - base seed passed in by the caller
//   thread_id - index of this thread (0, 1, 2, ...); used to derive a unique
//               RNG via make_thread_rng(seed, thread_id) from rng.hpp
//
// TODO: implement
//   1. Call make_thread_rng(seed, thread_id) to get a private RNG.
//   2. Construct a std::normal_distribution<double> with stddev = sigma * sqrt(dt).
//   3. Loop i from start to end (exclusive).
//        Set result[i][0] = 0.0.
//        Loop j from 1 to n_steps (inclusive): result[i][j] = result[i][j-1] + dist(rng).
void paths_worker(
    std::vector<std::vector<double>>& result,
    int start, int end,
    int n_steps, double dt, double sigma,
    uint64_t seed, int thread_id)
{
    auto rng = make_thread_rng(seed, thread_id);
    std::normal_distribution<double> dist(0.0, sigma * std::sqrt(dt));

    for (int i = start; i < end; ++i) {
        result[i][0] = 0.0;
        for (int j = 1; j <= n_steps; ++j) {
            result[i][j] = result[i][j - 1] + dist(rng);
        }
    }
}

// Worker for simulate_increments_parallel.
//
// Same setup as paths_worker, but stores raw increments instead of
// cumulative sums — no result[i][0] = 0 initialisation, no j-1 lookback.
//
// TODO: implement
//   1. Call make_thread_rng(seed, thread_id) to get a private RNG.
//   2. Construct a std::normal_distribution<double> with stddev = sigma * sqrt(dt).
//   3. Loop i from start to end (exclusive).
//        Loop j from 0 to n_steps (exclusive): result[i][j] = dist(rng).
void increments_worker(
    std::vector<std::vector<double>>& result,
    int start, int end,
    int n_steps, double dt, double sigma,
    uint64_t seed, int thread_id)
{
    auto rng = make_thread_rng(seed, thread_id);
    std::normal_distribution<double> dist(0.0, sigma * std::sqrt(dt));

    for (int i = start; i < end; ++i) {
        for (int j = 0; j < n_steps; ++j) {
            result[i][j] = dist(rng);
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel entry points
// ---------------------------------------------------------------------------

std::vector<std::vector<double>> simulate_paths_parallel(
    int n_steps, int n_paths, double dt, double sigma, uint64_t seed,
    int n_threads)
{
    validate_brownian_params(n_steps, n_paths, dt, sigma);

    n_threads = resolve_n_threads(n_threads, n_paths);

    std::vector<std::vector<double>> result(n_paths, std::vector<double>(n_steps + 1));

    const int chunk = n_paths / n_threads;

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        const int start = t * chunk;
        const int end   = (t == n_threads - 1) ? n_paths : start + chunk;
        threads.emplace_back(paths_worker, std::ref(result), start, end,
                             n_steps, dt, sigma, seed, t);
    }
    for (auto& th : threads) th.join();

    return result;
}

std::vector<std::vector<double>> simulate_increments_parallel(
    int n_steps, int n_paths, double dt, double sigma, uint64_t seed,
    int n_threads)
{
    validate_brownian_params(n_steps, n_paths, dt, sigma);

    n_threads = resolve_n_threads(n_threads, n_paths);

    std::vector<std::vector<double>> result(n_paths, std::vector<double>(n_steps));

    const int chunk = n_paths / n_threads;

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        const int start = t * chunk;
        const int end   = (t == n_threads - 1) ? n_paths : start + chunk;
        threads.emplace_back(increments_worker, std::ref(result), start, end,
                             n_steps, dt, sigma, seed, t);
    }
    for (auto& th : threads) th.join();

    return result;
}
