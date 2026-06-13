#include "gbm.hpp"
#include "rng.hpp"
#include "utils.hpp"
#include "parallel.hpp"

#include <random>
#include <cmath>
#include <cstdint>
#include <vector>
#include <thread>

namespace {

// One thread's batch of GBM paths: rows [start, end), private RNG stream.
// drift and vol are precomputed once per worker since they are loop-invariant.
void gbm_paths_worker(
    std::vector<std::vector<double>>& result,
    int start, int end,
    int n_steps, double drift, double vol, double s0,
    uint64_t seed, int thread_id)
{
    auto rng = make_thread_rng(seed, thread_id);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int i = start; i < end; ++i) {
        result[i][0] = s0;
        for (int j = 1; j <= n_steps; ++j) {
            result[i][j] = result[i][j - 1] * std::exp(drift + vol * dist(rng));
        }
    }
}

}  // namespace

std::vector<std::vector<double>> simulate_gbm_paths(
    int n_steps, int n_paths, double dt, double mu, double sigma,
    double s0, uint64_t seed)
{
    validate_gbm_params(n_steps, n_paths, dt, sigma, s0);

    auto rng = make_seeded_rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    const double drift = (mu - 0.5 * sigma * sigma) * dt;
    const double vol   = sigma * std::sqrt(dt);

    std::vector<std::vector<double>> result(n_paths, std::vector<double>(n_steps + 1));
    for (int i = 0; i < n_paths; ++i) {
        result[i][0] = s0;
        for (int j = 1; j <= n_steps; ++j) {
            result[i][j] = result[i][j - 1] * std::exp(drift + vol * dist(rng));
        }
    }
    return result;
}

std::vector<std::vector<double>> simulate_gbm_paths_parallel(
    int n_steps, int n_paths, double dt, double mu, double sigma,
    double s0, uint64_t seed, int n_threads)
{
    validate_gbm_params(n_steps, n_paths, dt, sigma, s0);
    n_threads = resolve_n_threads(n_threads, n_paths);

    const double drift = (mu - 0.5 * sigma * sigma) * dt;
    const double vol   = sigma * std::sqrt(dt);

    std::vector<std::vector<double>> result(n_paths, std::vector<double>(n_steps + 1));
    const int chunk = n_paths / n_threads;

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        const int start = t * chunk;
        const int end   = (t == n_threads - 1) ? n_paths : start + chunk;
        threads.emplace_back(gbm_paths_worker, std::ref(result), start, end,
                             n_steps, drift, vol, s0, seed, t);
    }
    for (auto& th : threads) th.join();
    return result;
}

std::vector<double> simulate_gbm_terminal(
    int n_paths, double T, double mu, double sigma, double s0, uint64_t seed)
{
    require_gt0_int(n_paths, "n_paths");
    require_gt0(T, "T");
    require_ge0(sigma, "sigma");
    require_gt0(s0, "s0");

    auto rng = make_seeded_rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    const double drift = (mu - 0.5 * sigma * sigma) * T;
    const double vol   = sigma * std::sqrt(T);

    std::vector<double> out(n_paths);
    for (int i = 0; i < n_paths; ++i) {
        out[i] = s0 * std::exp(drift + vol * dist(rng));
    }
    return out;
}
