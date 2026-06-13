#include "pricing.hpp"
#include "rng.hpp"
#include "utils.hpp"
#include "special.hpp"

#include <random>
#include <cmath>
#include <cstdint>
#include <algorithm>  // std::max
#include <utility>

namespace {

// Undiscounted European payoff at expiry.
inline double payoff(double sT, double K, bool is_call) {
    return is_call ? std::max(sT - K, 0.0) : std::max(K - sT, 0.0);
}

}  // namespace

double black_scholes_price(double s0, double K, double r, double sigma,
                           double T, bool is_call)
{
    require_gt0(s0, "s0");
    require_gt0(K, "K");
    require_ge0(sigma, "sigma");
    require_gt0(T, "T");

    const double disc = std::exp(-r * T);

    // Degenerate zero-volatility case: the payoff is deterministic, priced off
    // the forward. Handling it explicitly avoids a divide-by-zero in d1/d2.
    if (sigma == 0.0) {
        const double fwd = s0 * std::exp(r * T);
        return disc * payoff(fwd, K, is_call);
    }

    const double srt = sigma * std::sqrt(T);
    const double d1 = (std::log(s0 / K) + (r + 0.5 * sigma * sigma) * T) / srt;
    const double d2 = d1 - srt;

    if (is_call)
        return s0 * norm_cdf(d1) - K * disc * norm_cdf(d2);
    return K * disc * norm_cdf(-d2) - s0 * norm_cdf(-d1);
}

std::pair<double, double> mc_european_price(
    double s0, double K, double r, double sigma, double T,
    int n_paths, bool is_call, uint64_t seed, bool antithetic)
{
    require_gt0(s0, "s0");
    require_gt0(K, "K");
    require_ge0(sigma, "sigma");
    require_gt0(T, "T");
    require_gt0_int(n_paths, "n_paths");

    auto rng = make_seeded_rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    const double drift = (r - 0.5 * sigma * sigma) * T;
    const double vol   = sigma * std::sqrt(T);
    const double disc  = std::exp(-r * T);

    // Accumulate per-sample payoffs. With antithetic variates the independent
    // statistical unit is the *pair average*, so the standard error is taken
    // over n_pairs units rather than over the 2*n_pairs raw evaluations.
    double sum = 0.0, sum_sq = 0.0;
    long n_units;

    if (antithetic) {
        n_units = std::max(n_paths / 2, 1);
        for (long i = 0; i < n_units; ++i) {
            const double z = dist(rng);
            const double s_up = s0 * std::exp(drift + vol * z);
            const double s_dn = s0 * std::exp(drift - vol * z);
            const double y = 0.5 * (payoff(s_up, K, is_call) +
                                    payoff(s_dn, K, is_call));
            sum += y;
            sum_sq += y * y;
        }
    } else {
        n_units = n_paths;
        for (long i = 0; i < n_units; ++i) {
            const double z = dist(rng);
            const double sT = s0 * std::exp(drift + vol * z);
            const double y = payoff(sT, K, is_call);
            sum += y;
            sum_sq += y * y;
        }
    }

    const double mean = sum / static_cast<double>(n_units);
    const double sample_var = (n_units > 1)
        ? (sum_sq - static_cast<double>(n_units) * mean * mean) / (n_units - 1)
        : 0.0;

    const double price = disc * mean;
    const double std_error = disc * std::sqrt(sample_var / static_cast<double>(n_units));
    return {price, std_error};
}
