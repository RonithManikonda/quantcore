#pragma once

#include <cstdint>
#include <utility>  // std::pair

// ---------------------------------------------------------------------------
// European option pricing
//
// Two independent routes to the same number, which is the point: the Monte
// Carlo estimate should converge to the closed-form Black-Scholes price, and
// the gap should sit inside a couple of Monte Carlo standard errors.
// ---------------------------------------------------------------------------

// Closed-form Black-Scholes price of a European option.
//
//   s0      - spot price (> 0)
//   K       - strike (> 0)
//   r       - risk-free rate (continuously compounded; may be negative)
//   sigma   - volatility (>= 0)
//   T       - time to expiry (> 0)
//   is_call - true for a call, false for a put
double black_scholes_price(double s0, double K, double r, double sigma,
                           double T, bool is_call);

// Monte Carlo price of a European option under the risk-neutral measure
// (drift = r), simulating terminal prices from the GBM lognormal law.
//
// Returns {price, standard_error}. The standard error is the Monte Carlo
// sampling error of the price estimate and shrinks like 1/sqrt(n).
//
//   n_paths    - number of simulated terminal prices (> 0)
//   antithetic - if true, use antithetic variates (pairs Z, -Z) for variance
//                reduction; n_paths is split into n_paths/2 independent pairs
std::pair<double, double> mc_european_price(
    double s0, double K, double r, double sigma, double T,
    int n_paths, bool is_call, uint64_t seed, bool antithetic);
