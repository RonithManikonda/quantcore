#pragma once

#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// Geometric Brownian Motion (GBM)
//
//   dS = mu * S dt + sigma * S dW
//
// The exact solution is used for discretisation (no Euler-Maruyama bias):
//
//   S_{k+1} = S_k * exp( (mu - 0.5 sigma^2) dt + sigma sqrt(dt) Z ),  Z ~ N(0,1)
//
// This is the standard model for asset prices under Black-Scholes assumptions.
// ---------------------------------------------------------------------------

// Simulate n_paths GBM price paths, each with n_steps time steps.
//
// Parameters:
//   n_steps - number of time steps per path (> 0)
//   n_paths - number of independent paths (> 0)
//   dt      - length of each time step (> 0)
//   mu      - drift (annualised); may be negative
//   sigma   - volatility (>= 0)
//   s0      - initial price (> 0)
//   seed    - RNG seed; same args + seed reproduce the output exactly
//
// Returns: [n_paths][n_steps + 1]. Every row starts at s0.
std::vector<std::vector<double>> simulate_gbm_paths(
    int n_steps, int n_paths, double dt, double mu, double sigma,
    double s0, uint64_t seed);

// Parallel version of simulate_gbm_paths. Same output contract; the result is
// deterministic for a given (seed, n_threads). n_threads = 0 auto-detects.
std::vector<std::vector<double>> simulate_gbm_paths_parallel(
    int n_steps, int n_paths, double dt, double mu, double sigma,
    double s0, uint64_t seed, int n_threads = 0);

// Simulate only the terminal values S(T), drawn directly from the closed-form
// lognormal law rather than stepping through time. Memory-light (one value per
// path) and the natural input for Monte Carlo option pricing.
//
// Returns: a length-n_paths vector of S(T) samples.
std::vector<double> simulate_gbm_terminal(
    int n_paths, double T, double mu, double sigma, double s0, uint64_t seed);
