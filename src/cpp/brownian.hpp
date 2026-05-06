#pragma once

#include <vector>
#include <cstdint>

// Simulate n_paths independent Brownian motion paths, each with n_steps time steps.
//
// Parameters:
//   n_steps  - number of time steps per path (must be > 0)
//   n_paths  - number of independent paths (must be > 0)
//   dt       - length of each time step (must be > 0)
//   sigma    - diffusion coefficient / volatility (must be >= 0)
//   seed     - RNG seed; same seed + same args always produces the same output
//
// Returns: [n_paths][n_steps + 1]
//   Each row is one path. W[i][0] = 0 for all i.
//   W[i][k] = W[i][k-1] + dW  where  dW ~ N(0, sigma^2 * dt)
std::vector<std::vector<double>> simulate_paths(
    int n_steps, int n_paths, double dt, double sigma, uint64_t seed);

// Simulate the raw Brownian increments dW ~ N(0, sigma^2 * dt).
//
// Returns: [n_paths][n_steps]
//   Each entry is one independent Gaussian increment.
//   To reconstruct a full path: cumsum along each row and prepend 0.
std::vector<std::vector<double>> simulate_increments(
    int n_steps, int n_paths, double dt, double sigma, uint64_t seed);
