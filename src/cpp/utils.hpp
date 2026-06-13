#pragma once

// Input validation helpers.
// Call validate_brownian_params() at the top of every public function so
// Python callers get a clear error message for invalid arguments.

#include <stdexcept>
#include <string>

inline void validate_brownian_params(int n_steps, int n_paths, double dt, double sigma) {
    if (n_steps <= 0)
        throw std::invalid_argument("n_steps must be > 0");
    if (n_paths <= 0)
        throw std::invalid_argument("n_paths must be > 0");
    if (dt <= 0.0)
        throw std::invalid_argument("dt must be > 0");
    if (sigma < 0.0)
        throw std::invalid_argument("sigma must be >= 0");
}

// ---------------------------------------------------------------------------
// Generic scalar guards, reused by the GBM, pricing, Markov, and stats layers.
// Each throws std::invalid_argument (surfaced to Python as ValueError) with a
// message naming the offending parameter.
// ---------------------------------------------------------------------------

inline void require_gt0_int(int v, const char* name) {
    if (v <= 0)
        throw std::invalid_argument(std::string(name) + " must be > 0");
}

inline void require_gt0(double v, const char* name) {
    if (v <= 0.0)
        throw std::invalid_argument(std::string(name) + " must be > 0");
}

inline void require_ge0(double v, const char* name) {
    if (v < 0.0)
        throw std::invalid_argument(std::string(name) + " must be >= 0");
}

// Geometric Brownian motion adds a strictly positive starting price s0 to the
// usual Brownian parameter checks.
inline void validate_gbm_params(int n_steps, int n_paths, double dt,
                                double sigma, double s0) {
    validate_brownian_params(n_steps, n_paths, dt, sigma);
    require_gt0(s0, "s0");
}
