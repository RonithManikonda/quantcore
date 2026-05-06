#pragma once

// Input validation helpers.
// Call validate_brownian_params() at the top of every public function so
// Python callers get a clear error message for invalid arguments.

#include <stdexcept>

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
