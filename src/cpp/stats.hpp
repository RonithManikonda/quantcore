#pragma once

#include <vector>
#include <utility>  // std::pair

// ---------------------------------------------------------------------------
// Statistics utilities.
//
// These take data arrays as input from Python (1D samples or 2D data matrices
// with observations in rows, variables in columns) and compute everything in
// C++. They are validated against NumPy / textbook formulas in the tests.
// ---------------------------------------------------------------------------

struct RegressionResult {
    double intercept;                 // 0.0 when fit_intercept is false
    std::vector<double> coefficients; // one slope per feature (column of X)
    double r_squared;                 // coefficient of determination
};

// Sample mean of a 1D array (must be non-empty).
double stat_mean(const std::vector<double>& x);

// Variance with `ddof` delta degrees of freedom (ddof=0 population, ddof=1
// sample). Requires len(x) > ddof.
double stat_variance(const std::vector<double>& x, int ddof);

// Standard deviation = sqrt(variance).
double stat_std(const std::vector<double>& x, int ddof);

// Covariance matrix of X [n_obs][n_vars]; returns an [n_vars][n_vars] matrix.
// Matches numpy.cov(X, rowvar=False, ddof=ddof).
std::vector<std::vector<double>> covariance_matrix(
    const std::vector<std::vector<double>>& X, int ddof);

// Pearson correlation matrix of X [n_obs][n_vars] -> [n_vars][n_vars].
// Matches numpy.corrcoef(X, rowvar=False).
std::vector<std::vector<double>> correlation_matrix(
    const std::vector<std::vector<double>>& X);

// Ordinary least squares fit of y ~ X (optionally with an intercept), solved
// via the normal equations. X is [n_obs][n_features], y is length n_obs.
RegressionResult linear_regression(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    bool fit_intercept);

// Two-sided confidence interval for the population mean, using the normal
// (large-sample) approximation: mean +/- z * s / sqrt(n). Returns {low, high}.
std::pair<double, double> mean_confidence_interval(
    const std::vector<double>& x, double confidence);
