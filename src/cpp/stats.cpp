#include "stats.hpp"
#include "utils.hpp"
#include "special.hpp"

#include <vector>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

// Solve the n x n linear system A x = b by Gaussian elimination with partial
// pivoting. A and b are taken by value (modified locally). Throws if the
// system is (numerically) singular — e.g. perfectly collinear features.
std::vector<double> solve_linear_system(
    std::vector<std::vector<double>> A, std::vector<double> b)
{
    const int n = static_cast<int>(A.size());

    for (int col = 0; col < n; ++col) {
        // partial pivot: largest magnitude entry in this column
        int pivot = col;
        double best = std::abs(A[col][col]);
        for (int r = col + 1; r < n; ++r) {
            if (std::abs(A[r][col]) > best) {
                best = std::abs(A[r][col]);
                pivot = r;
            }
        }
        if (best < 1e-12)
            throw std::invalid_argument(
                "regression system is singular (collinear features?)");

        std::swap(A[col], A[pivot]);
        std::swap(b[col], b[pivot]);

        for (int r = col + 1; r < n; ++r) {
            const double factor = A[r][col] / A[col][col];
            for (int c = col; c < n; ++c)
                A[r][c] -= factor * A[col][c];
            b[r] -= factor * b[col];
        }
    }

    std::vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j)
            s -= A[i][j] * x[j];
        x[i] = s / A[i][i];
    }
    return x;
}

}  // namespace

double stat_mean(const std::vector<double>& x) {
    if (x.empty())
        throw std::invalid_argument("input array must be non-empty");
    double s = 0.0;
    for (double v : x) s += v;
    return s / static_cast<double>(x.size());
}

double stat_variance(const std::vector<double>& x, int ddof) {
    const int n = static_cast<int>(x.size());
    if (n <= ddof)
        throw std::invalid_argument("len(x) must be > ddof");
    const double m = stat_mean(x);
    double ss = 0.0;
    for (double v : x) ss += (v - m) * (v - m);
    return ss / static_cast<double>(n - ddof);
}

double stat_std(const std::vector<double>& x, int ddof) {
    return std::sqrt(stat_variance(x, ddof));
}

std::vector<std::vector<double>> covariance_matrix(
    const std::vector<std::vector<double>>& X, int ddof)
{
    if (X.empty())
        throw std::invalid_argument("X must have at least one observation");
    const int n_obs = static_cast<int>(X.size());
    const int n_vars = static_cast<int>(X[0].size());
    if (n_obs <= ddof)
        throw std::invalid_argument("number of observations must be > ddof");

    std::vector<double> mean(n_vars, 0.0);
    for (int o = 0; o < n_obs; ++o)
        for (int v = 0; v < n_vars; ++v)
            mean[v] += X[o][v];
    for (int v = 0; v < n_vars; ++v)
        mean[v] /= static_cast<double>(n_obs);

    std::vector<std::vector<double>> cov(n_vars, std::vector<double>(n_vars, 0.0));
    for (int o = 0; o < n_obs; ++o) {
        for (int a = 0; a < n_vars; ++a) {
            const double da = X[o][a] - mean[a];
            for (int b = 0; b < n_vars; ++b)
                cov[a][b] += da * (X[o][b] - mean[b]);
        }
    }
    const double denom = static_cast<double>(n_obs - ddof);
    for (int a = 0; a < n_vars; ++a)
        for (int b = 0; b < n_vars; ++b)
            cov[a][b] /= denom;
    return cov;
}

std::vector<std::vector<double>> correlation_matrix(
    const std::vector<std::vector<double>>& X)
{
    // ddof cancels in the correlation, so any consistent choice works.
    std::vector<std::vector<double>> cov = covariance_matrix(X, 1);
    const int n = static_cast<int>(cov.size());

    std::vector<double> sd(n);
    for (int i = 0; i < n; ++i)
        sd[i] = std::sqrt(cov[i][i]);

    std::vector<std::vector<double>> corr(n, std::vector<double>(n, 0.0));
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
            corr[a][b] = cov[a][b] / (sd[a] * sd[b]);
    return corr;
}

RegressionResult linear_regression(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    bool fit_intercept)
{
    const int n_obs = static_cast<int>(X.size());
    if (n_obs == 0)
        throw std::invalid_argument("X must have at least one observation");
    if (static_cast<int>(y.size()) != n_obs)
        throw std::invalid_argument("X and y must have the same number of observations");

    const int n_feat = static_cast<int>(X[0].size());
    const int p = n_feat + (fit_intercept ? 1 : 0);
    if (n_obs < p)
        throw std::invalid_argument("need at least as many observations as parameters");

    // Build the normal equations A = D^T D, rhs = D^T y, where each design row
    // D is [1, x_1, ..., x_k] (the leading 1 present only with an intercept).
    std::vector<std::vector<double>> A(p, std::vector<double>(p, 0.0));
    std::vector<double> rhs(p, 0.0);
    std::vector<double> row(p);

    for (int o = 0; o < n_obs; ++o) {
        int idx = 0;
        if (fit_intercept) row[idx++] = 1.0;
        for (int f = 0; f < n_feat; ++f) row[idx++] = X[o][f];

        for (int i = 0; i < p; ++i) {
            rhs[i] += row[i] * y[o];
            for (int j = 0; j < p; ++j)
                A[i][j] += row[i] * row[j];
        }
    }

    const std::vector<double> sol = solve_linear_system(A, rhs);

    RegressionResult res;
    int idx = 0;
    res.intercept = fit_intercept ? sol[idx++] : 0.0;
    res.coefficients.assign(sol.begin() + idx, sol.end());

    // Coefficient of determination R^2 = 1 - SS_res / SS_tot.
    const double ybar = stat_mean(y);
    double ss_res = 0.0, ss_tot = 0.0;
    for (int o = 0; o < n_obs; ++o) {
        double yhat = res.intercept;
        for (int f = 0; f < n_feat; ++f)
            yhat += res.coefficients[f] * X[o][f];
        ss_res += (y[o] - yhat) * (y[o] - yhat);
        ss_tot += (y[o] - ybar) * (y[o] - ybar);
    }
    res.r_squared = (ss_tot > 0.0) ? 1.0 - ss_res / ss_tot : 0.0;
    return res;
}

std::pair<double, double> mean_confidence_interval(
    const std::vector<double>& x, double confidence)
{
    if (confidence <= 0.0 || confidence >= 1.0)
        throw std::invalid_argument("confidence must be in (0, 1)");
    const int n = static_cast<int>(x.size());
    if (n < 2)
        throw std::invalid_argument("need at least 2 observations");

    const double m = stat_mean(x);
    const double se = stat_std(x, 1) / std::sqrt(static_cast<double>(n));
    const double z = norm_ppf(0.5 * (1.0 + confidence));  // two-sided z-score
    return {m - z * se, m + z * se};
}
