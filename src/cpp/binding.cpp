#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>   // py::array_t — needed to return numpy arrays
#include <pybind11/stl.h>     // automatic std::vector <-> list conversion (kept for future use)

#include "brownian.hpp"
#include "gbm.hpp"
#include "pricing.hpp"
#include "markov.hpp"
#include "stats.hpp"

namespace py = pybind11;

// Convert a 2D vector-of-vectors (row-major) to a contiguous numpy array.
// The C++ functions return std::vector<std::vector<T>> so that they stay
// independent of pybind11; this helper does the conversion at the boundary.
// Templated on T so it serves both double (simulations) and int (Markov states).
template <typename T>
static py::array_t<T> to_numpy_2d(const std::vector<std::vector<T>>& data) {
    if (data.empty()) return py::array_t<T>({py::ssize_t(0), py::ssize_t(0)});
    py::ssize_t rows = static_cast<py::ssize_t>(data.size());
    py::ssize_t cols = static_cast<py::ssize_t>(data[0].size());
    py::array_t<T> arr({rows, cols});
    auto buf = arr.template mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < rows; ++i)
        for (py::ssize_t j = 0; j < cols; ++j)
            buf(i, j) = data[i][j];
    return arr;
}

// Convert a 1D std::vector<double> to a contiguous numpy array.
static py::array_t<double> to_numpy_1d(const std::vector<double>& data) {
    py::ssize_t n = static_cast<py::ssize_t>(data.size());
    py::array_t<double> arr(n);
    auto buf = arr.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < n; ++i)
        buf(i) = data[i];
    return arr;
}

// Convert a 2D numpy array (any numeric dtype, contiguous or not — forcecast
// handles the conversion) into a row-major vector-of-vectors of doubles, for
// the routines that take a matrix as input (Markov P, stats data matrices).
static std::vector<std::vector<double>> numpy_to_matrix(
    py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    if (arr.ndim() != 2)
        throw std::invalid_argument("expected a 2D array");
    auto buf = arr.unchecked<2>();
    py::ssize_t rows = arr.shape(0);
    py::ssize_t cols = arr.shape(1);
    std::vector<std::vector<double>> out(rows, std::vector<double>(cols));
    for (py::ssize_t i = 0; i < rows; ++i)
        for (py::ssize_t j = 0; j < cols; ++j)
            out[i][j] = buf(i, j);
    return out;
}

// Convert a 1D numpy array (any numeric dtype) into a std::vector<double>.
static std::vector<double> numpy_to_vector(
    py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    if (arr.ndim() != 1)
        throw std::invalid_argument("expected a 1D array");
    auto buf = arr.unchecked<1>();
    py::ssize_t n = arr.shape(0);
    std::vector<double> out(n);
    for (py::ssize_t i = 0; i < n; ++i)
        out[i] = buf(i);
    return out;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core for quantcore";

    m.def("simulate_paths",
        [](int n_steps, int n_paths, double dt, double sigma, uint64_t seed) {
            return to_numpy_2d(simulate_paths(n_steps, n_paths, dt, sigma, seed));
        },
        py::arg("n_steps"),
        py::arg("n_paths"),
        py::arg("dt"),
        py::arg("sigma") = 1.0,
        py::arg("seed")  = 0,
        "Simulate Brownian motion paths.\n\n"
        "Returns a numpy array of shape (n_paths, n_steps + 1).\n"
        "Each row is one path; all rows start at 0.");

    m.def("simulate_increments",
        [](int n_steps, int n_paths, double dt, double sigma, uint64_t seed) {
            return to_numpy_2d(simulate_increments(n_steps, n_paths, dt, sigma, seed));
        },
        py::arg("n_steps"),
        py::arg("n_paths"),
        py::arg("dt"),
        py::arg("sigma") = 1.0,
        py::arg("seed")  = 0,
        "Simulate Brownian motion increments dW ~ N(0, sigma^2 * dt).\n\n"
        "Returns a numpy array of shape (n_paths, n_steps).");

    m.def("simulate_paths_parallel",
        [](int n_steps, int n_paths, double dt, double sigma, uint64_t seed, int n_threads) {
            return to_numpy_2d(simulate_paths_parallel(n_steps, n_paths, dt, sigma, seed, n_threads));
        },
        py::arg("n_steps"),
        py::arg("n_paths"),
        py::arg("dt"),
        py::arg("sigma")     = 1.0,
        py::arg("seed")      = 0,
        py::arg("n_threads") = 0,
        "Multithreaded Brownian motion paths.\n\n"
        "Same output as simulate_paths. n_threads=0 auto-detects core count.\n"
        "Returns a numpy array of shape (n_paths, n_steps + 1).");

    m.def("simulate_increments_parallel",
        [](int n_steps, int n_paths, double dt, double sigma, uint64_t seed, int n_threads) {
            return to_numpy_2d(simulate_increments_parallel(n_steps, n_paths, dt, sigma, seed, n_threads));
        },
        py::arg("n_steps"),
        py::arg("n_paths"),
        py::arg("dt"),
        py::arg("sigma")     = 1.0,
        py::arg("seed")      = 0,
        py::arg("n_threads") = 0,
        "Multithreaded Brownian motion increments.\n\n"
        "Same output as simulate_increments. n_threads=0 auto-detects core count.\n"
        "Returns a numpy array of shape (n_paths, n_steps).");

    // -----------------------------------------------------------------------
    // Geometric Brownian motion
    // -----------------------------------------------------------------------

    m.def("simulate_gbm_paths",
        [](int n_steps, int n_paths, double dt, double mu, double sigma,
           double s0, uint64_t seed) {
            return to_numpy_2d(simulate_gbm_paths(n_steps, n_paths, dt, mu, sigma, s0, seed));
        },
        py::arg("n_steps"),
        py::arg("n_paths"),
        py::arg("dt"),
        py::arg("mu")    = 0.0,
        py::arg("sigma") = 1.0,
        py::arg("s0")    = 1.0,
        py::arg("seed")  = 0,
        "Simulate geometric Brownian motion price paths.\n\n"
        "Returns a numpy array of shape (n_paths, n_steps + 1).\n"
        "Each row is one path; all rows start at s0.");

    m.def("simulate_gbm_paths_parallel",
        [](int n_steps, int n_paths, double dt, double mu, double sigma,
           double s0, uint64_t seed, int n_threads) {
            return to_numpy_2d(simulate_gbm_paths_parallel(
                n_steps, n_paths, dt, mu, sigma, s0, seed, n_threads));
        },
        py::arg("n_steps"),
        py::arg("n_paths"),
        py::arg("dt"),
        py::arg("mu")        = 0.0,
        py::arg("sigma")     = 1.0,
        py::arg("s0")        = 1.0,
        py::arg("seed")      = 0,
        py::arg("n_threads") = 0,
        "Multithreaded geometric Brownian motion paths.\n\n"
        "Same output as simulate_gbm_paths. n_threads=0 auto-detects core count.");

    m.def("simulate_gbm_terminal",
        [](int n_paths, double T, double mu, double sigma, double s0, uint64_t seed) {
            return to_numpy_1d(simulate_gbm_terminal(n_paths, T, mu, sigma, s0, seed));
        },
        py::arg("n_paths"),
        py::arg("T"),
        py::arg("mu")    = 0.0,
        py::arg("sigma") = 1.0,
        py::arg("s0")    = 1.0,
        py::arg("seed")  = 0,
        "Simulate terminal GBM values S(T) from the closed-form lognormal law.\n\n"
        "Returns a 1D numpy array of length n_paths.");

    // -----------------------------------------------------------------------
    // Option pricing
    // -----------------------------------------------------------------------

    m.def("black_scholes_price", &black_scholes_price,
        py::arg("s0"),
        py::arg("K"),
        py::arg("r"),
        py::arg("sigma"),
        py::arg("T"),
        py::arg("is_call"),
        "Closed-form Black-Scholes price of a European option.\n\n"
        "is_call = True for a call, False for a put.");

    m.def("mc_european_price", &mc_european_price,
        py::arg("s0"),
        py::arg("K"),
        py::arg("r"),
        py::arg("sigma"),
        py::arg("T"),
        py::arg("n_paths"),
        py::arg("is_call"),
        py::arg("seed")       = 0,
        py::arg("antithetic") = true,
        "Monte Carlo price of a European option under the risk-neutral measure.\n\n"
        "Returns a (price, standard_error) tuple.");

    // -----------------------------------------------------------------------
    // Markov chains
    // -----------------------------------------------------------------------

    m.def("simulate_markov",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> P,
           int n_steps, int n_chains, int start_state, uint64_t seed) {
            return to_numpy_2d(simulate_markov(
                numpy_to_matrix(P), n_steps, n_chains, start_state, seed));
        },
        py::arg("P"),
        py::arg("n_steps"),
        py::arg("n_chains")    = 1,
        py::arg("start_state") = 0,
        py::arg("seed")        = 0,
        "Simulate discrete-time Markov chains over a finite state space.\n\n"
        "P is a row-stochastic transition matrix. Returns an int array of\n"
        "shape (n_chains, n_steps + 1); column 0 is start_state.");

    // -----------------------------------------------------------------------
    // Statistics utilities (take array input, compute in C++)
    // -----------------------------------------------------------------------

    m.def("stat_mean",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> x) {
            return stat_mean(numpy_to_vector(x));
        },
        py::arg("x"), "Sample mean of a 1D array.");

    m.def("stat_variance",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> x, int ddof) {
            return stat_variance(numpy_to_vector(x), ddof);
        },
        py::arg("x"), py::arg("ddof") = 0, "Variance with `ddof` degrees of freedom.");

    m.def("stat_std",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> x, int ddof) {
            return stat_std(numpy_to_vector(x), ddof);
        },
        py::arg("x"), py::arg("ddof") = 0, "Standard deviation with `ddof` degrees of freedom.");

    m.def("covariance_matrix",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> X, int ddof) {
            return to_numpy_2d(covariance_matrix(numpy_to_matrix(X), ddof));
        },
        py::arg("X"), py::arg("ddof") = 1,
        "Covariance matrix of X (observations in rows, variables in columns).");

    m.def("correlation_matrix",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> X) {
            return to_numpy_2d(correlation_matrix(numpy_to_matrix(X)));
        },
        py::arg("X"), "Pearson correlation matrix of X (observations in rows).");

    m.def("linear_regression",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> X,
           py::array_t<double, py::array::c_style | py::array::forcecast> y,
           bool fit_intercept) {
            RegressionResult res = linear_regression(
                numpy_to_matrix(X), numpy_to_vector(y), fit_intercept);
            return py::make_tuple(res.intercept,
                                  to_numpy_1d(res.coefficients),
                                  res.r_squared);
        },
        py::arg("X"), py::arg("y"), py::arg("fit_intercept") = true,
        "Ordinary least squares fit. Returns (intercept, coefficients, r_squared).");

    m.def("mean_confidence_interval",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> x, double confidence) {
            return mean_confidence_interval(numpy_to_vector(x), confidence);
        },
        py::arg("x"), py::arg("confidence") = 0.95,
        "Two-sided normal-approximation confidence interval for the mean.\n\n"
        "Returns a (low, high) tuple.");
}
