#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>   // py::array_t — needed to return numpy arrays
#include <pybind11/stl.h>     // automatic std::vector <-> list conversion (kept for future use)

#include "brownian.hpp"

namespace py = pybind11;

// Convert a 2D vector-of-vectors (row-major) to a contiguous numpy array.
// The C++ functions return std::vector<std::vector<double>> so that they stay
// independent of pybind11; this helper does the conversion at the boundary.
static py::array_t<double> to_numpy_2d(const std::vector<std::vector<double>>& data) {
    if (data.empty()) return py::array_t<double>({0, 0});
    py::ssize_t rows = static_cast<py::ssize_t>(data.size());
    py::ssize_t cols = static_cast<py::ssize_t>(data[0].size());
    py::array_t<double> arr({rows, cols});
    auto buf = arr.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < rows; ++i)
        for (py::ssize_t j = 0; j < cols; ++j)
            buf(i, j) = data[i][j];
    return arr;
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
}
