#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

double add(double a, double b);
std::vector<double> linespace(double start, double end, std::size_t n);
double mean(std::vector<double> data);

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
	m.doc() = "C++ core for quantcore";
	m.def("add", &add, "Add two numbers");
	m.def("linespace", &linespace, "Returns a linespace of a given size");
	m.def("mean", &mean, "Returns the mean of the input data");
}
