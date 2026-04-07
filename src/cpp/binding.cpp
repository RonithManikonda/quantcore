#include <pybind11/pybind11.h>

double add(double a, double b);

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
	m.doc() = "C++ core for quantcore";
	m.def("add", &add, "Add two numbers");
}
