#include <vector>
#include <cstddef>

double add(double a, double b) {
	return a + b;
}

std::vector<double> linespace(double start, double end, std::size_t n) {
	std::vector<double> res(n);
	double step = (end - start) / (n - 1);

	if (n == 0) return res;
	if (n == 1) return {start};


	for (int i = 0; i < n; i++) {
		res[i] = start + (i * step);
	}

	return res;
}

double mean(std::vector<double> data) {
	if (data.size() == 0) return 0;
	double sum = 0;

	for (double val : data) {
		sum += val;
	}

	return sum / data.size();
}
