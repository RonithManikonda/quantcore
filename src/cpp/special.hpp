#pragma once

// Standard-normal special functions shared by the pricing (Black-Scholes) and
// statistics (confidence interval) layers. Header-only so they inline freely.

#include <cmath>
#include <stdexcept>

// Standard normal probability density function.
inline double norm_pdf(double x) {
    constexpr double inv_sqrt_2pi = 0.3989422804014327;  // 1 / sqrt(2*pi)
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

// Standard normal cumulative distribution function, via the complementary
// error function (accurate to full double precision, no lookup tables).
inline double norm_cdf(double x) {
    constexpr double inv_sqrt2 = 0.7071067811865476;  // 1 / sqrt(2)
    return 0.5 * std::erfc(-x * inv_sqrt2);
}

// Inverse standard normal CDF (the probit / quantile function).
//
// Acklam's rational approximation gives ~1e-9 relative accuracy; a single
// Halley refinement step using norm_cdf then takes it to full precision.
// Used to turn a confidence level (e.g. 0.975) into a z-score.
inline double norm_ppf(double p) {
    if (p <= 0.0 || p >= 1.0) {
        if (p == 0.0) return -INFINITY;
        if (p == 1.0) return INFINITY;
        throw std::invalid_argument("norm_ppf: p must be in (0, 1)");
    }

    // Coefficients for Acklam's algorithm.
    static const double a[] = {-3.969683028665376e+01,  2.209460984245205e+02,
                               -2.759285104469687e+02,  1.383577518672690e+02,
                               -3.066479806614716e+01,  2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01,  1.615858368580409e+02,
                               -1.556989798598866e+02,  6.680131188771972e+01,
                               -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                               -2.400758277161838e+00, -2.549732539343734e+00,
                                4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[] = { 7.784695709041462e-03,  3.224671290700398e-01,
                                2.445134137142996e+00,  3.754408661907416e+00};

    constexpr double p_low = 0.02425;
    constexpr double p_high = 1.0 - p_low;

    double x;
    if (p < p_low) {
        double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= p_high) {
        double q = p - 0.5;
        double r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        double q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }

    // One Halley step to polish to full double precision.
    double e = norm_cdf(x) - p;
    double u = e / norm_pdf(x);
    x -= u / (1.0 + 0.5 * x * u);
    return x;
}
