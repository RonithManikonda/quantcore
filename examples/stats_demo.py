"""
Statistics utilities demo — quantcore Phase 7

Runs the covariance / correlation / regression / confidence-interval helpers on
synthetic data with known structure, so the recovered numbers can be checked
against the values used to generate the data.

Run with:
    python examples/stats_demo.py
"""

import numpy as np

from quantcore import (
    covariance,
    correlation,
    linear_regression,
    confidence_interval,
    mean,
    std,
)

SEPARATOR = "\n" + "─" * 60

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Covariance and correlation of correlated columns
# ---------------------------------------------------------------------------

print(SEPARATOR)
print("1. COVARIANCE AND CORRELATION")
print(SEPARATOR)

# Two assets: y is partly driven by x, plus its own noise.
x = rng.normal(0, 1, size=5000)
y = 0.8 * x + rng.normal(0, 0.6, size=5000)
data = np.column_stack([x, y])

cov = covariance(data)
corr = correlation(data)

print("  Covariance matrix:")
print("   ", np.array2string(cov, precision=4, prefix="    "))
print()
print("  Correlation matrix:")
print("   ", np.array2string(corr, precision=4, prefix="    "))
print()
print(f"  corr(x, y) = {corr[0, 1]:.4f}  (positive, as constructed)")

# ---------------------------------------------------------------------------
# 2. Linear regression recovering known coefficients
# ---------------------------------------------------------------------------

print(SEPARATOR)
print("2. ORDINARY LEAST SQUARES")
print(SEPARATOR)

n = 4000
X = rng.normal(0, 1, size=(n, 2))
true_intercept = 5.0
true_beta = np.array([2.5, -1.2])
noise = rng.normal(0, 0.3, size=n)
target = true_intercept + X @ true_beta + noise

fit = linear_regression(X, target, fit_intercept=True)

print(f"  {'parameter':<14}{'estimated':>12}{'true':>12}")
print(f"  {'-' * 38}")
print(f"  {'intercept':<14}{fit.intercept:>12.4f}{true_intercept:>12.4f}")
for i, (est, tru) in enumerate(zip(fit.coefficients, true_beta)):
    print(f"  {'beta_' + str(i + 1):<14}{est:>12.4f}{tru:>12.4f}")
print()
print(f"  R-squared: {fit.r_squared:.4f}")

# ---------------------------------------------------------------------------
# 3. Confidence interval for a mean
# ---------------------------------------------------------------------------

print(SEPARATOR)
print("3. CONFIDENCE INTERVAL FOR THE MEAN")
print(SEPARATOR)

sample = rng.normal(loc=10.0, scale=3.0, size=500)
ci95 = confidence_interval(sample, confidence=0.95)
ci99 = confidence_interval(sample, confidence=0.99)

print(f"  Sample: n={sample.size}, mean={mean(sample):.4f}, std={std(sample, ddof=1):.4f}")
print(f"  True mean used to generate the data: 10.0")
print()
print(f"  95% CI: [{ci95.low:.4f}, {ci95.high:.4f}]")
print(f"  99% CI: [{ci99.low:.4f}, {ci99.high:.4f}]   (wider, as expected)")
