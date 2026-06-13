"""
Geometric Brownian motion demo — quantcore Phase 7

Shows the GBM public API and verifies the lognormal properties that make it
the standard model for asset prices.

Run with:
    python examples/gbm_demo.py
"""

import numpy as np

from quantcore import gbm_paths, gbm_terminal

SEPARATOR = "\n" + "─" * 60

# ---------------------------------------------------------------------------
# 1. Shape, starting condition, positivity
# ---------------------------------------------------------------------------

print(SEPARATOR)
print("1. SHAPE, START PRICE, AND POSITIVITY")
print(SEPARATOR)

# 5 paths, one trading year of daily steps, 8% drift, 20% vol, $100 spot
paths = gbm_paths(n_steps=252, n_paths=5, dt=1 / 252, mu=0.08, sigma=0.2,
                  s0=100.0, seed=42)

print(f"  Output shape : {paths.shape}   (n_paths, n_steps + 1)")
print(f"  Start prices : {paths[:, 0]}   (all = s0)")
print(f"  All positive : {bool(np.all(paths > 0))}   (GBM can never go <= 0)")
print()
print("  Final prices at t = 1 year:")
for i, v in enumerate(paths[:, -1]):
    print(f"    path {i}: {v:8.2f}")

# ---------------------------------------------------------------------------
# 2. Terminal distribution vs closed-form lognormal moments
# ---------------------------------------------------------------------------

print(SEPARATOR)
print("2. TERMINAL MOMENTS  —  E[S_T] = s0 * exp(mu * T)")
print(SEPARATOR)

mu, sigma, T, s0 = 0.08, 0.2, 1.0, 100.0
terminal = gbm_terminal(n_paths=200_000, T=T, mu=mu, sigma=sigma, s0=s0, seed=1)

expected_mean = s0 * np.exp(mu * T)
expected_median = s0 * np.exp((mu - 0.5 * sigma ** 2) * T)

print(f"  Parameters: mu={mu}, sigma={sigma}, T={T}, s0={s0}, 200k samples")
print()
print(f"  {'quantity':<14}{'measured':>14}{'theory':>14}")
print(f"  {'-' * 42}")
print(f"  {'mean':<14}{terminal.mean():>14.4f}{expected_mean:>14.4f}")
print(f"  {'median':<14}{np.median(terminal):>14.4f}{expected_median:>14.4f}")
print()
print("  mean > median because the lognormal distribution is right-skewed.")

# ---------------------------------------------------------------------------
# 3. Effect of drift and volatility on the terminal spread
# ---------------------------------------------------------------------------

print(SEPARATOR)
print("3. EFFECT OF DRIFT AND VOLATILITY")
print(SEPARATOR)

print(f"  {'mu':>6}{'sigma':>8}{'p5':>12}{'median':>12}{'p95':>12}")
print(f"  {'-' * 50}")
for mu, sigma in [(0.0, 0.1), (0.0, 0.3), (0.1, 0.2), (-0.1, 0.2)]:
    s = gbm_terminal(n_paths=100_000, T=1.0, mu=mu, sigma=sigma, s0=100.0, seed=7)
    p5, med, p95 = np.percentile(s, [5, 50, 95])
    print(f"  {mu:>6.2f}{sigma:>8.2f}{p5:>12.2f}{med:>12.2f}{p95:>12.2f}")

print()
print("  Higher sigma widens the terminal spread; positive mu shifts it up.")
