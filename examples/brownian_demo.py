"""
Brownian motion demo — quantcore Phase 5

Walks through the key properties of the simulator section by section.
Each section prints what it checks and why, so you can read the output
alongside the code and build intuition for the model.

Run with:
    python examples/brownian_demo.py
"""

import numpy as np
from quantcore.stochastic import brownian_paths, brownian_increments

SEPARATOR = "\n" + "─" * 60

# ---------------------------------------------------------------------------
# 1. Basic API shape check
# ---------------------------------------------------------------------------

print(SEPARATOR)
print("1. SHAPE AND STARTING CONDITION")
print(SEPARATOR)

# 5 paths, 252 daily steps (one trading year), 20% annual volatility
paths = brownian_paths(n_steps=252, n_paths=5, dt=1/252, sigma=0.2, seed=42)

print(f"  Output shape : {paths.shape}")
print(f"  Expected     : (5, 253)  — n_paths rows, n_steps+1 columns")
print(f"  dtype        : {paths.dtype}")
print()
print(f"  First column (W(0) = 0 for every path):")
print(f"    {paths[:, 0]}")
print()
print(f"  Final values at t=1 year:")
for i, v in enumerate(paths[:, -1]):
    print(f"    path {i}: {v:+.4f}")

# ---------------------------------------------------------------------------
# 2. Seed reproducibility and independence
# ---------------------------------------------------------------------------

print(SEPARATOR)
print("2. SEED REPRODUCIBILITY")
print(SEPARATOR)

a = brownian_paths(n_steps=100, n_paths=3, dt=0.01, seed=7)
b = brownian_paths(n_steps=100, n_paths=3, dt=0.01, seed=7)
c = brownian_paths(n_steps=100, n_paths=3, dt=0.01, seed=8)

print(f"  Same seed (7) → identical arrays : {np.array_equal(a, b)}")
print(f"  Diff seed (8) → different arrays : {not np.array_equal(a, c)}")
print()
print("  Max absolute difference between two same-seed runs:")
print(f"    {np.max(np.abs(a - b)):.2e}  (should be exactly 0)")

# ---------------------------------------------------------------------------
# 3. Paths vs increments — the cumsum relationship
# ---------------------------------------------------------------------------

print(SEPARATOR)
print("3. PATHS vs INCREMENTS — CUMSUM RELATIONSHIP")
print(SEPARATOR)

# These must be called with the same seed to get matching sequences
inc  = brownian_increments(n_steps=50, n_paths=4, dt=0.01, seed=99)
full = brownian_paths     (n_steps=50, n_paths=4, dt=0.01, seed=99)

reconstructed = np.concatenate(
    [np.zeros((4, 1)), np.cumsum(inc, axis=1)], axis=1
)

print(f"  increments shape : {inc.shape}   — (n_paths, n_steps)")
print(f"  paths shape      : {full.shape}  — (n_paths, n_steps+1)")
print()
print(f"  cumsum(increments) matches paths exactly: {np.allclose(reconstructed, full)}")
print(f"  Max absolute error: {np.max(np.abs(reconstructed - full)):.2e}")

# ---------------------------------------------------------------------------
# 4. Variance grows linearly with time  (the sqrt-t law)
# ---------------------------------------------------------------------------

print(SEPARATOR)
print("4. VARIANCE SCALING  —  Var(W(t)) = sigma^2 * t")
print(SEPARATOR)

sigma, dt = 1.0, 0.01
n_paths_stat = 10_000  # large sample for accurate estimates

paths = brownian_paths(n_steps=100, n_paths=n_paths_stat, dt=dt, sigma=sigma, seed=0)

print(f"  Parameters: sigma={sigma}, dt={dt}, n_paths={n_paths_stat}")
print()
print(f"  {'t':>6}  {'step k':>6}  {'measured var':>14}  {'expected var':>14}  {'error %':>8}")
print(f"  {'-'*56}")
for k in [1, 5, 10, 25, 50, 100]:
    t         = k * dt
    measured  = paths[:, k].var()
    expected  = sigma**2 * t
    err_pct   = abs(measured - expected) / expected * 100
    print(f"  {t:>6.2f}  {k:>6}  {measured:>14.6f}  {expected:>14.6f}  {err_pct:>7.2f}%")

print()
print("  Key insight: variance grows linearly with time (not with sqrt(t)).")
print("  Standard deviation grows with sqrt(t) — that's the 'sqrt-t rule'")
print("  behind annualising volatility in finance.")

# ---------------------------------------------------------------------------
# 5. Increment distribution — are the draws actually Gaussian?
# ---------------------------------------------------------------------------

print(SEPARATOR)
print("5. INCREMENT DISTRIBUTION  —  dW ~ N(0, sigma^2 * dt)")
print(SEPARATOR)

sigma, dt = 2.0, 0.05
inc = brownian_increments(n_steps=500, n_paths=2_000, dt=dt, sigma=sigma, seed=1)

flat = inc.flatten()
theoretical_std = sigma * np.sqrt(dt)

print(f"  Parameters: sigma={sigma}, dt={dt}")
print(f"  Total draws: {flat.size:,}")
print()
print(f"  {'Statistic':<22}  {'Measured':>12}  {'Expected':>12}")
print(f"  {'-'*50}")
print(f"  {'mean':<22}  {flat.mean():>12.6f}  {'0.000000':>12}")
print(f"  {'std dev':<22}  {flat.std():>12.6f}  {theoretical_std:>12.6f}")
print(f"  {'variance':<22}  {flat.var():>12.6f}  {sigma**2 * dt:>12.6f}")

# skewness and excess kurtosis — both should be near 0 for a Gaussian
mean = flat.mean()
std  = flat.std()
skew = np.mean(((flat - mean) / std) ** 3)
kurt = np.mean(((flat - mean) / std) ** 4) - 3  # excess kurtosis
print(f"  {'skewness (expect 0)':<22}  {skew:>12.4f}  {'0.0000':>12}")
print(f"  {'excess kurtosis (exp 0)':<22}  {kurt:>12.4f}  {'0.0000':>12}")

# ---------------------------------------------------------------------------
# 6. Effect of sigma — scaling the diffusion coefficient
# ---------------------------------------------------------------------------

print(SEPARATOR)
print("6. EFFECT OF SIGMA ON SPREAD")
print(SEPARATOR)

dt = 1/252   # daily steps
n_paths_sigma = 5_000
n_steps = 252

print(f"  Measuring final spread (std of W(T)) at T=1 year, dt=1/252")
print()
print(f"  {'sigma':>8}  {'measured std(W(T))':>20}  {'expected std(W(T))':>20}")
print(f"  {'-'*54}")

for sigma in [0.1, 0.2, 0.5, 1.0, 2.0]:
    p       = brownian_paths(n_steps=n_steps, n_paths=n_paths_sigma,
                             dt=dt, sigma=sigma, seed=0)
    meas    = p[:, -1].std()
    expected = sigma * np.sqrt(n_steps * dt)   # = sigma * sqrt(T)
    print(f"  {sigma:>8.1f}  {meas:>20.6f}  {expected:>20.6f}")

# ---------------------------------------------------------------------------
# 7. What a path "looks like" — terminal statistics across many paths
# ---------------------------------------------------------------------------

print(SEPARATOR)
print("7. TERMINAL VALUE DISTRIBUTION  (sigma=0.2, T=1yr, 20,000 paths)")
print(SEPARATOR)

paths = brownian_paths(n_steps=252, n_paths=20_000, dt=1/252, sigma=0.2, seed=42)
terminal = paths[:, -1]

pcts = [1, 5, 25, 50, 75, 95, 99]
quantiles = np.percentile(terminal, pcts)

print(f"  mean   : {terminal.mean():+.4f}  (expect ≈ 0)")
print(f"  std    : {terminal.std():.4f}   (expect ≈ {0.2 * np.sqrt(1):.4f})")
print()
print(f"  Percentiles of W(T=1):")
for p, q in zip(pcts, quantiles):
    bar = "█" * int(abs(q) * 30)
    sign = "+" if q >= 0 else ""
    print(f"    p{p:>2}:  {sign}{q:.4f}  {bar}")
