"""
Monte Carlo option pricing benchmark for quantcore (Phase 7).

Measures three things:
  1. Throughput  — simulated paths per second for the Monte Carlo pricer.
  2. Convergence — how the estimate and its standard error approach the
     closed-form Black-Scholes price as the path count grows.
  3. Variance reduction — the standard-error gain from antithetic variates at
     a fixed path budget.

Run with:
    python benchmarks/bench_pricing.py
"""

import time

from quantcore import black_scholes, mc_european

# At-the-money contract used throughout.
S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
EXACT = black_scholes(S0, K, R, SIGMA, T, "call")


def timeit(fn, reps=20):
    fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


# ---------------------------------------------------------------------------
# 1. Throughput
# ---------------------------------------------------------------------------

print(f"\n{'=' * 64}")
print("  Monte Carlo throughput  (European call, antithetic)")
print(f"{'=' * 64}")
print(f"  {'n_paths':>12}{'ms/call':>12}{'paths/s':>16}")
print(f"  {'-' * 40}")

for n in [10_000, 100_000, 1_000_000]:
    elapsed = timeit(lambda n=n: mc_european(
        S0, K, R, SIGMA, T, n_paths=n, seed=1, antithetic=True))
    print(f"  {n:>12,}{elapsed * 1000:>12.3f}{n / elapsed:>16,.0f}")

# ---------------------------------------------------------------------------
# 2. Convergence to Black-Scholes
# ---------------------------------------------------------------------------

print(f"\n{'=' * 64}")
print(f"  Convergence to Black-Scholes  (exact = {EXACT:.6f})")
print(f"{'=' * 64}")
print(f"  {'n_paths':>12}{'MC price':>14}{'std error':>12}{'|error|':>12}")
print(f"  {'-' * 50}")

for n in [1_000, 10_000, 100_000, 1_000_000, 10_000_000]:
    res = mc_european(S0, K, R, SIGMA, T, n_paths=n, seed=2024, antithetic=True)
    print(f"  {n:>12,}{res.price:>14.6f}{res.std_error:>12.6f}"
          f"{abs(res.price - EXACT):>12.6f}")

# ---------------------------------------------------------------------------
# 3. Variance reduction from antithetic variates
# ---------------------------------------------------------------------------

print(f"\n{'=' * 64}")
print("  Antithetic variance reduction  (same path budget)")
print(f"{'=' * 64}")
print(f"  {'n_paths':>12}{'plain SE':>12}{'antithetic SE':>16}{'SE ratio':>12}")
print(f"  {'-' * 52}")

for n in [50_000, 200_000, 1_000_000]:
    plain = mc_european(S0, K, R, SIGMA, T, n_paths=n, seed=7, antithetic=False)
    anti = mc_european(S0, K, R, SIGMA, T, n_paths=n, seed=7, antithetic=True)
    ratio = plain.std_error / anti.std_error
    print(f"  {n:>12,}{plain.std_error:>12.6f}{anti.std_error:>16.6f}{ratio:>11.2f}x")

print("\nThe standard-error ratio squared is the effective speedup: an")
print("antithetic run reaches the same accuracy as that many plain paths.")
