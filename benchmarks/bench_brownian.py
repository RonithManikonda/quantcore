"""
Benchmark suite for quantcore Brownian motion functions.

Measures three things:
  1. Scaling with n_paths  (fixed n_steps=252, vary paths)
  2. Scaling with n_steps  (fixed n_paths=1000, vary steps)
  3. simulate_paths vs simulate_increments head-to-head

Run with:
    python benchmarks/bench_brownian.py

Record this output as your single-threaded baseline before adding
parallelism in Phase 6. The 'paths/s' column is the most useful
number for comparing threading speedup later.
"""

import time
import numpy as np
from quantcore.stochastic import brownian_paths, brownian_increments

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPS = 100  # repetitions to average over (reduces timer noise)


def timeit(fn, reps=REPS):
    """Return average wall-clock seconds over `reps` calls."""
    # one warm-up call so any lazy init doesn't skew the first rep
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"{'n_steps':>8}  {'n_paths':>8}  {'ms/call':>10}  "
          f"{'MB/call':>8}  {'paths/s':>12}")
    print("-" * 60)


def print_row(n_steps, n_paths, elapsed_s):
    ms       = elapsed_s * 1_000
    mb       = (n_paths * (n_steps + 1) * 8) / 1_048_576   # float64 = 8 bytes
    paths_s  = n_paths / elapsed_s
    print(f"{n_steps:>8}  {n_paths:>8}  {ms:>10.3f}  {mb:>8.2f}  {paths_s:>12.0f}")


# ---------------------------------------------------------------------------
# 1. Path-count scaling  (n_steps fixed at 252 — one trading year of daily steps)
# ---------------------------------------------------------------------------

print_header("Scaling with n_paths  (n_steps=252, dt=1/252, sigma=0.2)")

for n_paths in [100, 500, 1_000, 5_000, 10_000, 50_000]:
    elapsed = timeit(lambda p=n_paths: brownian_paths(
        n_steps=252, n_paths=p, dt=1/252, sigma=0.2, seed=42))
    print_row(252, n_paths, elapsed)

# ---------------------------------------------------------------------------
# 2. Step-count scaling  (n_paths fixed at 1000)
# ---------------------------------------------------------------------------

print_header("Scaling with n_steps  (n_paths=1000, dt=0.01, sigma=1.0)")

for n_steps in [50, 100, 252, 500, 1_000, 2_000]:
    elapsed = timeit(lambda s=n_steps: brownian_paths(
        n_steps=s, n_paths=1_000, dt=0.01, sigma=1.0, seed=42))
    print_row(n_steps, 1_000, elapsed)

# ---------------------------------------------------------------------------
# 3. simulate_paths vs simulate_increments  (same workload, head-to-head)
# ---------------------------------------------------------------------------

print_header("simulate_paths vs simulate_increments  (n_steps=252, n_paths=5000)")

N_STEPS, N_PATHS = 252, 5_000

t_paths = timeit(lambda: brownian_paths(
    n_steps=N_STEPS, n_paths=N_PATHS, dt=1/252, sigma=0.2, seed=0))
t_inc = timeit(lambda: brownian_increments(
    n_steps=N_STEPS, n_paths=N_PATHS, dt=1/252, sigma=0.2, seed=0))

output_mb_paths = (N_PATHS * (N_STEPS + 1) * 8) / 1_048_576
output_mb_inc   = (N_PATHS *  N_STEPS      * 8) / 1_048_576

print(f"  simulate_paths      : {t_paths * 1000:8.3f} ms  "
      f"({output_mb_paths:.2f} MB output)")
print(f"  simulate_increments : {t_inc   * 1000:8.3f} ms  "
      f"({output_mb_inc:.2f} MB output)")
ratio = t_paths / t_inc if t_inc > 0 else float('inf')
print(f"  ratio (paths/inc)   : {ratio:.3f}x")
print()
print("Note: paths requires one extra column and a cumsum; increments")
print("are the raw draws. Any large ratio here is worth investigating.")

# ---------------------------------------------------------------------------
# 4. Memory throughput estimate
# ---------------------------------------------------------------------------

print_header("Memory throughput  (paths only, n_steps=252)")

print(f"  {'n_paths':>8}  {'output_MB':>10}  {'GB/s':>8}")
print(f"  {'-'*32}")

for n_paths in [1_000, 10_000, 50_000]:
    elapsed = timeit(lambda p=n_paths: brownian_paths(
        n_steps=252, n_paths=p, dt=1/252, sigma=0.2, seed=42))
    output_bytes = n_paths * 253 * 8
    gb_s = (output_bytes / 1_073_741_824) / elapsed
    print(f"  {n_paths:>8}  {output_bytes / 1_048_576:>10.2f}  {gb_s:>8.2f}")
