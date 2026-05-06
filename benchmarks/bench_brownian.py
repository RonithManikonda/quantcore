"""
Benchmark: brownian_paths at various (n_steps, n_paths) sizes.

Run with:
    python benchmarks/bench_brownian.py

Record the output before you add parallelism (Phase 6) so you have a
single-threaded baseline to compare against.
"""

import time
import numpy as np
from quantcore.stochastic import brownian_paths

REPS = 5  # average over this many runs to reduce timing noise

configs = [
    # (n_steps, n_paths)
    (100,      100),
    (100,    1_000),
    (100,   10_000),
    (1_000,    100),
    (1_000,  1_000),
    (1_000, 10_000),
]

print(f"{'n_steps':>8}  {'n_paths':>8}  {'ms/call':>10}  {'paths/s':>12}")
print("-" * 46)

for n_steps, n_paths in configs:
    t0 = time.perf_counter()
    for _ in range(REPS):
        brownian_paths(n_steps=n_steps, n_paths=n_paths, dt=0.01, sigma=1.0, seed=42)
    elapsed = (time.perf_counter() - t0) / REPS
    paths_per_sec = n_paths / elapsed
    print(f"{n_steps:>8}  {n_paths:>8}  {elapsed * 1000:>10.3f}  {paths_per_sec:>12.0f}")
