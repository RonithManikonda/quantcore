"""
Parallel scaling benchmark for quantcore Brownian motion (Phase 6).

Measures wall-clock time for simulate_paths_parallel at a fixed workload
across an increasing thread count, and reports the speedup relative to the
single-threaded (n_threads=1) baseline.

Run with:
    python benchmarks/bench_parallel.py

Reading the output:
  - 'speedup' is t(1 thread) / t(n threads). Ideal linear scaling would
    match the thread count; in practice memory bandwidth and thread
    overhead cap it well below that for this memory-bound workload.
  - 'efficiency' is speedup / n_threads (1.00 = perfect scaling).
  - Small workloads show threading *overhead* dominating — that is the
    expected and useful negative result, not a bug.
"""

import os
import time

import numpy as np

from quantcore import brownian_paths_parallel

REPS = 20  # fewer reps than the serial suite — these workloads are large


def timeit(fn, reps=REPS):
    fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def bench_workload(n_steps, n_paths, thread_counts):
    print(f"\n{'=' * 64}")
    print(f"  n_steps={n_steps}, n_paths={n_paths:,}  "
          f"({n_paths * (n_steps + 1) * 8 / 1_048_576:.1f} MB output)")
    print(f"{'=' * 64}")
    print(f"  {'threads':>8}  {'ms/call':>10}  {'paths/s':>14}  "
          f"{'speedup':>9}  {'efficiency':>11}")
    print(f"  {'-' * 60}")

    baseline = None
    for nt in thread_counts:
        elapsed = timeit(lambda nt=nt: brownian_paths_parallel(
            n_steps=n_steps, n_paths=n_paths, dt=1 / 252, sigma=0.2,
            seed=42, n_threads=nt))
        if baseline is None:
            baseline = elapsed
        speedup = baseline / elapsed
        eff = speedup / nt
        print(f"  {nt:>8}  {elapsed * 1000:>10.3f}  {n_paths / elapsed:>14,.0f}  "
              f"{speedup:>8.2f}x  {eff:>10.0%}")


if __name__ == "__main__":
    cores = os.cpu_count() or 0
    print(f"Detected {cores} logical CPUs.")

    # thread counts to sweep — clamp to a sensible cap around the core count
    thread_counts = [1, 2, 4, 8]
    if cores >= 16:
        thread_counts.append(16)

    # Large workloads where compute dominates and threading should help.
    bench_workload(252, 50_000, thread_counts)
    bench_workload(252, 200_000, thread_counts)
    bench_workload(1_000, 50_000, thread_counts)

    # Small workload — overhead should dominate (threading does NOT help here).
    bench_workload(50, 500, thread_counts)

    print("\nTakeaway: threading pays off once each thread has enough work to")
    print("amortise its launch cost; for tiny workloads a single thread wins.")
