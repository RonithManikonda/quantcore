# Benchmarks

Reproducible performance numbers for the C++ kernels, measured from Python (so
they include the binding overhead a real caller pays). Regenerate any table
with the scripts in [`benchmarks/`](benchmarks/):

```bash
python benchmarks/bench_brownian.py   # single-threaded scaling
python benchmarks/bench_parallel.py   # 1-vs-many-threads speedup
python benchmarks/bench_pricing.py    # Monte Carlo pricing
```

**Environment.** Apple M4 Pro (14 cores), macOS, Apple clang 21, CPython 3.14,
NumPy 2.4. Each figure is the average of many repetitions after a warm-up call.
Absolute numbers vary by machine; the *ratios* (threading speedup, 1/√n error
decay, antithetic gain) are the portable, meaningful results.

---

## 1. Single-threaded Brownian motion

Throughput generating full paths, shape `(n_paths, n_steps + 1)`.

### Scaling with path count (`n_steps = 252`, one trading year)

| n_paths | ms/call | MB/call | paths/s |
|--------:|--------:|--------:|--------:|
|     100 |   0.101 |    0.19 |   987,886 |
|     500 |   0.657 |    0.97 |   760,600 |
|   1,000 |   1.490 |    1.93 |   670,959 |
|   5,000 |   9.133 |    9.65 |   547,477 |
|  10,000 |  19.673 |   19.30 |   508,316 |
|  50,000 |  95.387 |   96.51 |   524,182 |

### Scaling with step count (`n_paths = 1,000`)

| n_steps | ms/call | MB/call | paths/s |
|--------:|--------:|--------:|--------:|
|      50 |   0.227 |    0.39 | 4,396,490 |
|     100 |   0.514 |    0.77 | 1,946,092 |
|     252 |   1.492 |    1.93 |   670,459 |
|     500 |   3.370 |    3.82 |   296,693 |
|   1,000 |   6.848 |    7.64 |   146,036 |
|   2,000 |  14.002 |   15.27 |    71,417 |

Runtime is linear in the total number of Gaussian draws (`n_paths × n_steps`),
as expected. At large sizes throughput is limited by memory bandwidth (writing
the output array), measured at ~1 GB/s for the 96 MB case — so generating raw
`simulate_increments` is marginally cheaper than full paths (ratio ≈ 1.05×).

---

## 2. Parallel scaling (1 vs many threads)

Work is split into contiguous path batches, one private RNG per thread, each
writing into a disjoint slice of the output. Speedup is relative to the
single-threaded run; efficiency is `speedup / n_threads`.

### `n_steps = 252`, `n_paths = 200,000` (386 MB output)

| threads | ms/call | paths/s | speedup | efficiency |
|--------:|--------:|--------:|--------:|-----------:|
|       1 | 394.6   |   506,799 |  1.00× |   100% |
|       2 | 217.2   |   920,606 |  1.82× |    91% |
|       4 | 131.8   | 1,516,988 |  2.99× |    75% |
|       8 |  87.4   | 2,288,809 |  4.52× |    56% |

### `n_steps = 1,000`, `n_paths = 50,000` (382 MB output, more compute per write)

| threads | ms/call | paths/s | speedup | efficiency |
|--------:|--------:|--------:|--------:|-----------:|
|       1 | 386.2   |   129,458 |  1.00× |   100% |
|       2 | 207.7   |   240,783 |  1.86× |    93% |
|       4 | 120.3   |   415,572 |  3.21× |    80% |
|       8 |  73.3   |   682,418 |  5.27× |    66% |

### Small workload — `n_steps = 50`, `n_paths = 500` (0.2 MB)

| threads | ms/call | speedup |
|--------:|--------:|--------:|
|       1 |  0.161  |  1.00× |
|       2 |  0.113  |  1.43× |
|       4 |  0.100  |  1.60× |
|       8 |  0.115  |  1.40× |

**Takeaways.** Threading scales well once each thread has real work: the
compute-heavier `n_steps = 1,000` case reaches **5.3× on 8 threads**, while the
more bandwidth-bound `n_steps = 252` case tops out around 4.5× as memory
traffic becomes the bottleneck. For tiny workloads, thread launch cost
dominates and a single thread wins — the speedup peaks at 4 threads and then
*regresses*. This is the kind of "measure before you parallelise" result the
benchmark is meant to surface.

---

## 3. Monte Carlo option pricing

European call, at-the-money (`S0 = K = 100`, `r = 5%`, `σ = 20%`, `T = 1`).
Closed-form Black-Scholes price: **10.450584**.

### Throughput (antithetic)

| n_paths | ms/call | paths/s |
|--------:|--------:|--------:|
|    10,000 | 0.111 | 90,074,875 |
|   100,000 | 1.233 | 81,071,220 |
| 1,000,000 | 12.928 | 77,348,593 |

Terminal-value pricing draws one Gaussian and one `exp` per path with no large
array to write back, so it runs at ~**77–90 million paths/second**.

### Convergence to Black-Scholes

| n_paths | MC price | std error | \|error\| |
|--------:|---------:|----------:|----------:|
|       1,000 | 10.675001 | 0.362924 | 0.224418 |
|      10,000 | 10.563265 | 0.103570 | 0.112682 |
|     100,000 | 10.497437 | 0.033045 | 0.046854 |
|   1,000,000 | 10.446410 | 0.010374 | 0.004173 |
|  10,000,000 | 10.448094 | 0.003287 | 0.002490 |

The standard error falls like `1/√n` — a 100× increase in paths roughly
10×-tightens the estimate — and the actual error tracks it.

### Antithetic variance reduction (same path budget)

| n_paths | plain SE | antithetic SE | SE ratio |
|--------:|---------:|--------------:|---------:|
|    50,000 | 0.065267 | 0.046052 | 1.42× |
|   200,000 | 0.032844 | 0.023081 | 1.42× |
| 1,000,000 | 0.014717 | 0.010373 | 1.42× |

Antithetic sampling consistently cuts the standard error by ~1.42×, i.e. it
reaches the same accuracy as **~2× as many** independent paths, for no extra
draws.
