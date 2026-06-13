# QuantCore

**A quantitative-finance library with multithreaded C++ compute kernels and a clean, typed Python API.**

[![CI](https://github.com/RonithManikonda/quantcore/actions/workflows/ci.yml/badge.svg)](https://github.com/RonithManikonda/quantcore/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://en.cppreference.com/w/cpp/17)

QuantCore implements the building blocks of Monte Carlo quantitative finance —
Brownian motion, geometric Brownian motion, option pricing, Markov chains, and
statistics — with the numerically heavy parts written in **C++** and exposed to
Python through **pybind11**. The Python layer stays thin and well-typed; the C++
layer does the work and parallelises across threads when it pays off.

```python
import quantcore as qc

# Price a European call two ways and watch them agree
bs = qc.black_scholes(s0=100, K=100, r=0.05, sigma=0.2, T=1.0, option_type="call")
mc = qc.mc_european(s0=100, K=100, r=0.05, sigma=0.2, T=1.0, n_paths=1_000_000)

print(f"Black-Scholes : {bs:.4f}")
print(f"Monte Carlo   : {mc.price:.4f} ± {mc.std_error:.4f}")
# Black-Scholes : 10.4506
# Monte Carlo   : 10.4565 ± 0.0104
```

---

## Highlights

- **Brownian motion** — paths and raw increments, seeded and reproducible, with
  a **multithreaded** variant that scales to ~5× on 8 cores.
- **Geometric Brownian motion** — exact lognormal discretisation (no Euler
  bias), plus memory-light terminal-value sampling for pricing.
- **Option pricing** — closed-form **Black-Scholes** and a risk-neutral **Monte
  Carlo** pricer with **antithetic variates** and a reported standard error.
- **Markov chains** — discrete-state simulation from a validated transition
  matrix, with an analytic stationary-distribution helper.
- **Statistics** — mean/variance/std, covariance & correlation matrices, OLS
  regression with R², and confidence intervals — all in C++, all checked
  against NumPy.
- **Engineering** — deterministic RNG streams, strict input validation, a
  100+-test pytest suite, benchmark scripts, runnable examples, and CI on Linux
  and macOS across Python 3.10–3.13.

## Why it's built this way

The project is a study in **bridging Python and C++ cleanly**:

- The C++ compute functions are **independent of pybind11** — they take and
  return plain `std::vector`s. A single boundary file (`binding.cpp`) handles
  all NumPy ↔ C++ conversion, so the math stays testable and portable.
- Randomness is **deterministic and thread-safe**: each thread derives its own
  RNG stream from `(seed, thread_id)` and writes into a disjoint slice of the
  output, so results are reproducible for a given `(seed, n_threads)` and the
  single-threaded result is reproduced exactly when `n_threads = 1`.
- Every numerical routine is **validated against a trusted reference** (NumPy or
  a closed-form result), not just against itself.

---

## Installation

Requires a C++17 compiler, CMake, and Python ≥ 3.10. The build (CMake +
pybind11) is driven by `scikit-build-core`, so a normal pip install just works:

```bash
git clone https://github.com/RonithManikonda/quantcore.git
cd quantcore

python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"     # builds the C++ extension + dev tools

pytest -q                              # run the test suite
```

---

## Quickstart

### Brownian motion (single- and multithreaded)

```python
import quantcore as qc

# 1,000 paths, 252 daily steps, 20% vol — shape (1000, 253), every path starts at 0
paths = qc.brownian_paths(n_steps=252, n_paths=1_000, dt=1/252, sigma=0.2, seed=42)

# Same contract, parallelised across path batches (0 = auto-detect cores)
fast = qc.brownian_paths_parallel(n_steps=252, n_paths=1_000_000, dt=1/252,
                                  sigma=0.2, seed=42, n_threads=8)

# Raw increments dW ~ N(0, sigma^2 dt); cumsum + a leading 0 reconstructs a path
inc = qc.brownian_increments(n_steps=252, n_paths=1_000, dt=1/252, seed=42)
```

### Geometric Brownian motion

```python
# Asset-price paths: dS = mu S dt + sigma S dW, starting at s0, always positive
prices = qc.gbm_paths(n_steps=252, n_paths=10_000, dt=1/252,
                      mu=0.08, sigma=0.2, s0=100.0, seed=0)

# Terminal prices only, drawn straight from the lognormal law (memory-light)
terminal = qc.gbm_terminal(n_paths=1_000_000, T=1.0, mu=0.08, sigma=0.2, s0=100.0)
```

### Option pricing

```python
call = qc.black_scholes(s0=100, K=105, r=0.05, sigma=0.2, T=1.0, option_type="call")
put  = qc.black_scholes(s0=100, K=105, r=0.05, sigma=0.2, T=1.0, option_type="put")

res = qc.mc_european(s0=100, K=105, r=0.05, sigma=0.2, T=1.0,
                     n_paths=500_000, option_type="call", antithetic=True)
print(res.price, res.std_error)   # MCResult(price=..., std_error=...)
```

### Markov chains

```python
import numpy as np

P = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.4, 0.3],
              [0.2, 0.45, 0.35]])

chains = qc.markov_chain(P, n_steps=1_000, n_chains=500, start_state=0, seed=7)
pi = qc.stationary_distribution(P)   # long-run state probabilities
```

### Statistics

```python
import numpy as np
rng = np.random.default_rng(0)

X = rng.normal(size=(1000, 3))
y = X @ [1.5, -2.0, 0.5] + 0.1 * rng.normal(size=1000)

cov  = qc.covariance(X)              # (3, 3) covariance matrix
corr = qc.correlation(X)             # (3, 3) correlation matrix
fit  = qc.linear_regression(X, y)    # RegressionResult(intercept, coefficients, r_squared)
ci   = qc.confidence_interval(y, confidence=0.95)   # ConfidenceInterval(low, high)
```

Runnable versions of all of the above live in [`examples/`](examples/).

---

## Performance

Measured on an Apple M4 Pro (14 cores) from Python — full details and tables in
[**BENCHMARKS.md**](BENCHMARKS.md).

| Workload | Result |
|---|---|
| Brownian paths, parallel scaling (compute-heavy) | **5.3× on 8 threads** |
| Monte Carlo pricing throughput | **~77–90M paths/second** |
| MC error vs Black-Scholes (10M paths) | within **0.003** |
| Antithetic variates | **~1.42×** lower standard error (≈ 2× the paths, free) |

The benchmarks deliberately include a case where threading *doesn't* help (tiny
workloads, where launch overhead dominates) — knowing when not to parallelise is
part of the point.

---

## Project layout

```
quantcore/
├── src/
│   ├── cpp/                    # C++ compute kernels (kept pybind11-free)
│   │   ├── binding.cpp         #   the single pybind11 boundary (all NumPy <-> C++)
│   │   ├── brownian.{hpp,cpp}  #   Brownian motion (serial + parallel)
│   │   ├── gbm.{hpp,cpp}       #   geometric Brownian motion
│   │   ├── pricing.{hpp,cpp}   #   Black-Scholes + Monte Carlo
│   │   ├── markov.{hpp,cpp}    #   Markov chains
│   │   ├── stats.{hpp,cpp}     #   covariance, correlation, OLS, CI
│   │   ├── rng.hpp             #   seeded + per-thread RNG streams
│   │   ├── parallel.hpp        #   thread-count policy
│   │   ├── special.hpp         #   normal CDF / inverse CDF
│   │   └── utils.hpp           #   input validation
│   └── python/quantcore/       # thin, typed Python API over _core
│       ├── stochastic.py       #   Brownian + GBM
│       ├── pricing.py · markov.py · stats.py
│       └── py.typed
├── tests/                      # pytest suite (104 tests)
├── benchmarks/                 # bench_brownian · bench_parallel · bench_pricing
├── examples/                   # runnable demos for every feature
├── CMakeLists.txt · pyproject.toml
├── BENCHMARKS.md · CHANGELOG.md · LICENSE
```

## API conventions

- **Shapes.** Simulators return `(n_paths, n_steps + 1)` for paths (column 0 is
  the start value) and `(n_paths, n_steps)` for increments. Everything is
  `float64` NumPy, except Markov state trajectories which are integer.
- **Seeds.** Every stochastic function takes a `seed`; identical arguments
  produce identical output. Parallel runs are deterministic per `(seed,
  n_threads)`.
- **Validation.** Invalid inputs (non-positive `dt`/`steps`/`paths`, negative
  volatility, a transition matrix whose rows don't sum to 1, …) raise
  `ValueError` with a message naming the offending parameter.
- **Threads.** `n_threads = 0` auto-detects the core count and is capped at
  `n_paths`.

## Testing

```bash
pytest -q                  # all 104 tests
pytest tests/test_pricing.py -v
```

Tests cover output shapes, seed reproducibility, input validation, statistical
properties (variance growth, lognormal moments, transition frequencies), and
direct cross-checks against NumPy (`cov`, `corrcoef`, `lstsq`) and closed-form
Black-Scholes values.

## Roadmap

- Variance-reduction beyond antithetics (control variates, QMC / Sobol)
- American / path-dependent payoffs (Longstaff–Schwartz)
- Additional processes (Ornstein–Uhlenbeck, Heston stochastic volatility)
- Parallelised Markov simulation across chains

## License

[MIT](LICENSE) © Ronith Manikonda
