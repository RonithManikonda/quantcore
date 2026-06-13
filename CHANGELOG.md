# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/), and the project follows
the milestone plan it was built against (foundation → bindings → numerics →
randomness → parallelism → broader models → polish).

## [0.1.0] — 2026

First complete release. Every milestone from the project plan is implemented,
tested, and benchmarked.

### Added

**Packaging and bindings**
- `scikit-build-core` + `pybind11` build producing the `quantcore._core`
  extension, installable in editable mode (`pip install -e .`).
- Clean Python API layer over the C++ kernels; `py.typed` marker so the
  package ships type information.

**Brownian motion (single-threaded)**
- `brownian_paths` and `brownian_increments` with a seeded `mt19937_64` RNG,
  full input validation, and a documented `(n_paths, n_steps[+1])` output
  contract.

**Parallelism**
- `brownian_paths_parallel` / `brownian_increments_parallel`: work split across
  path batches, each thread with a private RNG derived from `(seed, thread_id)`
  writing into a disjoint output slice. Deterministic for a given
  `(seed, n_threads)`; `n_threads=1` reproduces the serial result bit-for-bit.

**Geometric Brownian motion**
- `gbm_paths` (exact lognormal discretisation), `gbm_paths_parallel`, and
  `gbm_terminal` for drawing `S(T)` directly from the closed-form law.

**Option pricing**
- `black_scholes` closed-form European call/put pricing.
- `mc_european` risk-neutral Monte Carlo pricer with antithetic variates and a
  reported standard error.

**Markov chains**
- `markov_chain` discrete-state simulator taking a transition matrix as input
  (validated square / non-negative / row-stochastic).
- `stationary_distribution` analytic companion (left eigenvector for
  eigenvalue 1).

**Statistics**
- `mean`, `variance`, `std`, `covariance`, `correlation`,
  `linear_regression` (OLS with R²), and `confidence_interval`, all taking
  NumPy array input and validated against NumPy references.

**Tooling**
- pytest suite (100+ tests) covering shapes, reproducibility, validation,
  statistical properties, and cross-checks against NumPy and closed-form
  results.
- Benchmark scripts for single-threaded scaling, parallel speedup, and Monte
  Carlo pricing throughput; results recorded in
  [`BENCHMARKS.md`](BENCHMARKS.md).
- GitHub Actions CI building and testing on Linux and macOS across Python
  3.10–3.13.
- Runnable examples for every part of the public API.

[0.1.0]: https://github.com/RonithManikonda/quantcore/releases/tag/v0.1.0
