"""
Minimal demo of the quantcore Brownian motion API.
"""

import numpy as np
from quantcore.stochastic import brownian_paths, brownian_increments

# --- Basic usage --------------------------------------------------------

paths = brownian_paths(n_steps=252, n_paths=5, dt=1 / 252, sigma=0.2, seed=42)
print(f"paths shape : {paths.shape}")          # (5, 253)
print(f"starts at 0 : {np.all(paths[:, 0] == 0.0)}")
print(f"final values: {np.round(paths[:, -1], 4)}")

# --- Seed reproducibility -----------------------------------------------

a = brownian_paths(n_steps=100, n_paths=3, dt=0.01, seed=7)
b = brownian_paths(n_steps=100, n_paths=3, dt=0.01, seed=7)
print(f"\nsame seed → identical output: {np.array_equal(a, b)}")

# --- Variance grows linearly with time: Var(W(t)) ≈ sigma^2 * t --------

sigma, dt = 1.0, 0.01
paths = brownian_paths(n_steps=100, n_paths=5000, dt=dt, sigma=sigma, seed=0)
print("\nVariance check (sigma=1, dt=0.01):")
for k in [10, 50, 100]:
    t = k * dt
    print(f"  Var(W({t:.2f}))  measured={paths[:, k].var():.4f}  expected={t:.4f}")

# --- Relationship between paths and increments --------------------------

inc = brownian_increments(n_steps=50, n_paths=4, dt=0.01, seed=99)
reconstructed = np.concatenate([np.zeros((4, 1)), np.cumsum(inc, axis=1)], axis=1)
ref = brownian_paths(n_steps=50, n_paths=4, dt=0.01, seed=99)
print(f"\ncumsum(increments) == paths: {np.allclose(reconstructed, ref)}")
