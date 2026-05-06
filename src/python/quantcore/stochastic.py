import numpy as np

from quantcore._core import simulate_paths as _simulate_paths
from quantcore._core import simulate_increments as _simulate_increments


def brownian_paths(
    n_steps: int,
    n_paths: int,
    dt: float,
    sigma: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Simulate standard Brownian motion paths.

    Parameters
    ----------
    n_steps : int
        Number of time steps per path. Must be > 0.
    n_paths : int
        Number of independent paths to simulate. Must be > 0.
    dt : float
        Length of each time step (e.g. 1/252 for daily steps in a year). Must be > 0.
    sigma : float
        Diffusion coefficient / volatility. Must be >= 0. Default 1.0.
    seed : int
        RNG seed. The same seed + same arguments always produce identical output. Default 0.

    Returns
    -------
    np.ndarray
        Shape ``(n_paths, n_steps + 1)``, dtype float64.
        Each row is one path. Column 0 is always 0.0 (W(0) = 0).
    """
    return _simulate_paths(n_steps, n_paths, dt, sigma, seed)


def brownian_increments(
    n_steps: int,
    n_paths: int,
    dt: float,
    sigma: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Simulate Brownian motion increments dW ~ N(0, sigma^2 * dt).

    Parameters
    ----------
    n_steps : int
        Number of increments per path. Must be > 0.
    n_paths : int
        Number of independent paths. Must be > 0.
    dt : float
        Time step size. Must be > 0.
    sigma : float
        Diffusion coefficient. Must be >= 0. Default 1.0.
    seed : int
        RNG seed. Default 0.

    Returns
    -------
    np.ndarray
        Shape ``(n_paths, n_steps)``, dtype float64.
        ``np.cumsum(result, axis=1)`` prepended with zeros gives ``brownian_paths``.
    """
    return _simulate_increments(n_steps, n_paths, dt, sigma, seed)
