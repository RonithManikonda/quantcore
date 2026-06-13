import numpy as np

from quantcore._core import simulate_paths as _simulate_paths
from quantcore._core import simulate_increments as _simulate_increments
from quantcore._core import simulate_paths_parallel as _simulate_paths_parallel
from quantcore._core import simulate_increments_parallel as _simulate_increments_parallel


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


def brownian_paths_parallel(
    n_steps: int,
    n_paths: int,
    dt: float,
    sigma: float = 1.0,
    seed: int = 0,
    n_threads: int = 0,
) -> np.ndarray:
    """
    Multithreaded Brownian motion paths.

    Same output contract as brownian_paths — identical shape, dtype, and
    starting condition. Results are deterministic for a given (seed, n_threads)
    pair but will differ from the single-threaded version because each thread
    uses its own derived RNG stream.

    Parameters
    ----------
    n_steps : int
        Number of time steps per path. Must be > 0.
    n_paths : int
        Number of independent paths. Must be > 0.
    dt : float
        Length of each time step. Must be > 0.
    sigma : float
        Diffusion coefficient. Must be >= 0. Default 1.0.
    seed : int
        Base RNG seed. Each thread derives its own seed from this. Default 0.
    n_threads : int
        Number of worker threads. 0 = auto-detect hardware concurrency. Default 0.

    Returns
    -------
    np.ndarray
        Shape ``(n_paths, n_steps + 1)``, dtype float64.
    """
    return _simulate_paths_parallel(n_steps, n_paths, dt, sigma, seed, n_threads)


def brownian_increments_parallel(
    n_steps: int,
    n_paths: int,
    dt: float,
    sigma: float = 1.0,
    seed: int = 0,
    n_threads: int = 0,
) -> np.ndarray:
    """
    Multithreaded Brownian motion increments.

    Same output contract as brownian_increments.

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
        Base RNG seed. Default 0.
    n_threads : int
        Number of worker threads. 0 = auto-detect. Default 0.

    Returns
    -------
    np.ndarray
        Shape ``(n_paths, n_steps)``, dtype float64.
    """
    return _simulate_increments_parallel(n_steps, n_paths, dt, sigma, seed, n_threads)
