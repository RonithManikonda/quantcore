"""
European option pricing: closed-form Black-Scholes and Monte Carlo.

The two agree by construction — the Monte Carlo estimator simulates terminal
prices under the risk-neutral measure and discounts the expected payoff, which
is exactly what the Black-Scholes formula evaluates in closed form. Comparing
them is the headline correctness check for the simulation engine.
"""

from collections import namedtuple

from quantcore._core import black_scholes_price as _black_scholes_price
from quantcore._core import mc_european_price as _mc_european_price

# A Monte Carlo price comes with its own sampling uncertainty, so we return both
# the estimate and its standard error rather than a bare float.
MCResult = namedtuple("MCResult", ["price", "std_error"])

_OPTION_TYPES = {"call": True, "put": False}


def _is_call(option_type: str) -> bool:
    try:
        return _OPTION_TYPES[option_type.lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"option_type must be 'call' or 'put', got {option_type!r}"
        ) from None


def black_scholes(
    s0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
) -> float:
    """
    Closed-form Black-Scholes price of a European option.

    Parameters
    ----------
    s0 : float
        Spot price. Must be > 0.
    K : float
        Strike price. Must be > 0.
    r : float
        Continuously compounded risk-free rate. May be negative.
    sigma : float
        Volatility. Must be >= 0.
    T : float
        Time to expiry, in years. Must be > 0.
    option_type : str
        ``"call"`` or ``"put"``. Default ``"call"``.

    Returns
    -------
    float
        The option's present value.
    """
    return _black_scholes_price(s0, K, r, sigma, T, _is_call(option_type))


def mc_european(
    s0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int = 100_000,
    option_type: str = "call",
    seed: int = 0,
    antithetic: bool = True,
) -> MCResult:
    """
    Monte Carlo price of a European option under the risk-neutral measure.

    Simulates ``n_paths`` terminal prices from the GBM lognormal law (drift
    ``r``), averages the discounted payoff, and reports the Monte Carlo
    standard error alongside the estimate.

    Parameters
    ----------
    s0 : float
        Spot price. Must be > 0.
    K : float
        Strike price. Must be > 0.
    r : float
        Continuously compounded risk-free rate. May be negative.
    sigma : float
        Volatility. Must be >= 0.
    T : float
        Time to expiry, in years. Must be > 0.
    n_paths : int
        Number of simulated terminal prices. Must be > 0. Default 100,000.
    option_type : str
        ``"call"`` or ``"put"``. Default ``"call"``.
    seed : int
        RNG seed. Default 0.
    antithetic : bool
        Use antithetic variates (pairs of Z, -Z) for variance reduction.
        With antithetic sampling ``n_paths`` is split into ``n_paths // 2``
        independent pairs. Default True.

    Returns
    -------
    MCResult
        Named tuple ``(price, std_error)``. The Black-Scholes price should sit
        within a few standard errors of ``price``.
    """
    price, std_error = _mc_european_price(
        s0, K, r, sigma, T, n_paths, _is_call(option_type), seed, antithetic)
    return MCResult(price=price, std_error=std_error)
