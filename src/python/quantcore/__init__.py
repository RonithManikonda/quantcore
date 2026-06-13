from .stochastic import (
    brownian_paths,
    brownian_increments,
    brownian_paths_parallel,
    brownian_increments_parallel,
    gbm_paths,
    gbm_paths_parallel,
    gbm_terminal,
)
from .pricing import (
    black_scholes,
    mc_european,
    MCResult,
)
from .markov import (
    markov_chain,
    stationary_distribution,
)
from .stats import (
    mean,
    variance,
    std,
    covariance,
    correlation,
    linear_regression,
    confidence_interval,
    RegressionResult,
    ConfidenceInterval,
)

__version__ = "0.1.0"

__all__ = [
    # Brownian motion
    "brownian_paths",
    "brownian_increments",
    "brownian_paths_parallel",
    "brownian_increments_parallel",
    # Geometric Brownian motion
    "gbm_paths",
    "gbm_paths_parallel",
    "gbm_terminal",
    # Option pricing
    "black_scholes",
    "mc_european",
    "MCResult",
    # Markov chains
    "markov_chain",
    "stationary_distribution",
    # Statistics
    "mean",
    "variance",
    "std",
    "covariance",
    "correlation",
    "linear_regression",
    "confidence_interval",
    "RegressionResult",
    "ConfidenceInterval",
    "__version__",
]
