"""
Markov chain demo — quantcore Phase 7

Simulates a small weather-style Markov chain and checks that the long-run
visit frequencies converge to the analytic stationary distribution.

Run with:
    python examples/markov_demo.py
"""

import numpy as np

from quantcore import markov_chain, stationary_distribution

SEPARATOR = "\n" + "─" * 60

STATES = ["Sunny", "Cloudy", "Rainy"]

# Row i = distribution over tomorrow's weather given today is state i.
P = np.array([
    [0.70, 0.20, 0.10],   # Sunny  -> ...
    [0.30, 0.40, 0.30],   # Cloudy -> ...
    [0.20, 0.45, 0.35],   # Rainy  -> ...
])

# ---------------------------------------------------------------------------
# 1. A single simulated trajectory
# ---------------------------------------------------------------------------

print(SEPARATOR)
print("1. ONE SIMULATED TRAJECTORY (first 20 days, starting Sunny)")
print(SEPARATOR)

trajectory = markov_chain(P, n_steps=19, n_chains=1, start_state=0, seed=7)[0]
print("  " + " ".join(STATES[s][:3] for s in trajectory))

# ---------------------------------------------------------------------------
# 2. Long-run visit frequencies vs the stationary distribution
# ---------------------------------------------------------------------------

print(SEPARATOR)
print("2. EMPIRICAL FREQUENCIES vs STATIONARY DISTRIBUTION")
print(SEPARATOR)

chain = markov_chain(P, n_steps=1_000_000, n_chains=1, start_state=0, seed=123)[0]
empirical = np.array([np.mean(chain == s) for s in range(len(STATES))])
pi = stationary_distribution(P)

print(f"  Simulated {chain.size - 1:,} transitions.")
print()
print(f"  {'state':<10}{'empirical':>12}{'stationary':>12}")
print(f"  {'-' * 34}")
for name, emp, stat in zip(STATES, empirical, pi):
    print(f"  {name:<10}{emp:>12.4f}{stat:>12.4f}")

print()
print("  The chain forgets its starting state and settles into the stationary")
print("  distribution — the long-run fraction of days in each weather state.")

# ---------------------------------------------------------------------------
# 3. Many independent chains in parallel state space
# ---------------------------------------------------------------------------

print(SEPARATOR)
print("3. DISTRIBUTION ACROSS MANY CHAINS AT A FIXED HORIZON")
print(SEPARATOR)

chains = markov_chain(P, n_steps=30, n_chains=50_000, start_state=0, seed=0)
day30 = chains[:, -1]
dist = np.array([np.mean(day30 == s) for s in range(len(STATES))])

print(f"  State distribution across 50,000 chains after 30 steps:")
for name, frac in zip(STATES, dist):
    bar = "█" * int(frac * 40)
    print(f"    {name:<8}{frac:>7.4f}  {bar}")
