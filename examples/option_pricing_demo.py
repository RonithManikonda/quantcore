"""
European option pricing demo — quantcore Phase 7

Compares the closed-form Black-Scholes price with the Monte Carlo estimator,
showing convergence, the standard-error report, and the effect of antithetic
variates.

Run with:
    python examples/option_pricing_demo.py
"""

import math

from quantcore import black_scholes, mc_european

SEPARATOR = "\n" + "─" * 60

# Standard contract: at-the-money, 5% rate, 20% vol, one year.
S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0

# ---------------------------------------------------------------------------
# 1. Closed-form prices and put-call parity
# ---------------------------------------------------------------------------

print(SEPARATOR)
print("1. BLACK-SCHOLES CLOSED FORM")
print(SEPARATOR)

call = black_scholes(S0, K, R, SIGMA, T, "call")
put = black_scholes(S0, K, R, SIGMA, T, "put")

print(f"  Contract: S0={S0}, K={K}, r={R}, sigma={SIGMA}, T={T}")
print(f"  Call price : {call:.6f}")
print(f"  Put price  : {put:.6f}")
print()
parity_lhs = call - put
parity_rhs = S0 - K * math.exp(-R * T)
print(f"  Put-call parity  C - P = S0 - K*exp(-rT):")
print(f"    {parity_lhs:.6f}  ==  {parity_rhs:.6f}")

# ---------------------------------------------------------------------------
# 2. Monte Carlo convergence to the closed form
# ---------------------------------------------------------------------------

print(SEPARATOR)
print("2. MONTE CARLO CONVERGENCE  (call option)")
print(SEPARATOR)

print(f"  Black-Scholes reference: {call:.6f}")
print()
print(f"  {'n_paths':>10}{'MC price':>12}{'std error':>12}{'|error|':>12}")
print(f"  {'-' * 46}")
for n in [1_000, 10_000, 100_000, 1_000_000]:
    res = mc_european(S0, K, R, SIGMA, T, n_paths=n, option_type="call",
                      seed=2024, antithetic=True)
    print(f"  {n:>10,}{res.price:>12.6f}{res.std_error:>12.6f}"
          f"{abs(res.price - call):>12.6f}")

print()
print("  The estimate tightens like 1/sqrt(n): 100x more paths -> ~10x smaller")
print("  standard error.")

# ---------------------------------------------------------------------------
# 3. Variance reduction with antithetic variates
# ---------------------------------------------------------------------------

print(SEPARATOR)
print("3. ANTITHETIC VARIATES  (same path budget)")
print(SEPARATOR)

n = 200_000
plain = mc_european(S0, K, R, SIGMA, T, n_paths=n, seed=99, antithetic=False)
anti = mc_european(S0, K, R, SIGMA, T, n_paths=n, seed=99, antithetic=True)

print(f"  n_paths = {n:,}")
print(f"  {'method':<16}{'price':>12}{'std error':>12}")
print(f"  {'-' * 40}")
print(f"  {'plain':<16}{plain.price:>12.6f}{plain.std_error:>12.6f}")
print(f"  {'antithetic':<16}{anti.price:>12.6f}{anti.std_error:>12.6f}")
print()
reduction = plain.std_error / anti.std_error
print(f"  Antithetic sampling cuts the standard error by {reduction:.2f}x")
print(f"  for free — equivalent to ~{reduction ** 2:.1f}x as many plain paths.")
