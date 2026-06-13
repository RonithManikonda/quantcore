"""
Tests for European option pricing (Phase 7).

The closed-form Black-Scholes price anchors correctness:
  - it must reproduce textbook values and satisfy put-call parity;
  - the Monte Carlo estimate must converge to it within a few standard errors;
  - antithetic variates must actually reduce the Monte Carlo standard error.
"""

import math

import pytest

from quantcore import black_scholes, mc_european


# Standard textbook contract used throughout.
ATM = dict(s0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)


# ---------------------------------------------------------------------------
# Black-Scholes closed form
# ---------------------------------------------------------------------------

def test_black_scholes_known_values():
    # Textbook values for s0=K=100, r=5%, sigma=20%, T=1.
    call = black_scholes(option_type="call", **ATM)
    put = black_scholes(option_type="put", **ATM)
    assert call == pytest.approx(10.450583572, abs=1e-6)
    assert put == pytest.approx(5.573526022, abs=1e-6)


def test_put_call_parity():
    # C - P = s0 - K * exp(-r T)
    call = black_scholes(option_type="call", **ATM)
    put = black_scholes(option_type="put", **ATM)
    lhs = call - put
    rhs = ATM["s0"] - ATM["K"] * math.exp(-ATM["r"] * ATM["T"])
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_zero_vol_is_discounted_intrinsic_of_forward():
    # With sigma=0 the call is worth disc * max(forward - K, 0).
    s0, K, r, T = 100.0, 90.0, 0.05, 1.0
    price = black_scholes(s0=s0, K=K, r=r, sigma=0.0, T=T, option_type="call")
    fwd = s0 * math.exp(r * T)
    expected = math.exp(-r * T) * max(fwd - K, 0.0)
    assert price == pytest.approx(expected, abs=1e-9)


def test_deep_itm_call_approaches_forward_minus_pv_strike():
    # A deep in-the-money call behaves almost like a forward.
    price = black_scholes(s0=200.0, K=100.0, r=0.05, sigma=0.2, T=1.0,
                          option_type="call")
    lower_bound = 200.0 - 100.0 * math.exp(-0.05)  # no-arbitrage lower bound
    assert price >= lower_bound - 1e-9


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_bs_invalid_inputs_raise(option_type):
    with pytest.raises((ValueError, RuntimeError)):
        black_scholes(s0=-1.0, K=100.0, r=0.05, sigma=0.2, T=1.0,
                      option_type=option_type)


def test_bad_option_type_raises():
    with pytest.raises(ValueError):
        black_scholes(option_type="straddle", **ATM)


# ---------------------------------------------------------------------------
# Monte Carlo convergence to Black-Scholes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mc_converges_to_black_scholes(option_type):
    exact = black_scholes(option_type=option_type, **ATM)
    mc = mc_european(n_paths=400_000, option_type=option_type, seed=12345,
                     antithetic=True, **ATM)
    # estimate must sit within ~4 standard errors of the closed form
    assert abs(mc.price - exact) < 4.0 * mc.std_error
    assert abs(mc.price - exact) < 0.05


def test_mc_standard_error_shrinks_with_more_paths():
    small = mc_european(n_paths=10_000, seed=1, antithetic=False, **ATM)
    large = mc_european(n_paths=160_000, seed=1, antithetic=False, **ATM)
    # 16x the paths should roughly quarter the standard error (1/sqrt(n))
    assert large.std_error < small.std_error
    ratio = small.std_error / large.std_error
    assert 3.0 < ratio < 5.0


def test_antithetic_reduces_standard_error():
    plain = mc_european(n_paths=100_000, seed=7, antithetic=False, **ATM)
    anti = mc_european(n_paths=100_000, seed=7, antithetic=True, **ATM)
    assert anti.std_error < plain.std_error


def test_mc_invalid_inputs_raise():
    with pytest.raises((ValueError, RuntimeError)):
        mc_european(n_paths=0, **ATM)
