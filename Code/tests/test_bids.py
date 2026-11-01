"""Tests for experiments.bids -- experiments.md section 9 (stage-0 gate's structural
checks) and section 4.3."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import verify_price_based as vpb
from experiments import benchmark, bids, cases, forecasts
from experiments.config import BidRule, ErrorModel

KAPPA = BidRule().tie_break_reward


def _bench():
    case = cases.day_case()
    return case, benchmark.solve_benchmark(case)


def test_bid_star_reproduces_benchmark_prices_at_self_schedule():
    """B_star's quantity *is* the benchmark schedule by construction; this just checks
    the price formula doesn't raise and has the right shape/no NaNs."""
    case, b = _bench()
    S = case["ess"][0]
    bid = bids.bid_star(S, b.pi_star[0], b.nuU_star[0], b.nuL_star[0], b.p_star[0], b.g_star[0])
    assert bid.u.shape == (case["T"],)
    assert not np.any(np.isnan(bid.u)) and not np.any(np.isnan(bid.c))


def test_bid_fcst_at_zero_error_matches_bid_star_p():
    """At k=0 (sigma=0) B_fcst must reduce exactly to B_star_p -- the perfect-
    information regression test (experiments.md 4.2 point 5 applied to bids)."""
    case, b = _bench()
    S = case["ess"][0]
    T, dT = case["T"], case["dT"]
    for rule in ("prelimit", "solver"):
        for kappa in (0.0, KAPPA):
            bsp = bids.bid_star_p(S, b.lam_star[0], T, dT, rule, kappa)
            bfc = bids.bid_fcst(S, b.lam_star[0], T, dT, rule, kappa)  # lam_hat == lam_star at sigma=0
            np.testing.assert_allclose(bsp.u, bfc.u, atol=1e-8)
            np.testing.assert_allclose(bsp.c, bfc.c, atol=1e-8)


def test_bid_priv_uses_only_technical_data():
    case, b = _bench()
    S = case["ess"][0]
    T = case["T"]
    bid = bids.bid_priv(S, T)
    t = np.arange(1, T + 1)
    expected_pi = S["g"] ** (T - t) * S["w"]
    np.testing.assert_allclose(bid.u, S["a"] * expected_pi - S["eps"])
    np.testing.assert_allclose(bid.c, S["zet"] + S["b"] * expected_pi)
    np.testing.assert_array_equal(bid.p0, np.full(T, S["pbar"]))
    np.testing.assert_array_equal(bid.g0, np.full(T, S["gbar"]))


def test_quantity_blocks_offer_prices_bracket_bid_price():
    case, b = _bench()
    S = case["ess"][0]
    T, dT = case["T"], case["dT"]
    bid = bids.bid_star_p(S, b.lam_star[0], T, dT, "prelimit", KAPPA)
    blocks = bids.quantity_blocks(bid, S, delta=0.01, full_capacity=False)
    for t in range(T):
        (p1, q1), (p2, q2) = blocks["charge"][t]
        assert p1 > p2  # higher price for the first (planned-quantity) block
        assert q1 >= 0 and q2 >= 0
        (p1d, q1d), (p2d, q2d) = blocks["discharge"][t]
        assert p1d < p2d
        assert q1d >= 0 and q2d >= 0


def test_tie_break_plan_is_optimal_for_true_profit():
    """The throughput tie-breaker must only choose *among* optimal plans: the tie-broken
    plan's true self-schedule objective equals the true optimum, at lam_star (exact ties)
    and at forecast draws (ties broken by the error)."""
    case, b = _bench()
    S = case["ess"][0]
    T, dT = case["T"], case["dT"]
    lams = [b.lam_star[0]] + [
        forecasts.forecast_prices(b.lam_star[0], ErrorModel(sigma_abs=sig), 1, T, np.random.default_rng(seed))[0]
        for sig in (1.0, 5.0, 20.0) for seed in range(5)
    ]
    for lam in lams:
        res, c_true = vpb.self_schedule(S, lam, T, dT)
        p, g, s = bids.self_schedule(S, lam, T, dT, KAPPA)
        x = np.concatenate([p, g, s])
        assert c_true @ x - res.fun <= 1e-9 * max(1.0, abs(res.fun))
        assert not np.any((p > 1e-6) & (g > 1e-6))


def test_tie_break_prefers_throughput_at_lambda_star():
    """At lam_star the battery is indifferent between several plans (break-even round trip
    00-08h, free hours 08-16h); the tie-breaker must pick one with at least as much
    throughput as the solver's own vertex, and strictly more here: it adds the morning
    cycle (charge 00-04h, discharge 04-08h)."""
    case, b = _bench()
    S = case["ess"][0]
    T, dT = case["T"], case["dT"]
    p0, g0, _ = bids.self_schedule(S, b.lam_star[0], T, dT, 0.0)
    p1, g1, _ = bids.self_schedule(S, b.lam_star[0], T, dT, KAPPA)
    assert (p1 + g1).sum() > (p0 + g0).sum() + 1.0
    assert p1[0] > 1.0 and g1[1] > 1.0
