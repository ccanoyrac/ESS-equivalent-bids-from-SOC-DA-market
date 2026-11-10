"""Tests for experiments.market / experiments.realtime / the full pipeline -- the
stage-0 gate's core checks (experiments.md section 5 "Stage 0 gate")."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments import benchmark, bids, cases, market, realtime, runner
from experiments.config import BidRule, ErrorModel, RealTime, RunSettings


def _bench():
    case = cases.day_case()
    return case, benchmark.solve_benchmark(case)


def test_b_star_reproduces_baldick_dispatch_exactly():
    """B_star's bids must clear to exactly Baldick's own (p, g); the only expected
    discrepancy is the clearing price landing within the quantity rule's own `delta`
    spread of lambda_star."""
    case, b = _bench()
    S = case["ess"][0]
    bid = bids.bid_star(S, b.pi_star[0], b.nuU_star[0], b.nuL_star[0], b.p_star[0], b.g_star[0])
    blocks = bids.quantity_blocks(bid, S, delta=0.01, full_capacity=False)
    res = market.clear_cd(case, [blocks], voll=1000.0)

    np.testing.assert_allclose(res.p_da[0], b.p_star[0], atol=1e-6)
    np.testing.assert_allclose(res.g_da[0], b.g_star[0], atol=1e-6)
    assert np.max(np.abs(res.lam_cd - b.lam_star[0])) <= 0.01 + 1e-9
    assert np.all(res.shed < 1e-9)


def test_no_simultaneous_charge_and_discharge():
    """experiments.md 4.1: 'curtailment must be explicit'; the storage analogue is
    checked here directly for every bid family -- never both p>0 and g>0 in the same
    hour."""
    case, b = _bench()
    S = case["ess"][0]
    T, dT = case["T"], case["dT"]
    families = {
        "B_star": bids.bid_star(S, b.pi_star[0], b.nuU_star[0], b.nuL_star[0], b.p_star[0], b.g_star[0]),
        "B_star_p": bids.bid_star_p(S, b.lam_star[0], T, dT, "prelimit", BidRule().tie_break_reward),
        "B_priv": bids.bid_priv(S, T),
    }
    for name, bid in families.items():
        blocks = bids.quantity_blocks(bid, S, delta=0.01, full_capacity=(name == "B_priv"))
        res = market.clear_cd(case, [blocks], voll=1000.0)
        simultaneous = (res.p_da[0] > 1e-6) & (res.g_da[0] > 1e-6)
        assert not np.any(simultaneous), f"{name}: simultaneous charge+discharge at hours {np.where(simultaneous)[0]}"


def test_feasible_schedule_is_a_noop_on_feasible_input():
    case, b = _bench()
    S = case["ess"][0]
    T, dT = case["T"], case["dT"]
    rep = realtime.feasible_schedule(S, b.p_star[0], b.g_star[0], T, dT)
    assert rep.undeliverable_mwh < 1e-8
    np.testing.assert_allclose(rep.p, b.p_star[0], atol=1e-6)
    np.testing.assert_allclose(rep.g, b.g_star[0], atol=1e-6)


def test_stage0_b_star_gate():
    """The stage-0 gate's first, hardest requirement: B_star gives dSW within 1e-6 of 0
    (experiments.md section 5)."""
    case, bench = _bench()
    settings = RunSettings(
        error=ErrorModel(), bid=BidRule(family="B_star"), realtime=RealTime(psi=0.0),
        draws=1, setting_id="gate_B_star",
    )
    row = runner.run_draw(case, bench, settings, draw=0)
    assert abs(row["dSW"]) < 1e-6


def test_dsw_never_negative():
    """experiments.md 4.7: dSW = SW_star - SW_real must be >= -1e-6 for every family."""
    case, bench = _bench()
    for family, rule in [("B_star_p", "prelimit"), ("B_fcst", "solver"), ("B_priv", "prelimit")]:
        error = ErrorModel(sigma_abs=5.0) if family == "B_fcst" else ErrorModel()
        settings = RunSettings(
            error=error, bid=BidRule(family=family, rule=rule), realtime=RealTime(psi=10.0),
            draws=3, base_seed=1, setting_id=f"dsw_check_{family}",
        )
        df = runner.run_setting(case, bench, settings)
        assert (df["dSW"] >= -1e-6).all(), f"{family}: negative dSW found\n{df}"


def test_stage1_two_batteries_b_star_gate():
    """Theorem 1 with two batteries: Baldick's own duals and quantities for both give dSW = 0, and
    with sigma = sigma_w = 0 B_fcst is exactly B_star_p."""
    case = cases.stage1_case()
    bench = benchmark.solve_benchmark(case)
    rows = {}
    for fam in ("B_star", "B_star_p", "B_fcst"):
        st = RunSettings(error=ErrorModel(sigma_abs=0.0, sigma_w=0.0), bid=BidRule(family=fam),
                         realtime=RealTime(psi=10.0), n_ess=2, draws=1, setting_id="s1_test")
        rows[fam] = runner.run_draw(case, bench, st, 0)
    assert abs(rows["B_star"]["dSW"]) < 1e-6
    assert abs(rows["B_fcst"]["dSW"] - rows["B_star_p"]["dSW"]) < 1e-9


def test_ess_profit_settles_deviations_with_a_psi_penalty():
    """welfare.ess_profit: B_star delivers what it sold, so it pays no penalty and earns the
    benchmark battery's profit (to the quantity rule's delta); B_priv's undeliverable energy
    is charged psi per MWh, and its components add up."""
    from experiments import welfare
    case, b = _bench()
    S, T, dT = case["ess"][0], case["T"], case["dT"]
    bench_profit = welfare.ess_profit(S, b.p_star[0], b.g_star[0], b.p_star[0], b.g_star[0], b.s_star[0][-1],
                                      b.lam_star[0], 10.0, dT).profit
    common = dict(error=ErrorModel(), realtime=RealTime(psi=10.0), draws=1)
    star = runner.run_draw(case, b, RunSettings(bid=BidRule(family="B_star"), **common), 0)
    assert star["ess_deviation_MWh"] < 1e-6 and star["ess_imbalance_penalty"] < 1e-6
    assert abs(star["ess_profit"] - bench_profit) <= 0.01 * dT * float(np.sum(b.p_star[0] + b.g_star[0])) + 1e-6
    priv = runner.run_draw(case, b, RunSettings(bid=BidRule(family="B_priv"), **common), 0)
    assert abs(priv["ess_deviation_MWh"] - priv["undeliverable_MWh"]) < 1e-6
    assert abs(priv["ess_imbalance_penalty"] - 10.0 * priv["ess_deviation_MWh"]) < 1e-6
    total = priv["ess_energy_payments"] - priv["ess_imbalance_penalty"] - priv["ess_degradation"] + priv["ess_terminal_value"]
    assert abs(priv["ess_profit"] - total) < 1e-6
