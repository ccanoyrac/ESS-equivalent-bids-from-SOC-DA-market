"""Regression tests locking the solver to Baldick (2026) Tables 2-4. These must keep
passing after any refactor -- see ROADMAP.md, "Exact reproduction" (stage 1).

Not every quantity in the paper's tables is actually pinned down by the LP. Intervals 3-4
share identical demand (300 MW) and RAC (500 MW), and in the base case thermal and storage
discharge tie exactly in marginal cost across intervals 5-6 (thermal is interior, not at a
bound) -- both are genuine continuous degeneracies of the optimum, confirmed algebraically
in ROADMAP.md, "Price formation & dispatch degeneracy", not solver artifacts. Tests below
assert exact values only where the LP structure forces them, and pair sums / ranges where
it does not; the objective is always exact since alternate optima share the same value.

Prices are asserted against the EXACT dual-feasible range computed by
`markets.day_ahead.price_ranges` (see `test_price_ranges_are_dual_verified` and
ROADMAP.md), not against the paper's stated ranges -- those turned out not to all hold
in this exact reproduction (in particular, the paper's claimed $0-$43.1 range for
interval 1, and $41-$50 range for the base case's interval 6, both come out unique here).
"""

from __future__ import annotations

import pytest

from experiments.reproduce_baldick import CASES as SYSTEM_CASES
from experiments.reproduce_baldick import run_case
from markets.day_ahead import price_ranges, solve_day_ahead

# Common to all three cases: intervals 1-2 are never part of a tie (no other interval
# shares their demand/RAC/marginal-cost structure), so their dispatch is forced.
COMMON_FORCED = {
    "renewable": {0: 200, 1: 100},
    "thermal": {0: 0, 1: 10, 2: 0, 3: 0},
    "discharge": {0: 0, 1: 90, 2: 0, 3: 0},
    "charge": {0: 100, 1: 0},
}

CASES = {
    "base_case": {
        "forced": {
            "renewable": {4: 25, 5: 0},
            "charge": {4: 0, 5: 0},
        },
        "soc": {1: 0, 3: 800, 5: 0},
        "pair_sums": {
            (2, 3): {"renewable": 822.222, "charge": 222.222},
            (4, 5): {"thermal": 75, "discharge": 200},
        },
        "objective": 19_448.89,
        # Every interval is dual-unique in this case -- see test_price_ranges_are_dual_verified.
        "prices": {0: 43.1, 1: 50, 2: 0, 3: 0, 4: 50, 5: 50},
    },
    "high_willingness_to_pay": {
        "forced": {
            "renewable": {4: 25, 5: 0},
            "thermal": {4: 100, 5: 100},
            "discharge": {4: 75, 5: 0},
            "charge": {4: 0, 5: 0},
        },
        "soc": {1: 0, 3: 800, 5: 500},
        "pair_sums": {(2, 3): {"renewable": 822.222, "charge": 222.222}},
        "objective": 16_448.89,
        "prices": {0: 43.1, 1: 50, 2: 0, 3: 0, 4: 56},
        # The one genuine range found anywhere in these three cases (width ~6).
        "price_ranges": {5: (50, 56)},
    },
    "increased_max_soc": {
        "forced": {
            "renewable": {4: 25, 5: 0},
            "thermal": {4: 0, 5: 0},
            "discharge": {4: 175, 5: 100},
            "charge": {4: 0, 5: 0},
        },
        "soc": {1: 0, 3: 1200, 5: 100},
        "pair_sums": {(2, 3): {"renewable": 933.333, "charge": 333.333}},
        "objective": 1_193.33,
        "prices": {0: 43.1, 1: 50, 2: 0, 3: 0, 4: 41, 5: 41},
    },
}


@pytest.mark.parametrize("name", CASES.keys())
def test_forced_dispatch_matches_paper(name: str) -> None:
    result = run_case(name)
    arrays = {
        "renewable": result.renewable[0],
        "thermal": result.thermal[0],
        "discharge": result.discharge[0],
        "charge": result.charge[0],
    }
    for series, expected_by_index in COMMON_FORCED.items():
        for i, value in expected_by_index.items():
            assert arrays[series][i] == pytest.approx(value, abs=1e-2), f"{name}: {series}[{i}]"
    for series, expected_by_index in CASES[name]["forced"].items():
        for i, value in expected_by_index.items():
            assert arrays[series][i] == pytest.approx(value, abs=1e-2), f"{name}: {series}[{i}]"


@pytest.mark.parametrize("name", CASES.keys())
def test_soc_checkpoints_match_paper(name: str) -> None:
    result = run_case(name)
    for i, value in CASES[name]["soc"].items():
        assert result.soc[0, i] == pytest.approx(value, abs=1e-2), f"{name}: soc[{i}]"


@pytest.mark.parametrize("name", CASES.keys())
def test_degenerate_pairs_sum_to_paper_totals(name: str) -> None:
    """Individual splits within a tied pair of intervals are not unique, but their sum
    always is -- it is pinned down by the forced SOC checkpoints bracketing the pair."""
    result = run_case(name)
    arrays = {
        "renewable": result.renewable[0],
        "thermal": result.thermal[0],
        "discharge": result.discharge[0],
        "charge": result.charge[0],
    }
    for (i, j), totals in CASES[name]["pair_sums"].items():
        for series, expected_total in totals.items():
            actual = arrays[series][i] + arrays[series][j]
            assert actual == pytest.approx(expected_total, abs=1e-2), f"{name}: {series}[{i}]+{series}[{j}]"


@pytest.mark.parametrize("name", CASES.keys())
def test_objective_matches_paper(name: str) -> None:
    # The optimal objective value is unique even when the optimal dispatch/prices are not.
    result = run_case(name)
    assert result.objective == pytest.approx(CASES[name]["objective"], abs=1.0)


@pytest.mark.parametrize("name", CASES.keys())
def test_prices_match_paper_where_forced(name: str) -> None:
    result = run_case(name)
    for i, price in CASES[name]["prices"].items():
        assert result.price[i] == pytest.approx(price, abs=1e-1), f"{name}: price[{i}]"
    for i, (lo, hi) in CASES[name].get("price_ranges", {}).items():
        assert lo - 1e-6 <= result.price[i] <= hi + 1e-6, f"{name}: price[{i}] out of range"


@pytest.mark.parametrize("name", CASES.keys())
def test_price_ranges_are_dual_verified(name: str) -> None:
    """Directly check every interval's true dual-feasible price range (not just the
    handful of points test_prices_match_paper_where_forced happens to assert), using the
    explicit-dual tool in markets/pricing.py. This is what CASES[name]["prices"] and
    ["price_ranges"] above are actually derived from."""
    system, terminal_values = SYSTEM_CASES[name]
    ranges = price_ranges(system, terminal_values)
    expected_ranges = CASES[name].get("price_ranges", {})
    for i, (lo, hi) in enumerate(ranges):
        if i in expected_ranges:
            exp_lo, exp_hi = expected_ranges[i]
            assert lo == pytest.approx(exp_lo, abs=1e-1), f"{name}: price_ranges[{i}] lower bound"
            assert hi == pytest.approx(exp_hi, abs=1e-1), f"{name}: price_ranges[{i}] upper bound"
        else:
            assert hi - lo < 1e-3, f"{name}: price[{i}] expected unique, found range [{lo}, {hi}]"
            assert lo == pytest.approx(CASES[name]["prices"][i], abs=1e-1), f"{name}: price_ranges[{i}]"


@pytest.mark.parametrize("name", CASES.keys())
def test_price_rule_min_max_bracket_default(name: str) -> None:
    """price_rule='min'/'max' must always bracket the default (solver-arbitrary) price,
    and must coincide with it wherever price_ranges shows the interval is unique."""
    system, terminal_values = SYSTEM_CASES[name]
    default = solve_day_ahead(system, terminal_values)
    lo = solve_day_ahead(system, terminal_values, price_rule="min")
    hi = solve_day_ahead(system, terminal_values, price_rule="max")
    for i in range(len(default.price)):
        assert lo.price[i] <= default.price[i] + 1e-6, f"{name}: min > default at price[{i}]"
        assert default.price[i] <= hi.price[i] + 1e-6, f"{name}: default > max at price[{i}]"
        if i not in CASES[name].get("price_ranges", {}):
            assert lo.price[i] == pytest.approx(hi.price[i], abs=1e-1), f"{name}: expected unique at price[{i}]"
    # Price selection never changes dispatch or system cost.
    assert lo.objective == pytest.approx(default.objective, abs=1.0)
    assert hi.objective == pytest.approx(default.objective, abs=1.0)


@pytest.mark.parametrize("name", CASES.keys())
def test_ess_surplus_is_non_negative(name: str) -> None:
    # Paper Section 4.5: with LMP settlement, ESS surplus is always non-negative at
    # the optimum. The exact value depends on which (equally valid) degenerate price
    # is chosen where price_ranges shows a genuine range -- see footnote 17.
    result = run_case(name)
    assert result.ess_surplus[0] >= -1.0
