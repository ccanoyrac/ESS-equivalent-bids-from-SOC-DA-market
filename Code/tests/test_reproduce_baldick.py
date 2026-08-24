"""Regression tests locking the solver to Baldick (2026) Tables 2-4. These must keep
passing after any refactor -- see ROADMAP.md, "Exact reproduction" (stage 1).

Not every quantity in the paper's tables is actually pinned down by the LP. Intervals 3-4
share identical demand (300 MW) and RAC (500 MW), and in the base case thermal and storage
discharge tie exactly in marginal cost across intervals 5-6 (thermal is interior, not at a
bound) -- both are genuine continuous degeneracies of the optimum, confirmed algebraically
in ROADMAP.md, "Price formation & dispatch degeneracy", not solver artifacts. Tests below
assert exact values only where the LP structure forces them, and pair sums / ranges where
it does not; the objective is always exact since alternate optima share the same value.
"""

from __future__ import annotations

import pytest

from experiments.reproduce_baldick import run_case

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
        "prices": {1: 50, 2: 0, 3: 0},
        "price_ranges": {0: (0, 43.2)},
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
        "prices": {1: 50, 2: 0, 3: 0, 4: 56, 5: 56},
        "price_ranges": {0: (0, 43.2)},
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
        "prices": {1: 50, 2: 0, 3: 0, 4: 41, 5: 41},
        "price_ranges": {0: (0, 43.2)},
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
    for i, (lo, hi) in CASES[name]["price_ranges"].items():
        assert lo - 1e-6 <= result.price[i] <= hi + 1e-6, f"{name}: price[{i}] out of range"


@pytest.mark.parametrize("name", CASES.keys())
def test_ess_surplus_is_non_negative(name: str) -> None:
    # Paper Section 4.5: with LMP settlement, ESS surplus is always non-negative at
    # the optimum. The exact value depends on which (equally valid) degenerate price
    # is chosen for intervals 1 and, in the base case, 6 -- see footnote 17.
    result = run_case(name)
    assert result.ess_surplus[0] >= -1.0
