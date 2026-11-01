"""GUIDE.md section 8: the comparison loop. For each forecast quality `(k, bias)`, build
the DP/curve pipeline once (from the mean of that quality's scenarios) and commit it as
the conventional market's curve; then clear against every *realised* scenario and
compare against Baldick's own per-scenario benchmark.

Demand is used deterministically throughout (see `data.py` / GUIDE.md section 2's
caveat that `load_DA`/`load_RT` is a fixed scaling, not a forecast error) -- only wind
and PV carry scenario uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import baldick, clearing, data, dp, errors
from src.baldick import StorageSpec
from src.curves import build_curve


def forecast_price(
    thermal: pd.DataFrame, demand: np.ndarray, wind_mean: np.ndarray, pv_mean: np.ndarray, storage: StorageSpec
) -> np.ndarray:
    """`hat_lambda_t` (GUIDE.md sections 5 and 7.6): the price from clearing Baldick's
    own LP against the *mean* scenario (section 7.6: "the forecast used in the DP comes
    from clearing the market against the mean scenario"). Solved *with* the storage
    included, not a no-battery merit-order approximation -- at k=0 the mean scenario
    equals the single realised scenario exactly, so this must reproduce Baldick's own
    equilibrium price path bit-for-bit, which is what keeps k=0's welfare gap at zero
    (GUIDE.md section 7.5 / section 9.2's regression test).
    """
    return baldick.solve_baldick(thermal, demand, wind_mean, pv_mean, storage, dt=1.0).price


@dataclass
class ComparisonRow:
    k: float
    bias: float
    scenario: int
    sw_baldick: float
    sw_conventional: float
    gap: float  # sw_baldick - sw_conventional, GUIDE.md section 1's Delta SW


def run_comparison(
    thermal: pd.DataFrame,
    demand: np.ndarray,
    wind_da: np.ndarray,
    pv_da: np.ndarray,
    wind_ar1: errors.AR1Params,
    pv_ar1: errors.AR1Params,
    storage: StorageSpec,
    delta_grid: float,
    k_values: tuple[float, ...],
    bias: float,
    n_scenarios: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One day's worth of GUIDE.md section 8's loop, across every `k` in `k_values` at
    a fixed `bias`. Returns one row per (k, scenario) with both designs' welfare and
    the gap between them -- `k=0` should give `gap ~ 0` for every scenario (they're all
    identical to the DA forecast; see `tests/test_equivalence.py` for the real
    regression test of that claim, which this loop does not re-verify on its own).
    """
    # k=0 rows here should come out near zero, but not to the tight tolerance
    # `tests/test_equivalence.py` enforces -- that dedicated test isolates the SOC-grid
    # discretisation error alone; a few-percent residual in an ad-hoc run through this
    # loop is normal and reflects grid resolution (`delta_grid`) and/or degenerate
    # thermal-block ties (GUIDE.md section 10 -- perturb block costs by a few cents to
    # break them) rather than a design flaw. Tune `delta_grid` and re-check k=0 before
    # trusting a ΔSW computed at k>0.
    rows: list[ComparisonRow] = []

    for k in k_values:
        wind_scenarios = errors.generate_scenarios(
            wind_da, wind_ar1, k, bias, n_scenarios, data.WIND_CAPACITY_MW, rng
        )
        pv_scenarios = errors.generate_scenarios(
            pv_da, pv_ar1, k, bias, n_scenarios, data.PV_CAPACITY_MW, rng
        )

        wind_mean = wind_scenarios.mean(axis=0)
        pv_mean = pv_scenarios.mean(axis=0)
        lam_hat = forecast_price(thermal, demand, wind_mean, pv_mean, storage)

        vf = dp.solve_value_function(storage, lam_hat, delta_grid)
        traj = dp.simulate_policy(vf, storage)
        T = len(demand)
        disc_curve = [build_curve(vf, t, storage, traj[t])[0] for t in range(T)]
        chg_curve = [build_curve(vf, t, storage, traj[t])[1] for t in range(T)]

        for i in range(n_scenarios):
            wind_realised = wind_scenarios[i]
            pv_realised = pv_scenarios[i]

            bald = baldick.solve_baldick(thermal, demand, wind_realised, pv_realised, storage, dt=1.0)
            conv = clearing.clear_conventional(
                thermal, demand, wind_realised, pv_realised, storage, disc_curve, chg_curve, dt=1.0
            )
            sw_bald = bald.social_welfare
            sw_conv = storage.b_slope * conv.soc[-1] - conv.true_cost(thermal, storage)

            rows.append(
                ComparisonRow(k=k, bias=bias, scenario=i, sw_baldick=sw_bald, sw_conventional=sw_conv, gap=sw_bald - sw_conv)
            )

    return pd.DataFrame(r.__dict__ for r in rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean gap and its spread per forecast quality `k` -- GUIDE.md section 8's
    "report mean gap, distribution of gap"."""
    return results.groupby(["k", "bias"])["gap"].agg(["mean", "std", "min", "max", "count"]).reset_index()
