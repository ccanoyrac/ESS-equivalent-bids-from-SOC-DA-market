"""Day-ahead clearing of the current design (CD) -- experiments.md section 4.4.

`verify_equivalence.build(case, "CD", bids)` only supports one price per ESS per side
(a single block), but the quantity rule (bids.quantity_blocks) needs two blocks per
side -- so this module writes the general multi-block clearing directly, as
experiments.md 4.4 anticipates ("reuse build() when it supports the blocks; otherwise
write the general version here").

Single-bus, hour-by-hour LP (experiments.md 4.4's "no ramps" branch): `day_case`'s
generator ramp limits are 1e6 MW/h, i.e. non-binding at this system's scale, so hour-by-
hour clearing is equivalent to one multi-hour LP and much simpler. A system with
genuinely binding ramps would need the multi-hour version instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class HourResult:
    q: np.ndarray  # (n_blocks,) MW, generator dispatch this hour
    p: list[float]  # per ESS, MW charge
    g: list[float]  # per ESS, MW discharge
    shed: float  # MW
    lam: float  # $/MWh


@dataclass
class DayAheadResult:
    q_da: np.ndarray  # (n_blocks, T) MW
    p_da: list[np.ndarray]  # per ESS, (T,) MW
    g_da: list[np.ndarray]  # per ESS, (T,) MW
    shed: np.ndarray  # (T,) MW
    lam_cd: np.ndarray  # (T,) $/MWh


def clear_hour(case: dict, t: int, ess_blocks: list[dict], voll: float) -> HourResult:
    """Clears bus 0 for hour `t` alone: generator blocks at their true marginal cost,
    each ESS's two-block charge/discharge curve from `bids.quantity_blocks`, plus a
    shedding block at `voll` (same device as `benchmark.with_shedding_generator`, built
    inline here since this is a single-hour LP, not a `verify_equivalence` case).
    """
    gens = case["gens"]
    D = float(case["D"][0, t])

    g_caps, g_costs = [], []
    for G in gens:
        for cost, qb in G["blocks"]:
            g_costs.append(cost)
            g_caps.append(qb[t])
    n_gen = len(g_costs)

    n_ess = len(ess_blocks)
    charge_specs = [eb["charge"][t] for eb in ess_blocks]  # list of [(price,qty), (price,qty)]
    discharge_specs = [eb["discharge"][t] for eb in ess_blocks]

    # variable order: gen blocks, [charge sub-blocks per ESS], [discharge sub-blocks per
    # ESS], shed. Each ESS contributes len(charge_specs[e]) + len(discharge_specs[e]) vars.
    c = list(g_costs)
    bounds = [(0, cap) for cap in g_caps]
    balance_coefs = [1.0] * n_gen  # generation: +1 in balance

    charge_slices, discharge_slices = [], []
    for e in range(n_ess):
        start = len(c)
        for price, qty in charge_specs[e]:
            c.append(-price)  # charging = demand; buying at `price` is a cost of -price per unit supplied... see sign note below
            bounds.append((0, max(qty, 0.0)))
            balance_coefs.append(-1.0)  # charging consumes from the balance (-1)
        charge_slices.append(slice(start, len(c)))
        start = len(c)
        for price, qty in discharge_specs[e]:
            c.append(price)  # discharging supplies energy, cost = price paid to the ESS
            bounds.append((0, max(qty, 0.0)))
            balance_coefs.append(1.0)
        discharge_slices.append(slice(start, len(c)))

    shed_idx = len(c)
    c.append(voll)
    bounds.append((0, D))  # shedding can never exceed demand itself
    balance_coefs.append(1.0)

    # Objective note on charging: minimising cost, a generator/offer is `+price*qty`
    # (buy less if expensive); a charge block is itself a BID (willing to pay up to
    # `price`), so including it as `-price*charge` in a cost-minimisation is exactly a
    # paired generator/load formulation -- the LP will charge up to the point where the
    # market price crosses the bid, precisely like `verify_equivalence.build`'s own CD
    # objective (`-dT*bids['u']*charge`) does (that factor of `dT` is folded in by the
    # caller scaling `delta`/prices consistently; this LP works in $/MWh * MW = $/h,
    # matching `verify_equivalence.build`'s own per-hour row scaling with `dT` applied
    # by the caller when summing results across hours, not inside this LP).
    A_eq = [balance_coefs]
    b_eq = [D]

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    assert res.status == 0, res.message

    # b_eq = +D here (unlike verify_equivalence.build's rows, whose RHS is -D), so the
    # marginal (d obj / d b_eq) is already +lambda directly -- no sign flip needed.
    lam = res.eqlin.marginals[0]
    q = np.array(res.x[:n_gen])
    p = [float(sum(res.x[charge_slices[e]])) for e in range(n_ess)]
    g = [float(sum(res.x[discharge_slices[e]])) for e in range(n_ess)]
    shed = float(res.x[shed_idx])
    return HourResult(q=q, p=p, g=g, shed=shed, lam=lam)


def clear_cd(case: dict, ess_blocks: list[dict], voll: float = 1000.0) -> DayAheadResult:
    """Clears every hour of `case` independently -- experiments.md 4.4."""
    T = case["T"]
    n_gen = sum(len(G["blocks"]) for G in case["gens"])
    n_ess = len(ess_blocks)

    q_da = np.zeros((n_gen, T))
    p_da = [np.zeros(T) for _ in range(n_ess)]
    g_da = [np.zeros(T) for _ in range(n_ess)]
    shed = np.zeros(T)
    lam_cd = np.zeros(T)

    for t in range(T):
        r = clear_hour(case, t, ess_blocks, voll)
        q_da[:, t] = r.q
        for e in range(n_ess):
            p_da[e][t] = r.p[e]
            g_da[e][t] = r.g[e]
        shed[t] = r.shed
        lam_cd[t] = r.lam

    return DayAheadResult(q_da=q_da, p_da=p_da, g_da=g_da, shed=shed, lam_cd=lam_cd)
