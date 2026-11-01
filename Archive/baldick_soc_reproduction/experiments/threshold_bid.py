"""Threshold bids and the concavity requirement -- built directly on Baldick (2026)'s own
worked example, not a separate stylized system.

The SOC-based design (paper Section 4.3) requires the declared willingness-to-pay `w` to
be non-increasing, i.e. the terminal bid `b` to be concave -- that is what makes the
storage owner's own problem convex, and so what makes its KKT conditions *sufficient*
for optimality rather than merely necessary. This experiment asks whether an economically
ordinary valuation can fail that requirement even when the owner is entirely sincere: a
threshold benefit (a capacity obligation, a warranty, a qualifying reserve level) that
pays a lump sum K on crossing a stock level Delta_0 puts a spike in the owner's true
marginal value at Delta_0. A spike is an increase, and the bid format forbids increases.
This is the same non-convexity that forces unit commitment to use binaries and uplift; the
difference is that the SOC bid format simply has no way to express it at all.

The system is exactly `experiments.reproduce_baldick`'s "increased_max_soc" case
(Table 4 of the paper, same demand/RAC/thermal/storage, s_max=1200 MWh, w=40 $/MWh) --
reused via its own public constants, not rebuilt. That case already has terminal_soc=100
MWh with no threshold at all: adding a threshold at Delta_0=200 (one 100 MWh block
further) sits exactly where the calibration in Sec. 7 of the spec asked for -- a genuine
near miss, in a scenario already reproduced, verified, and tabulated in the rest of this
codebase, not a new stylized one. (The base case's own s_max=800 doesn't leave physical
room for this: its ceiling on reachable terminal SOC coincides almost exactly with where
its economic optimum already sits -- see the notebook for why 1200 was needed instead.)

Reuses `core`/`terminal_value`/`markets.day_ahead` as-is for every run whose declared bid
is concave (runs B, C, D). Only Run A -- which needs the true *non*-concave valuation and
a hard floor on terminal SOC that `solve_day_ahead`'s API has no way to express -- gets a
one-off, hardcoded dispatch model, kept entirely in this file so nothing shared changes.

Run as `python -m experiments.threshold_bid`.
"""

from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from core.generator import RenewableGenerator, ThermalGenerator
from core.storage import StorageParams
from core.system import System
from experiments.reproduce_baldick import ALPHA, BETA, DEMAND, DT, EPSILON, GAMMA, RAC, THERMAL_COST, THERMAL_P_MAX, ZETA
from markets.day_ahead import DispatchResult, solve_day_ahead
from terminal_value.base import TerminalValueFunction

T = len(DEMAND)  # 6, Baldick's own base-case horizon

# Exactly reproduce_baldick.py's own "increased_max_soc" system (Table 4): same demand,
# RAC, thermal unit, and storage technical characteristics, s_max=1200 MWh, w=40 $/MWh,
# s0=0. Rebuilt from that module's own public constants rather than importing its
# private `_system` helper.
#
# s_max=1200 (rather than the base case's 800) is needed for headroom, not for its own
# sake: this system's own physical ceiling on terminal SOC -- how much a forced
# generation shortfall in intervals 5-6 (demand exceeds thermal+renewable there) can
# drain the tank by -- turns out to be exactly `s_max - 300` regardless of the bid. At
# s_max=800 that ceiling is 500, which is *already* where w=55's own economic optimum
# sits (Table 3) -- leaving no physical room for a threshold to push further, at any K.
# At s_max=1200, w=40's own optimum (Table 4) sits at Delta*=100, comfortably below its
# 900 MWh ceiling, which is exactly the room this experiment needs.
S_MAX = 1200.0
S0 = 0.0
W = 40.0  # Baldick's own base-case value -- the flat arbitrage part of w_true


def build_system() -> System:
    return System(
        demand=DEMAND,
        thermal=(ThermalGenerator("thermal", cost=THERMAL_COST, p_max=THERMAL_P_MAX),),
        renewables=(RenewableGenerator("renewable", capacity=RAC),),
        storages=(StorageParams(alpha=ALPHA, beta=BETA, gamma=GAMMA, s_min=0.0, s_max=S_MAX, s0=S0, epsilon=EPSILON, zeta=ZETA),),
        dt=DT,
    )


# --------------------------------------------------------------------------------------
# The true valuation: Baldick's own flat w=40 arbitrage value, plus a threshold benefit K
# on crossing Delta_0 -- calibrated below (see the notebook for the search). Breakpoints
# are coarse and round, matching the paper's own numbers: with no threshold at all, this
# system's own optimum already lands at Delta*=100 (Table 4); Delta_0=200 is exactly one
# 100 MWh block further, so K=0 gives a genuine one-block near miss "for free", without
# needing a separately-tuned arbitrage stack. sbar_T = gamma**T * s0 = 0 here (s0=0), so
# Delta = s_T directly -- no sbar_T/s0 decoupling to worry about, unlike a scenario that
# starts with nonzero initial SOC.
# --------------------------------------------------------------------------------------

BREAKPOINTS = (0.0, 100.0, 200.0, 1200.0)
THRESHOLD_BLOCK = 1  # the [100, 200] block, i.e. the one ending at Delta_0
THRESHOLD_LEVEL = BREAKPOINTS[THRESHOLD_BLOCK + 1]  # Delta_0 = 200
THRESHOLD_BENEFIT = 1200.0  # K, $ -- see the notebook for the search: z0/z1 tie at K=900


def _block_widths(breakpoints: tuple[float, ...] = BREAKPOINTS) -> tuple[float, ...]:
    return tuple(breakpoints[k + 1] - breakpoints[k] for k in range(len(breakpoints) - 1))


def arbitrage_marginal_values() -> tuple[float, ...]:
    """w's flat arbitrage part alone (K=0): W on every block -- reproduces
    reproduce_baldick.py's own "increased_max_soc" bid exactly, just expressed on finer
    breakpoints so the threshold block can be isolated from its neighbours.
    """
    return (W,) * len(_block_widths())


def true_marginal_values() -> tuple[float, ...]:
    """w_true: the flat W arbitrage value with K/width added to the block ending at
    Delta_0. Deliberately NOT wrapped in a TerminalValueFunction -- its whole point is
    that it fails that class's own non-increasing validation.
    """
    widths = _block_widths()
    bumped = list(arbitrage_marginal_values())
    bumped[THRESHOLD_BLOCK] += THRESHOLD_BENEFIT / widths[THRESHOLD_BLOCK]
    return tuple(bumped)


def piecewise_bid_value(breakpoints: tuple[float, ...], marginal_values: tuple[float, ...], deviation: float) -> float:
    """b(deviation) for an arbitrary (not necessarily monotone) piecewise-linear marginal
    value schedule -- the same integration TerminalValueFunction.value() does, duplicated
    here only because the true bid can't be constructed as a TerminalValueFunction at all
    (its own validation rejects the threshold spike by design).
    """
    idx0 = breakpoints.index(0.0)
    total = 0.0
    if deviation >= 0:
        remaining = deviation
        for k in range(idx0, len(marginal_values)):
            width = breakpoints[k + 1] - breakpoints[k]
            take = min(width, remaining)
            total += take * marginal_values[k]
            remaining -= take
            if remaining <= 0:
                break
    else:
        remaining = -deviation
        for k in range(idx0 - 1, -1, -1):
            width = breakpoints[k + 1] - breakpoints[k]
            take = min(width, remaining)
            total -= take * marginal_values[k]
            remaining -= take
            if remaining <= 0:
                break
    return total


def concave_envelope(breakpoints: tuple[float, ...], marginal_values: tuple[float, ...]) -> tuple[float, ...]:
    """The upper concave envelope of the points (breakpoints[j], b(breakpoints[j])): the
    smallest concave function everywhere >= b_true, touching it at a subset of the
    breakpoints. Computed as the upper convex hull of those points; its successive slopes
    are the returned, automatically non-increasing, marginal values.
    """
    xs = breakpoints
    ys = [0.0]
    for k, w in enumerate(marginal_values):
        ys.append(ys[-1] + w * (xs[k + 1] - xs[k]))

    hull: list[int] = []
    for i in range(len(xs)):
        while len(hull) >= 2:
            j, k = hull[-2], hull[-1]
            cross = (xs[k] - xs[j]) * (ys[i] - ys[j]) - (ys[k] - ys[j]) * (xs[i] - xs[j])
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(i)

    hull_x = [xs[i] for i in hull]
    hull_y = [ys[i] for i in hull]
    env_slopes = [(hull_y[i + 1] - hull_y[i]) / (hull_x[i + 1] - hull_x[i]) for i in range(len(hull) - 1)]

    out = []
    seg = 0
    for k in range(len(marginal_values)):
        while xs[k + 1] > hull_x[seg + 1] + 1e-9:
            seg += 1
        out.append(env_slopes[seg])
    return tuple(out)


def pava_nonincreasing(values: tuple[float, ...], weights: tuple[float, ...] | None = None) -> tuple[float, ...]:
    """Pool-Adjacent-Violators: the L2 projection of `values` onto the cone of
    non-increasing sequences, weighted by `weights` (block widths; equal-weight if None).
    Preserves sum(values[j] * weights[j]) exactly -- it only redistributes, never inflates
    or deflates -- which is what makes this the control against the envelope in Sec. 5.1.
    """
    n = len(values)
    if weights is None:
        weights = (1.0,) * n

    blocks: list[list[float]] = []
    for v, w in zip(values, weights):
        blocks.append([v, w, 1])
        while len(blocks) >= 2 and blocks[-2][0] < blocks[-1][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            merged_level = (v1 * w1 + v2 * w2) / (w1 + w2)
            blocks.append([merged_level, w1 + w2, n1 + n2])

    out: list[float] = []
    for level, _, count in blocks:
        out.extend([level] * count)
    return tuple(out)


def arbitrage_bid() -> TerminalValueFunction:
    """The flat W arbitrage value alone (K=0) -- a normal, admissible bid, reusable
    directly with solve_day_ahead."""
    return TerminalValueFunction(breakpoints=BREAKPOINTS, marginal_values=arbitrage_marginal_values())


def envelope_bid() -> TerminalValueFunction:
    return TerminalValueFunction(breakpoints=BREAKPOINTS, marginal_values=concave_envelope(BREAKPOINTS, true_marginal_values()))


def projected_bid() -> TerminalValueFunction:
    return TerminalValueFunction(
        breakpoints=BREAKPOINTS, marginal_values=pava_nonincreasing(true_marginal_values(), _block_widths())
    )


# --------------------------------------------------------------------------------------
# Run A's hardcoded dispatch model. This is the one genuine discrepancy from
# markets.day_ahead.solve_day_ahead: it needs (a) the true, non-concave valuation, which
# TerminalValueFunction's own validation refuses to hold, and (b) an optional hard floor
# on terminal SOC (the z=1 sub-case of Sec. 6.2), which solve_day_ahead's API has no way
# to express. Structurally this is otherwise the exact same LP as solve_day_ahead,
# specialised to this experiment's single thermal/renewable/storage system.
# --------------------------------------------------------------------------------------


@dataclass
class ThresholdDispatch:
    thermal: np.ndarray  # (T,)
    renewable: np.ndarray  # (T,)
    charge: np.ndarray  # (T,)
    discharge: np.ndarray  # (T,)
    soc: np.ndarray  # (T,)
    objective: float

    @property
    def terminal_deviation(self) -> float:
        return float(self.soc[-1])  # sbar_T = 0 here (s0=0), so Delta = s_T directly


def _solve_with_terminal_floor(system: System, bid: TerminalValueFunction, min_terminal_soc: float | None) -> ThresholdDispatch:
    """A one-off variant of solve_day_ahead for this experiment only: identical LP,
    specialised to a single thermal unit, a single renewable, and a single storage, with
    one addition -- `min_terminal_soc` forces storage[0]'s terminal SOC (not any other
    interval's) to sit at or above a floor. Used for both sub-cases of Run A
    (floor=None is exactly z=0; a floor is exactly z=1); kept separate from the shared
    solve_day_ahead so that function's API and every other experiment stay untouched.
    """
    thermal_gen = system.thermal[0]
    renewable_gen = system.renewables[0]
    storage = system.storages[0]
    up, down = bid.lp_segments()

    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model("threshold_run_a", env=env) as model:
        thermal = {t: model.addVar(lb=0.0, ub=thermal_gen.p_max, name=f"thermal[{t}]") for t in range(T)}
        renewable = {t: model.addVar(lb=0.0, ub=renewable_gen.capacity[t], name=f"renewable[{t}]") for t in range(T)}
        charge = {t: model.addVar(lb=0.0, ub=storage.p_max, name=f"charge[{t}]") for t in range(T)}
        discharge = {t: model.addVar(lb=0.0, ub=storage.g_max, name=f"discharge[{t}]") for t in range(T)}
        soc = {t: model.addVar(lb=storage.s_min, ub=storage.s_max, name=f"soc[{t}]") for t in range(T)}
        up_vars = [model.addVar(lb=0.0, ub=width, name=f"up[{k}]") for k, (width, _) in enumerate(up)]
        down_vars = [model.addVar(lb=0.0, ub=width, name=f"down[{k}]") for k, (width, _) in enumerate(down)]

        def soc_before(t: int):
            return soc[t - 1] if t > 0 else storage.s0

        for t in range(T):
            model.addConstr(
                soc[t] == DT * storage.alpha * charge[t] - DT * storage.beta * discharge[t] + storage.gamma * soc_before(t)
            )
            model.addConstr(storage.gamma * soc_before(t) + DT * storage.alpha * charge[t] <= storage.s_max)
            model.addConstr(storage.gamma * soc_before(t) - DT * storage.beta * discharge[t] >= storage.s_min)

        sbar_T = storage.sbar_T(T)
        model.addConstr(soc[T - 1] - gp.quicksum(up_vars) + gp.quicksum(down_vars) == sbar_T)
        if min_terminal_soc is not None:
            model.addConstr(soc[T - 1] >= min_terminal_soc)

        model.addConstrs(thermal[t] + renewable[t] + discharge[t] - charge[t] == system.demand[t] for t in range(T))

        model.setObjective(
            DT * gp.quicksum(thermal_gen.cost * thermal[t] for t in range(T))
            + DT * gp.quicksum(storage.epsilon * charge[t] + storage.zeta * discharge[t] for t in range(T))
            - gp.quicksum(mv * up_vars[k] for k, (_, mv) in enumerate(up))
            + gp.quicksum(mv * down_vars[k] for k, (_, mv) in enumerate(down)),
            GRB.MINIMIZE,
        )
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Run A sub-case (floor={min_terminal_soc}) failed: Gurobi status {model.Status}")

        return ThresholdDispatch(
            thermal=np.array([thermal[t].X for t in range(T)]),
            renewable=np.array([renewable[t].X for t in range(T)]),
            charge=np.array([charge[t].X for t in range(T)]),
            discharge=np.array([discharge[t].X for t in range(T)]),
            soc=np.array([soc[t].X for t in range(T)]),
            objective=model.ObjVal,
        )


@dataclass
class RunAResult:
    winning_z: int
    dispatch: ThresholdDispatch
    objective_true: float  # the z=0/z=1 objective, corrected for K -- comparable to B/C/D


def run_a(system: System, threshold_level: float = THRESHOLD_LEVEL, threshold_benefit: float = THRESHOLD_BENEFIT) -> RunAResult:
    """Solve the true, non-concave valuation without a MILP solver: exactly two LPs
    (Sec. 6.2), one per value of the (never-declared) binary z, taking the better
    objective. z=0 needs no floor; z=1 forces Delta >= Delta_0 and then earns -K, which is
    a constant and so is applied after solving rather than inside the LP.

    `threshold_level`/`threshold_benefit` default to the module's calibrated values but
    are exposed so the K=0 control (check 1) can call this with threshold_benefit=0 and
    confirm z=0 wins trivially (forcing a floor can never help when there's no reward for
    it), reproducing the same dispatch as runs B/C/D's own K=0 versions.
    """
    bid = arbitrage_bid()
    sbar_T = system.storages[0].sbar_T(T)

    z0 = _solve_with_terminal_floor(system, bid, min_terminal_soc=None)
    z1 = _solve_with_terminal_floor(system, bid, min_terminal_soc=sbar_T + threshold_level)
    obj_z1_true = z1.objective - threshold_benefit

    if z0.objective <= obj_z1_true:
        return RunAResult(winning_z=0, dispatch=z0, objective_true=z0.objective)
    return RunAResult(winning_z=1, dispatch=z1, objective_true=obj_z1_true)


# --------------------------------------------------------------------------------------
# Welfare/surplus, evaluated with the TRUE valuation regardless of which bid was
# submitted to reach that dispatch -- the crux of the experiment (Sec. 8).
# --------------------------------------------------------------------------------------


def true_welfare_and_surplus(
    system: System, thermal: np.ndarray, charge: np.ndarray, discharge: np.ndarray, price: np.ndarray | None, delta: float
) -> tuple[float, float]:
    storage = system.storages[0]
    operating_cost = DT * (
        system.thermal[0].cost * thermal.sum() + storage.epsilon * charge.sum() + storage.zeta * discharge.sum()
    )
    b_true = piecewise_bid_value(BREAKPOINTS, true_marginal_values(), delta)
    system_welfare = -operating_cost + b_true

    if price is None:
        owner_surplus = float("nan")
    else:
        owner_surplus = DT * float(np.dot(price, discharge - charge) - storage.epsilon * charge.sum() - storage.zeta * discharge.sum())
        owner_surplus += b_true
    return system_welfare, owner_surplus


def headline_row(label: str, bid_used: str, system: System, delta: float, thermal, charge, discharge, price=None) -> dict:
    """One row of the headline table (Sec. 8): welfare and surplus are *always* evaluated
    against the TRUE valuation, regardless of which bid actually produced this dispatch."""
    welfare, surplus = true_welfare_and_surplus(system, thermal, charge, discharge, price, delta)
    threshold_met = delta >= THRESHOLD_LEVEL - 1e-6
    return {
        "run": label,
        "bid used": bid_used,
        "Delta*": delta,
        "threshold met": threshold_met,
        "system welfare": welfare,
        "owner surplus": surplus,
        "penalty incurred": 0.0 if threshold_met else THRESHOLD_BENEFIT,
    }


def run_all() -> tuple[System, RunAResult, DispatchResult, DispatchResult, dict[str, DispatchResult], list[dict]]:
    """Runs A-D (Sec. 6). Returns (system, run_a_result, run_b_result, run_c_result,
    {bid_name: run_d_result}, headline_rows) -- everything the notebook displays."""
    system = build_system()

    a = run_a(system)
    b = solve_day_ahead(system, [envelope_bid()])
    c = solve_day_ahead(system, [projected_bid()])
    widths = _block_widths()
    d = {
        "true (K=0)": solve_day_ahead(system, [arbitrage_bid()]),
        "envelope (K=0)": solve_day_ahead(
            system, [TerminalValueFunction(BREAKPOINTS, concave_envelope(BREAKPOINTS, arbitrage_marginal_values()))]
        ),
        "projected (K=0)": solve_day_ahead(
            system, [TerminalValueFunction(BREAKPOINTS, pava_nonincreasing(arbitrage_marginal_values(), widths))]
        ),
    }

    rows = [
        headline_row("A", "true (binary)", system, a.dispatch.terminal_deviation, a.dispatch.thermal, a.dispatch.charge, a.dispatch.discharge, None),
        headline_row("B", "concave envelope", system, b.terminal_soc[0], b.thermal[0], b.charge[0], b.discharge[0], b.price),
        headline_row("C", "isotonic projection", system, c.terminal_soc[0], c.thermal[0], c.charge[0], c.discharge[0], c.price),
        headline_row("D", "control, K=0", system, d["true (K=0)"].terminal_soc[0], d["true (K=0)"].thermal[0], d["true (K=0)"].charge[0], d["true (K=0)"].discharge[0], d["true (K=0)"].price),
    ]
    return system, a, b, c, d, rows


def duality_gap(delta: float) -> float:
    """b_env(Delta*) - b_true(Delta*) at run B's own resulting dispatch (Sec. 5.1)."""
    b_env = piecewise_bid_value(BREAKPOINTS, concave_envelope(BREAKPOINTS, true_marginal_values()), delta)
    b_true = piecewise_bid_value(BREAKPOINTS, true_marginal_values(), delta)
    return b_env - b_true


def main() -> None:
    system, a, b, c, d, rows = run_all()
    for row in rows:
        surplus = "n/a" if np.isnan(row["owner surplus"]) else f"${row['owner surplus']:,.2f}"
        print(
            f"Run {row['run']:1s} ({row['bid used']:20s}): Delta*={row['Delta*']:6.2f}  "
            f"met={row['threshold met']!s:5s}  welfare(true)=${row['system welfare']:>10,.2f}  "
            f"surplus(true)={surplus}"
        )
    print(f"\nDuality gap at B's dispatch: ${duality_gap(b.terminal_soc[0]):,.2f}")


if __name__ == "__main__":
    main()
