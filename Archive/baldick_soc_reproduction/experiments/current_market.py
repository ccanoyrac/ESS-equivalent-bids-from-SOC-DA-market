"""The "current" market model: storage participates as a conventional paired
generator/load submitting a price for each interval, instead of having the ISO
optimize its SOC trajectory directly. This is the design Baldick (2026) explicitly
critiques in Section 3 -- "the ESS owner either specifies the intervals for functioning
as generator and functioning as load, or allows that decision to be made pursuant to an
amalgamated bid and offer" -- and it is ROADMAP.md stage 3's "conventional
interval-bidding benchmark," not yet built until now.

The storage owner needs a reference price to bid against every interval. Here that
reference is as favorable as it could possibly be: `pi_t`, the exact SOC costate that
`markets.day_ahead.solve_day_ahead` itself computed for the *same* system when it ran
Baldick's own SOC-based clearing (see `experiments/reproduce_baldick.py`). A real
self-scheduling owner would only ever have a forecast of this, never the true value --
so every gap this model shows relative to Baldick's own dispatch is a *lower bound* on
what forecast error would add in practice.

Storage bids like a normal paired generator/load: an offer to discharge at price=pi_t
(up to g_max) and a bid to charge at price=pi_t (up to p_max). In a standard
cost-minimizing economic dispatch a resource offered at cost c is dispatched exactly
when the clearing price >= c, and a load bid at price c consumes exactly when the
price <= c -- so "discharge when price > pi_t, charge when price < pi_t" falls directly
out of the LP; it is never hand-coded as an if/else.

Each interval is cleared *independently* (`solve_interval`), exactly as conventional
markets do -- one auction per hour, not one joint optimization over the horizon. That is
the whole point of the comparison: nothing here enforces energy conservation across
intervals, so `solve_sequential` also tracks what the resulting SOC trajectory *would
have been*, purely as bookkeeping after the fact, and flags every interval where it
falls outside the storage's own physical limits -- a failure mode that cannot occur in
Baldick's own jointly-optimized design by construction.

Run as `python -m experiments.current_market`.
"""

from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from core.storage import StorageParams
from core.system import System


@dataclass
class IntervalResult:
    thermal: np.ndarray  # (n_thermal,)
    renewable: np.ndarray  # (n_renewable,)
    charge: float
    discharge: float
    price: float


def solve_interval(system: System, t: int, pi_t: float, storage: StorageParams, power_limit: float) -> IntervalResult:
    """Clears interval `t` alone: thermal and renewable exactly as in `system`; storage
    bids as a paired generator (discharge, offered at `pi_t + zeta`) and load (charge,
    bid at `pi_t - epsilon`), each capped at `power_limit` MW. No SOC, no other interval
    -- that absence is the model, not an omission.

    Degradation (`epsilon`/`zeta`) is included in the objective for the same reason
    Baldick's own model always includes it: without it, whenever the clearing price
    lands exactly on `pi_t` (common here, since storage's own bid/offer frequently sets
    that price), charging and discharging at once nets to the same cost as doing
    neither, and the LP is free to pick an arbitrarily large, physically senseless
    simultaneous charge-and-discharge. With a strictly positive cost per MWh moved,
    the smallest-magnitude combination (never both at once, unless price sits exactly on
    the offer/bid) is the unique optimum, exactly as Sec. 5.3's arbitrage argument (K2)-(K3)
    already establishes for Baldick's own jointly-optimized model.
    """
    dt = system.dt

    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model("interval", env=env) as model:
        thermal = {
            g: model.addVar(lb=gen.p_min, ub=gen.p_max, name=f"thermal[{gen.name}]")
            for g, gen in enumerate(system.thermal)
        }
        renewable = {
            r: model.addVar(lb=0.0, ub=gen.capacity[t], name=f"renewable[{gen.name}]")
            for r, gen in enumerate(system.renewables)
        }
        discharge = model.addVar(lb=0.0, ub=power_limit, name="discharge")
        charge = model.addVar(lb=0.0, ub=power_limit, name="charge")

        balance = model.addConstr(
            gp.quicksum(thermal.values()) + gp.quicksum(renewable.values()) + discharge - charge == system.demand[t],
            name="balance",
        )

        # Discharge is a normal generator offer (cost=pi_t+zeta, dispatched iff
        # price>=pi_t+zeta); charge is a normal elastic-demand bid (price=pi_t-epsilon,
        # consumes iff price<=pi_t-epsilon) -- same "-bid*consumption" convention the
        # paper itself uses for D8's demand bids.
        model.setObjective(
            dt * gp.quicksum(gen.cost * thermal[g] for g, gen in enumerate(system.thermal))
            + dt * gp.quicksum(gen.cost * renewable[r] for r, gen in enumerate(system.renewables))
            + dt * (pi_t + storage.zeta) * discharge
            - dt * (pi_t - storage.epsilon) * charge,
            GRB.MINIMIZE,
        )
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"interval {t} failed to solve: Gurobi status {model.Status}")

        return IntervalResult(
            thermal=np.array([thermal[g].X for g in range(len(system.thermal))]),
            renewable=np.array([renewable[r].X for r in range(len(system.renewables))]),
            charge=charge.X,
            discharge=discharge.X,
            price=balance.Pi / dt,
        )


@dataclass
class SequentialDispatchResult:
    """One storage unit's worth of sequentially-cleared dispatch. `soc` and
    `soc_within_bounds` are derived by simulating the SOC dynamics *after* the fact --
    nothing in `solve_interval`'s own clearing enforces them."""

    thermal: np.ndarray  # (n_thermal, T)
    renewable: np.ndarray  # (n_renewable, T)
    charge: np.ndarray  # (T,)
    discharge: np.ndarray  # (T,)
    price: np.ndarray  # (T,)
    soc: np.ndarray  # (T,)
    soc_within_bounds: np.ndarray  # (T,) bool
    power_limit: float

    def to_frame(self) -> pd.DataFrame:
        T = len(self.price)
        return pd.DataFrame(
            {
                "Renewable (MW)": self.renewable.sum(axis=0).round(6),
                "Thermal (MW)": self.thermal.sum(axis=0).round(6),
                "ESS discharge (MW)": self.discharge.round(6),
                "ESS charge (MW)": self.charge.round(6),
                "SOC (MWh)": self.soc.round(6),
                "SOC within limits": self.soc_within_bounds,
                "Price ($/MWh)": self.price.round(6),
            },
            index=pd.RangeIndex(1, T + 1, name="Interval"),
        )

    def true_cost(self, system: System, storage: StorageParams) -> float:
        """The dispatch's *real* net cost: generation costs plus degradation, evaluated
        at the storage's actual parameters -- not the pi_t bid/offer terms, which are
        only the mechanism `solve_interval` uses to decide dispatch, not a real cost."""
        dt = system.dt
        gen_cost = sum(gen.cost * self.thermal[g].sum() for g, gen in enumerate(system.thermal))
        gen_cost += sum(gen.cost * self.renewable[r].sum() for r, gen in enumerate(system.renewables))
        degradation = storage.epsilon * self.charge.sum() + storage.zeta * self.discharge.sum()
        return dt * (gen_cost + degradation)


def solve_sequential(
    system: System, storage: StorageParams, pi: np.ndarray, power_limit: float | None = None
) -> SequentialDispatchResult:
    """Clears every interval of `system` independently (Sec. above), one at a time in
    order, offering/bidding storage against `pi[t]` each time. `power_limit` defaults to
    `s_max/dt` -- full charge or discharge within a single interval -- since Baldick's
    own storages have no explicit power rating (`p_max`/`g_max` default to infinite,
    fine for a *jointly*-optimized horizon but not for intervals cleared one by one).
    """
    T = system.T
    dt = system.dt
    if power_limit is None:
        power_limit = storage.s_max / dt

    thermal_arr = np.zeros((len(system.thermal), T))
    renewable_arr = np.zeros((len(system.renewables), T))
    charge_arr = np.zeros(T)
    discharge_arr = np.zeros(T)
    price_arr = np.zeros(T)
    soc_arr = np.zeros(T)

    soc_prev = storage.s0
    for t in range(T):
        r = solve_interval(system, t, float(pi[t]), storage, power_limit)
        thermal_arr[:, t] = r.thermal
        renewable_arr[:, t] = r.renewable
        charge_arr[t] = r.charge
        discharge_arr[t] = r.discharge
        price_arr[t] = r.price

        soc_prev = storage.gamma * soc_prev + dt * storage.alpha * r.charge - dt * storage.beta * r.discharge
        soc_arr[t] = soc_prev

    within_bounds = (soc_arr >= storage.s_min - 1e-6) & (soc_arr <= storage.s_max + 1e-6)

    return SequentialDispatchResult(
        thermal=thermal_arr,
        renewable=renewable_arr,
        charge=charge_arr,
        discharge=discharge_arr,
        price=price_arr,
        soc=soc_arr,
        soc_within_bounds=within_bounds,
        power_limit=power_limit,
    )


def main() -> None:
    from experiments.reproduce_baldick import CASES, run_case

    for name in CASES:
        system, _ = CASES[name]
        baldick = run_case(name)
        pi = baldick.pi[0]
        storage = system.storages[0]

        seq = solve_sequential(system, storage, pi)
        print(f"\n=== {name} ===")
        print(seq.to_frame().round(3).to_string())
        print(f"Baldick's own true cost ($):     {baldick.objective:,.2f}")
        print(f"Sequential model's true cost ($): {seq.true_cost(system, storage):,.2f}")
        if not seq.soc_within_bounds.all():
            print("WARNING: derived SOC violates the storage's own physical limits in at least one interval.")


if __name__ == "__main__":
    main()
