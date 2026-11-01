from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from core.system import System
from terminal_value.base import TerminalValueFunction


@dataclass
class DispatchResult:
    """Solution of the day-ahead ISO problem (paper eq. 9, generalized to N storage units):
    dispatch, SOC trajectories, market-clearing prices (from the power-balance duals), and
    the resulting ESS surplus. `charge`, `discharge`, and `soc` are (n_storage, T); so are
    `terminal_value` and `ess_surplus`, one entry per storage unit.

    `pi`, `nu_bar`, and `nu` are the remaining KKT duals from the paper's storage block
    (Eq. 8 / Table 1): `pi` is the SOC costate (dual of the SOC dynamics, Eq. 4 -- the
    marginal value of stored energy, $/MWh of SOC), and `nu_bar`/`nu` are the duals of the
    tightened upper/lower SOC bounds (Eqs. 6-7). All three are (n_storage, T), like `soc`."""

    thermal: np.ndarray
    renewable: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    price: np.ndarray
    objective: float
    terminal_value: np.ndarray
    ess_surplus: np.ndarray
    pi: np.ndarray
    nu_bar: np.ndarray
    nu: np.ndarray

    @property
    def terminal_soc(self) -> np.ndarray:
        return self.soc[:, -1]

    def to_frame(self) -> pd.DataFrame:
        """System-wide dispatch table, every participant type summed (paper Tables 2-4)."""
        T = len(self.price)
        return pd.DataFrame(
            {
                "Renewable (MW)": self.renewable.sum(axis=0).round(6),
                "Thermal (MW)": self.thermal.sum(axis=0).round(6),
                "ESS discharge (MW)": self.discharge.sum(axis=0).round(6),
                "ESS charge (MW)": self.charge.sum(axis=0).round(6),
                "SOC (MWh)": self.soc.sum(axis=0).round(6),
                "Price ($/MWh)": self.price.round(6),
            },
            index=pd.RangeIndex(1, T + 1, name="Interval"),
        )

    def to_owner_frame(self, system: System) -> pd.DataFrame:
        """Same quantities as `to_frame()`, broken out by individual owner instead of
        summed -- e.g. for seeing how each thermal/renewable/storage participant in
        `experiments/Multi_Participant_Baldick.py` fares individually. Columns are a
        2-level MultiIndex (quantity, owner name); Price is system-wide, so its owner
        level is blank.
        """
        T = len(self.price)
        index = pd.RangeIndex(1, T + 1, name="Interval")

        def block(values: np.ndarray, label: str, names: list[str]) -> pd.DataFrame:
            return pd.DataFrame(values.T.round(6), columns=pd.MultiIndex.from_product([[label], names]), index=index)

        parts = [
            block(self.renewable, "Renewable (MW)", [g.name for g in system.renewables]),
            block(self.thermal, "Thermal (MW)", [g.name for g in system.thermal]),
            block(self.discharge, "ESS discharge (MW)", [s.name for s in system.storages]),
            block(self.charge, "ESS charge (MW)", [s.name for s in system.storages]),
            block(self.soc, "SOC (MWh)", [s.name for s in system.storages]),
        ]
        price_col = pd.DataFrame({("Price ($/MWh)", ""): self.price.round(6)}, index=index)
        return pd.concat(parts + [price_col], axis=1)

    def to_dual_frame(self, system: System) -> pd.DataFrame:
        """The ISO's dual variables (paper Table 1 layout): the SOC costate `pi` (dual of
        the SOC dynamics -- the storage's marginal value of stored energy), the tightened
        SOC bound duals `nu_bar`/`nu`, and the nodal price `lambda`. One (owner-labelled)
        row per storage unit for pi/nu_bar/nu, one system-wide row for lambda; columns are
        intervals, transposed relative to `to_frame()`/`to_owner_frame()` to match the
        paper's own Table 1 (duals as rows, intervals as columns).
        """
        T = len(self.price)
        columns = pd.RangeIndex(1, T + 1, name="Interval")
        names = [s.name for s in system.storages]
        multi_owner = len(names) > 1

        rows: dict[str, np.ndarray] = {}
        for i, name in enumerate(names):
            suffix = f" [{name}]" if multi_owner else ""
            rows[f"π -- SOC costate ($/MWh){suffix}"] = self.pi[i].round(6)
            rows[f"ν̄ -- SOC upper-bound dual ($/MWh){suffix}"] = self.nu_bar[i].round(6)
            rows[f"ν -- SOC lower-bound dual ($/MWh){suffix}"] = self.nu[i].round(6)
        rows["λ -- price ($/MWh)"] = self.price.round(6)

        return pd.DataFrame(rows, index=columns).T

    def to_profit_frame(self, system: System) -> pd.DataFrame:
        """Realized profit per owner, settled at the market-clearing price `price`:
        dt*sum_t (price_t - cost)*dispatch_t for thermal and renewable owners (their
        offer cost is fixed and time-invariant, so this is exactly their producer
        surplus), and `ess_surplus` for storage, which already nets charging cost,
        discharge revenue, degradation, and the terminal value bid.
        """
        dt = system.dt
        profit: dict[str, float] = {}
        for i, gen in enumerate(system.thermal):
            profit[gen.name] = dt * float(np.dot(self.price - gen.cost, self.thermal[i]))
        for i, gen in enumerate(system.renewables):
            profit[gen.name] = dt * float(np.dot(self.price - gen.cost, self.renewable[i]))
        for i, storage in enumerate(system.storages):
            profit[storage.name] = float(self.ess_surplus[i])

        return pd.DataFrame.from_dict(profit, orient="index", columns=["Profit ($)"]).round(2)


def _ub(value: float) -> float:
    return GRB.INFINITY if value == float("inf") else value


def solve_day_ahead(system: System, terminal_values: Sequence[TerminalValueFunction]) -> DispatchResult:
    """Solve the ISO welfare-maximizing dispatch (9) with Gurobi, with the complementarity
    condition p_t perp g_t (3) relaxed as in the paper's own examples.

    min  dt*sum(cost_g * thermal_g,t) + dt*sum(cost_r * renewable_r,t)
       + sum_i [ dt*sum(epsilon_i*p_i,t + zeta_i*g_i,t) - b_i(s_i,T - sbar_i,T) ]
    s.t. per-storage SOC dynamics (4) and SOC limits (6)-(7), one set per storage unit i;
         power balance summed over every thermal, renewable, and storage participant.

    `terminal_values[i]` is storage unit `system.storages[i]`'s own bid; every ESS has an
    independent terminal value function, so heterogeneous willingness-to-pay is exactly
    where different storage owners compete (ROADMAP.md research direction 1).

    Uses the gurobipy package's built-in size-limited free license, which needs no setup
    and comfortably covers problems this size; a real (e.g. academic) license would be
    needed for the large-scale systems described in ROADMAP.md stage 2 (paper Section 4.8
    reports well over 100 ESS on an hourly 24-interval horizon).
    """
    if len(terminal_values) != len(system.storages):
        raise ValueError("terminal_values must have exactly one entry per system.storages unit")

    T = system.T
    dt = system.dt
    n_storage = len(system.storages)
    segments = [tv.lp_segments() for tv in terminal_values]

    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model("day_ahead", env=env) as model:
        thermal = {
            (g, t): model.addVar(lb=gen.p_min, ub=_ub(gen.p_max), name=f"thermal[{gen.name},{t}]")
            for g, gen in enumerate(system.thermal)
            for t in range(T)
        }
        renewable = {
            (r, t): model.addVar(lb=0.0, ub=gen.capacity[t], name=f"renewable[{gen.name},{t}]")
            for r, gen in enumerate(system.renewables)
            for t in range(T)
        }
        charge = {
            (i, t): model.addVar(lb=0.0, ub=_ub(storage.p_max), name=f"charge[{storage.name},{t}]")
            for i, storage in enumerate(system.storages)
            for t in range(T)
        }
        discharge = {
            (i, t): model.addVar(lb=0.0, ub=_ub(storage.g_max), name=f"discharge[{storage.name},{t}]")
            for i, storage in enumerate(system.storages)
            for t in range(T)
        }
        soc = {
            (i, t): model.addVar(lb=storage.s_min, ub=storage.s_max, name=f"soc[{storage.name},{t}]")
            for i, storage in enumerate(system.storages)
            for t in range(T)
        }
        up = [
            [model.addVar(lb=0.0, ub=_ub(width), name=f"up[{storage.name},{k}]") for k, (width, _) in enumerate(segments[i][0])]
            for i, storage in enumerate(system.storages)
        ]
        down = [
            [model.addVar(lb=0.0, ub=_ub(width), name=f"down[{storage.name},{k}]") for k, (width, _) in enumerate(segments[i][1])]
            for i, storage in enumerate(system.storages)
        ]

        def soc_before(i: int, t: int, storage):
            return soc[i, t - 1] if t > 0 else storage.s0

        soc_dynamics = {}
        soc_upper = {}
        soc_lower = {}
        for i, storage in enumerate(system.storages):
            # SOC dynamics (eq. 4): s_t = dt*alpha*p_t - dt*beta*g_t + gamma*s_{t-1}
            # Dual is the paper's pi_t (SOC costate, K4/K5) up to a sign: Gurobi reports
            # the negative of the paper's Lagrange multiplier for this equation as written.
            soc_dynamics[i] = model.addConstrs(
                (
                    soc[i, t]
                    == dt * storage.alpha * charge[i, t] - dt * storage.beta * discharge[i, t] + storage.gamma * soc_before(i, t, storage)
                    for t in range(T)
                ),
                name=f"soc_dynamics[{storage.name}]",
            )
            # Tight SOC upper bound (eq. 6): gamma*s_{t-1} + dt*alpha*p_t <= s_max
            # Dual is the paper's nu_bar_t (K11) up to a sign flip (Gurobi's raw dual on a
            # binding "<=" constraint is <= 0 in a minimization).
            soc_upper[i] = model.addConstrs(
                (storage.gamma * soc_before(i, t, storage) + dt * storage.alpha * charge[i, t] <= storage.s_max for t in range(T)),
                name=f"soc_upper[{storage.name}]",
            )
            # Tight SOC lower bound (eq. 7): gamma*s_{t-1} - dt*beta*g_t >= s_min
            # Dual is the paper's nu_t (K12) directly, no sign flip needed.
            soc_lower[i] = model.addConstrs(
                (storage.gamma * soc_before(i, t, storage) - dt * storage.beta * discharge[i, t] >= storage.s_min for t in range(T)),
                name=f"soc_lower[{storage.name}]",
            )
            # Terminal deviation link: s_i,T = sbar_i,T + sum(up_i) - sum(down_i)
            model.addConstr(
                soc[i, T - 1] - gp.quicksum(up[i]) + gp.quicksum(down[i]) == storage.sbar_T(T),
                name=f"terminal_deviation[{storage.name}]",
            )

        # Power balance: sum(thermal) + sum(renewable) + sum(discharge - charge) = demand
        balance = model.addConstrs(
            (
                gp.quicksum(thermal[g, t] for g in range(len(system.thermal)))
                + gp.quicksum(renewable[r, t] for r in range(len(system.renewables)))
                + gp.quicksum(discharge[i, t] - charge[i, t] for i in range(n_storage))
                == system.demand[t]
                for t in range(T)
            ),
            name="balance",
        )

        model.setObjective(
            dt * gp.quicksum(gen.cost * thermal[g, t] for g, gen in enumerate(system.thermal) for t in range(T))
            + dt * gp.quicksum(gen.cost * renewable[r, t] for r, gen in enumerate(system.renewables) for t in range(T))
            + dt
            * gp.quicksum(
                storage.epsilon * charge[i, t] + storage.zeta * discharge[i, t]
                for i, storage in enumerate(system.storages)
                for t in range(T)
            )
            - gp.quicksum(mv * up[i][k] for i in range(n_storage) for k, (_, mv) in enumerate(segments[i][0]))
            + gp.quicksum(mv * down[i][k] for i in range(n_storage) for k, (_, mv) in enumerate(segments[i][1])),
            GRB.MINIMIZE,
        )

        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"day-ahead LP failed to solve: Gurobi status {model.Status}")

        thermal_arr = np.array([[thermal[g, t].X for t in range(T)] for g in range(len(system.thermal))]).reshape(
            len(system.thermal), T
        )
        renewable_arr = np.array([[renewable[r, t].X for t in range(T)] for r in range(len(system.renewables))]).reshape(
            len(system.renewables), T
        )
        charge_arr = np.array([[charge[i, t].X for t in range(T)] for i in range(n_storage)])
        discharge_arr = np.array([[discharge[i, t].X for t in range(T)] for i in range(n_storage)])
        soc_arr = np.array([[soc[i, t].X for t in range(T)] for i in range(n_storage)])

        # The balance constraint's dual is $ per unit of its MW right-hand side; divide by
        # dt to convert to the conventional $/MWh LMP.
        price = np.array([balance[t].Pi for t in range(T)]) / dt
        # `+ 0.0` folds any -0.0 (an equality/bound that happened to price at exactly
        # zero) to +0.0, since Gurobi's raw sign for a non-binding dual is otherwise
        # arbitrary and would otherwise print as a stray "-0.000".
        pi_arr = np.array([[-soc_dynamics[i][t].Pi for t in range(T)] for i in range(n_storage)]) + 0.0
        nu_bar_arr = np.array([[-soc_upper[i][t].Pi for t in range(T)] for i in range(n_storage)]) + 0.0
        nu_arr = np.array([[soc_lower[i][t].Pi for t in range(T)] for i in range(n_storage)]) + 0.0
        objective = model.ObjVal

    term_values = np.array(
        [
            terminal_values[i].value(float(soc_arr[i, -1] - system.storages[i].sbar_T(T)))
            for i in range(n_storage)
        ]
    )
    revenue = dt * discharge_arr @ price
    charge_cost = dt * charge_arr @ price
    degradation = dt * np.array(
        [
            storage.epsilon * charge_arr[i].sum() + storage.zeta * discharge_arr[i].sum()
            for i, storage in enumerate(system.storages)
        ]
    )
    ess_surplus = revenue - charge_cost + term_values - degradation

    return DispatchResult(
        thermal=thermal_arr,
        renewable=renewable_arr,
        charge=charge_arr,
        discharge=discharge_arr,
        soc=soc_arr,
        price=price,
        objective=float(objective),
        terminal_value=term_values,
        ess_surplus=ess_surplus,
        pi=pi_arr,
        nu_bar=nu_bar_arr,
        nu=nu_arr,
    )
