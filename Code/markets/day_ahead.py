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
    `terminal_value` and `ess_surplus`, one entry per storage unit."""

    thermal: np.ndarray
    renewable: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    price: np.ndarray
    objective: float
    terminal_value: np.ndarray
    ess_surplus: np.ndarray

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

        for i, storage in enumerate(system.storages):
            # SOC dynamics (eq. 4): s_t = dt*alpha*p_t - dt*beta*g_t + gamma*s_{t-1}
            model.addConstrs(
                (
                    soc[i, t]
                    == dt * storage.alpha * charge[i, t] - dt * storage.beta * discharge[i, t] + storage.gamma * soc_before(i, t, storage)
                    for t in range(T)
                ),
                name=f"soc_dynamics[{storage.name}]",
            )
            # Tight SOC upper bound (eq. 6): gamma*s_{t-1} + dt*alpha*p_t <= s_max
            model.addConstrs(
                (storage.gamma * soc_before(i, t, storage) + dt * storage.alpha * charge[i, t] <= storage.s_max for t in range(T)),
                name=f"soc_upper[{storage.name}]",
            )
            # Tight SOC lower bound (eq. 7): gamma*s_{t-1} - dt*beta*g_t >= s_min
            model.addConstrs(
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
    )
