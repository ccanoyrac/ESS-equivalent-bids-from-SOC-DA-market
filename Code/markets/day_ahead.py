from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from core.system import System
from terminal_value.base import TerminalValueFunction


@dataclass
class DispatchResult:
    """Solution of the day-ahead ISO problem (paper eq. 9): dispatch, SOC trajectory,
    market-clearing prices (from the power-balance duals), and the resulting ESS surplus."""

    thermal: np.ndarray
    renewable: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    price: np.ndarray
    objective: float
    terminal_value: float
    ess_surplus: float

    @property
    def terminal_soc(self) -> float:
        return float(self.soc[-1])

    def to_frame(self) -> pd.DataFrame:
        """Interval-by-interval dispatch table, generators summed by type (paper Tables 2-4)."""
        T = len(self.price)
        return pd.DataFrame(
            {
                "Renewable (MW)": self.renewable.sum(axis=0).round(6),
                "Thermal (MW)": self.thermal.sum(axis=0).round(6),
                "ESS discharge (MW)": self.discharge.round(6),
                "ESS charge (MW)": self.charge.round(6),
                "SOC (MWh)": self.soc.round(6),
                "Price ($/MWh)": self.price.round(6),
            },
            index=pd.RangeIndex(1, T + 1, name="Interval"),
        )


def _ub(value: float) -> float:
    return GRB.INFINITY if value == float("inf") else value


def solve_day_ahead(system: System, terminal_value: TerminalValueFunction) -> DispatchResult:
    """Solve the ISO welfare-maximizing dispatch (9) with Gurobi, with the complementarity
    condition p_t perp g_t (3) relaxed as in the paper's own examples.

    min  dt*sum(cost_g * thermal_g,t) + dt*sum(cost_r * renewable_r,t)
       + dt*sum(epsilon*p_t + zeta*g_t) - b(s_T - sbar_T)
    s.t. SOC dynamics (4), SOC limits (6)-(7), power balance, generation and storage bounds.

    Uses the gurobipy package's built-in size-limited free license, which needs no setup
    and comfortably covers problems this size; a real (e.g. academic) license would be
    needed for the large-scale systems described in ROADMAP.md stage 2 (paper Section 4.8
    reports well over 100 ESS on an hourly 24-interval horizon).
    """
    T = system.T
    dt = system.dt
    storage = system.storage
    sbar_T = storage.sbar_T(T)
    upside, downside = terminal_value.lp_segments()

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
        charge = model.addVars(T, lb=0.0, ub=_ub(storage.p_max), name="charge")
        discharge = model.addVars(T, lb=0.0, ub=_ub(storage.g_max), name="discharge")
        soc = model.addVars(T, lb=storage.s_min, ub=storage.s_max, name="soc")
        up = [model.addVar(lb=0.0, ub=_ub(width), name=f"up[{k}]") for k, (width, _) in enumerate(upside)]
        down = [model.addVar(lb=0.0, ub=_ub(width), name=f"down[{k}]") for k, (width, _) in enumerate(downside)]

        def soc_before(t: int):
            return soc[t - 1] if t > 0 else storage.s0

        # SOC dynamics (eq. 4): s_t = dt*alpha*p_t - dt*beta*g_t + gamma*s_{t-1}
        model.addConstrs(
            (
                soc[t] == dt * storage.alpha * charge[t] - dt * storage.beta * discharge[t] + storage.gamma * soc_before(t)
                for t in range(T)
            ),
            name="soc_dynamics",
        )

        # Tight SOC upper bound (eq. 6): gamma*s_{t-1} + dt*alpha*p_t <= s_max
        model.addConstrs(
            (storage.gamma * soc_before(t) + dt * storage.alpha * charge[t] <= storage.s_max for t in range(T)),
            name="soc_upper",
        )

        # Tight SOC lower bound (eq. 7): gamma*s_{t-1} - dt*beta*g_t >= s_min
        model.addConstrs(
            (storage.gamma * soc_before(t) - dt * storage.beta * discharge[t] >= storage.s_min for t in range(T)),
            name="soc_lower",
        )

        # Power balance: thermal + renewable + discharge - charge = demand
        balance = model.addConstrs(
            (
                gp.quicksum(thermal[g, t] for g in range(len(system.thermal)))
                + gp.quicksum(renewable[r, t] for r in range(len(system.renewables)))
                + discharge[t]
                - charge[t]
                == system.demand[t]
                for t in range(T)
            ),
            name="balance",
        )

        # Terminal deviation link: s_T = sbar_T + sum(up) - sum(down)
        model.addConstr(
            soc[T - 1] - gp.quicksum(up) + gp.quicksum(down) == sbar_T,
            name="terminal_deviation",
        )

        model.setObjective(
            dt * gp.quicksum(gen.cost * thermal[g, t] for g, gen in enumerate(system.thermal) for t in range(T))
            + dt * gp.quicksum(gen.cost * renewable[r, t] for r, gen in enumerate(system.renewables) for t in range(T))
            + dt * gp.quicksum(storage.epsilon * charge[t] + storage.zeta * discharge[t] for t in range(T))
            - gp.quicksum(mv * up[k] for k, (_, mv) in enumerate(upside))
            + gp.quicksum(mv * down[k] for k, (_, mv) in enumerate(downside)),
            GRB.MINIMIZE,
        )

        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"day-ahead LP failed to solve: Gurobi status {model.Status}")

        thermal_arr = np.array(
            [[thermal[g, t].X for t in range(T)] for g in range(len(system.thermal))]
        ).reshape(len(system.thermal), T)
        renewable_arr = np.array(
            [[renewable[r, t].X for t in range(T)] for r in range(len(system.renewables))]
        ).reshape(len(system.renewables), T)
        charge_arr = np.array([charge[t].X for t in range(T)])
        discharge_arr = np.array([discharge[t].X for t in range(T)])
        soc_arr = np.array([soc[t].X for t in range(T)])

        # The balance constraint's dual is $ per unit of its MW right-hand side; divide by
        # dt to convert to the conventional $/MWh LMP.
        price = np.array([balance[t].Pi for t in range(T)]) / dt
        objective = model.ObjVal

    deviation = float(soc_arr[-1] - sbar_T)
    term_value = terminal_value.value(deviation)

    revenue = dt * float(np.dot(price, discharge_arr))
    charge_cost = dt * float(np.dot(price, charge_arr))
    degradation = dt * (storage.epsilon * float(np.sum(charge_arr)) + storage.zeta * float(np.sum(discharge_arr)))
    ess_surplus = revenue - charge_cost + term_value - degradation

    return DispatchResult(
        thermal=thermal_arr,
        renewable=renewable_arr,
        charge=charge_arr,
        discharge=discharge_arr,
        soc=soc_arr,
        price=price,
        objective=float(objective),
        terminal_value=term_value,
        ess_surplus=ess_surplus,
    )
