from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

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


def _bound(value: float) -> float | None:
    return None if math.isinf(value) else value


def solve_day_ahead(system: System, terminal_value: TerminalValueFunction) -> DispatchResult:
    """Solve the ISO welfare-maximizing dispatch (9) as a linear program, with the
    complementarity condition p_t perp g_t (3) relaxed as in the paper's own examples.

    min  dt*sum(cost_g * thermal_g,t) + dt*sum(cost_r * renewable_r,t)
       + dt*sum(epsilon*p_t + zeta*g_t) - b(s_T - sbar_T)
    s.t. SOC dynamics (4), SOC limits (6)-(7), power balance, generation and storage bounds.
    """
    T = system.T
    dt = system.dt
    storage = system.storage
    sbar_T = storage.sbar_T(T)
    n_th, n_re = len(system.thermal), len(system.renewables)
    upside, downside = terminal_value.lp_segments()
    n_up, n_down = len(upside), len(downside)

    idx: dict[str, slice] = {}
    cursor = 0

    def alloc(name: str, count: int) -> None:
        nonlocal cursor
        idx[name] = slice(cursor, cursor + count)
        cursor += count

    alloc("thermal", n_th * T)
    alloc("renewable", n_re * T)
    alloc("charge", T)
    alloc("discharge", T)
    alloc("soc", T)
    alloc("up", n_up)
    alloc("down", n_down)
    n_vars = cursor

    th_idx = lambda g, t: idx["thermal"].start + g * T + t
    re_idx = lambda r, t: idx["renewable"].start + r * T + t
    p_idx = lambda t: idx["charge"].start + t
    g_idx = lambda t: idx["discharge"].start + t
    s_idx = lambda t: idx["soc"].start + t

    c = np.zeros(n_vars)
    for g, gen in enumerate(system.thermal):
        c[[th_idx(g, t) for t in range(T)]] = dt * gen.cost
    for r, gen in enumerate(system.renewables):
        c[[re_idx(r, t) for t in range(T)]] = dt * gen.cost
    c[idx["charge"]] = dt * storage.epsilon
    c[idx["discharge"]] = dt * storage.zeta
    for k, (_, mv) in enumerate(upside):
        c[idx["up"].start + k] = -mv
    for k, (_, mv) in enumerate(downside):
        c[idx["down"].start + k] = mv

    bounds: list[tuple[float, float | None]] = [(0.0, None)] * n_vars
    for g, gen in enumerate(system.thermal):
        for t in range(T):
            bounds[th_idx(g, t)] = (gen.p_min, _bound(gen.p_max))
    for r, gen in enumerate(system.renewables):
        for t in range(T):
            bounds[re_idx(r, t)] = (0.0, gen.capacity[t])
    for t in range(T):
        bounds[p_idx(t)] = (0.0, _bound(storage.p_max))
        bounds[g_idx(t)] = (0.0, _bound(storage.g_max))
        bounds[s_idx(t)] = (storage.s_min, storage.s_max)
    for k, (width, _) in enumerate(upside):
        bounds[idx["up"].start + k] = (0.0, _bound(width))
    for k, (width, _) in enumerate(downside):
        bounds[idx["down"].start + k] = (0.0, _bound(width))

    A_eq, b_eq = [], []

    # SOC dynamics (4): s_t - gamma*s_{t-1} - dt*alpha*p_t + dt*beta*g_t = 0
    for t in range(T):
        row = np.zeros(n_vars)
        row[s_idx(t)] = 1.0
        row[p_idx(t)] = -dt * storage.alpha
        row[g_idx(t)] = dt * storage.beta
        if t == 0:
            rhs = storage.gamma * storage.s0
        else:
            row[s_idx(t - 1)] = -storage.gamma
            rhs = 0.0
        A_eq.append(row)
        b_eq.append(rhs)

    # power balance: thermal + renewable + discharge - charge = demand
    for t in range(T):
        row = np.zeros(n_vars)
        for g in range(n_th):
            row[th_idx(g, t)] = 1.0
        for r in range(n_re):
            row[re_idx(r, t)] = 1.0
        row[g_idx(t)] = 1.0
        row[p_idx(t)] = -1.0
        A_eq.append(row)
        b_eq.append(system.demand[t])

    # terminal deviation link: s_T - sum(up) + sum(down) = sbar_T
    row = np.zeros(n_vars)
    row[s_idx(T - 1)] = 1.0
    for k in range(n_up):
        row[idx["up"].start + k] = -1.0
    for k in range(n_down):
        row[idx["down"].start + k] = 1.0
    A_eq.append(row)
    b_eq.append(sbar_T)

    A_ub, b_ub = [], []

    # (6): gamma*s_{t-1} + dt*alpha*p_t <= s_max
    for t in range(T):
        row = np.zeros(n_vars)
        row[p_idx(t)] = dt * storage.alpha
        if t == 0:
            rhs = storage.s_max - storage.gamma * storage.s0
        else:
            row[s_idx(t - 1)] = storage.gamma
            rhs = storage.s_max
        A_ub.append(row)
        b_ub.append(rhs)

    # (7): gamma*s_{t-1} - dt*beta*g_t >= s_min  <=>  -gamma*s_{t-1} + dt*beta*g_t <= -s_min
    for t in range(T):
        row = np.zeros(n_vars)
        row[g_idx(t)] = dt * storage.beta
        if t == 0:
            rhs = storage.gamma * storage.s0 - storage.s_min
        else:
            row[s_idx(t - 1)] = -storage.gamma
            rhs = -storage.s_min
        A_ub.append(row)
        b_ub.append(rhs)

    result = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"day-ahead LP failed to solve: {result.message}")

    x = result.x
    thermal = x[idx["thermal"]].reshape(n_th, T) if n_th else np.zeros((0, T))
    renewable = x[idx["renewable"]].reshape(n_re, T) if n_re else np.zeros((0, T))
    charge = x[idx["charge"]]
    discharge = x[idx["discharge"]]
    soc = x[idx["soc"]]

    # power-balance equality rows follow the T SOC-dynamics rows in A_eq. Divide by dt
    # because the balance constraint is in MW while the dual is $ per unit of that
    # constraint's RHS (MW); dt converts it to the conventional $/MWh LMP.
    price = result.eqlin.marginals[T : T + T] / dt

    deviation = float(soc[-1] - sbar_T)
    term_value = terminal_value.value(deviation)

    revenue = dt * float(np.dot(price, discharge))
    charge_cost = dt * float(np.dot(price, charge))
    degradation = dt * (storage.epsilon * float(np.sum(charge)) + storage.zeta * float(np.sum(discharge)))
    ess_surplus = revenue - charge_cost + term_value - degradation

    return DispatchResult(
        thermal=thermal,
        renewable=renewable,
        charge=charge,
        discharge=discharge,
        soc=soc,
        price=price,
        objective=float(result.fun),
        terminal_value=term_value,
        ess_surplus=ess_surplus,
    )
