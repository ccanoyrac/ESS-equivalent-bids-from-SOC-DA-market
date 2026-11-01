"""Model 1 (GUIDE.md section 4): the Baldick SOC-based benchmark LP. The ISO clears
thermal blocks, wind, PV, and one storage device jointly over a horizon, optimising the
storage's SOC trajectory directly (no per-interval storage bid/offer). This is the
upper-bound design the whole thesis measures the conventional-curve design against.

Solver note: GUIDE.md suggests Pyomo/CVXPY + HiGHS ("Gurobi is unnecessary") because the
problem is tiny. On this machine `scipy`'s compiled HiGHS backend fails to import --
Windows Application Control blocks its DLLs -- so this module uses `gurobipy` instead,
exactly as the archived `Archive/baldick_soc_reproduction/` framework did. Nothing about
the model depends on the solver; swap in HiGHS freely on a machine where it loads.

Demand is treated as inelastic (`q_t = demand_t` fixed) -- GUIDE.md section 10 notes this
is fine as long as it's stated: welfare then reduces to `-cost`, since the constant
`U_t(demand_t)` term drops out of every comparison anyway (it's identical across designs).
"""

from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB


@dataclass(frozen=True)
class StorageSpec:
    """One storage device, in GUIDE.md's own notation (NOT the archived framework's --
    here `alpha`/`beta` are charge/discharge *efficiencies*, both <= 1; the archived code
    used `beta` as the inverse of discharge efficiency, which is a different convention).
    """

    alpha: float  # charge efficiency, 0 < alpha <= 1
    beta: float  # discharge efficiency, 0 < beta <= 1
    s_min: float  # MWh
    s_max: float  # MWh
    s0: float  # MWh, initial SOC
    c_max: float  # MW, charge power limit
    d_max: float  # MW, discharge power limit
    epsilon: float = 0.0  # EUR/MWh, charge marginal cost
    zeta: float = 0.0  # EUR/MWh, discharge marginal cost
    b_slope: float = 0.0  # EUR/MWh, linear terminal value b(s) = b_slope * s
    name: str = "storage"

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if not (0.0 < self.beta <= 1.0):
            raise ValueError(f"beta must be in (0, 1], got {self.beta}")
        if not (self.s_min <= self.s0 <= self.s_max):
            raise ValueError(f"s0={self.s0} must lie in [{self.s_min}, {self.s_max}]")
        if self.c_max < 0.0 or self.d_max < 0.0:
            raise ValueError("c_max and d_max must be non-negative")


# GUIDE.md section 3's suggested battery: "100 MW / 400 MWh, alpha=beta=0.95 ... enough
# for a 4-hour shift". b_slope is deliberately left for the caller to set and sweep --
# section 10 warns a careless terminal value (e.g. 0) empties the battery by T and
# contaminates every downstream result.
DEFAULT_STORAGE = StorageSpec(
    alpha=0.95, beta=0.95, s_min=0.0, s_max=400.0, s0=200.0,
    c_max=100.0, d_max=100.0, epsilon=0.0, zeta=0.0, b_slope=0.0,
)


@dataclass
class BaldickResult:
    thermal: np.ndarray  # (n_blocks, T) MW
    wind: np.ndarray  # (T,) MW dispatched (<= availability; curtailed if less)
    pv: np.ndarray  # (T,) MW dispatched
    charge: np.ndarray  # (T,) MW
    discharge: np.ndarray  # (T,) MW
    soc: np.ndarray  # (T,) MWh, end-of-interval SOC (s_1..s_T)
    price: np.ndarray  # (T,) EUR/MWh, lambda_t (dual of balance)
    mu: np.ndarray  # (T,) EUR/MWh, dual of SOC dynamics
    thermal_names: tuple[str, ...]
    generation_cost: float  # EUR, thermal fuel cost only
    degradation_cost: float  # EUR
    terminal_value: float  # EUR, b_slope * s_T realised
    dt: float

    @property
    def social_welfare(self) -> float:
        """`-objective` in the constant-inelastic-demand convention (see module
        docstring): the part of welfare that varies across designs. Valid for comparing
        two designs on the *same* demand series; not a claim about absolute welfare."""
        return self.terminal_value - self.generation_cost - self.degradation_cost

    def to_frame(self) -> pd.DataFrame:
        T = len(self.price)
        return pd.DataFrame(
            {
                "Thermal (MW)": self.thermal.sum(axis=0).round(6),
                "Wind (MW)": self.wind.round(6),
                "PV (MW)": self.pv.round(6),
                "Charge (MW)": self.charge.round(6),
                "Discharge (MW)": self.discharge.round(6),
                "SOC (MWh)": self.soc.round(6),
                "Price (EUR/MWh)": self.price.round(6),
                "mu (EUR/MWh)": self.mu.round(6),
            },
            index=pd.RangeIndex(1, T + 1, name="Interval"),
        )


def solve_baldick(
    thermal: pd.DataFrame,
    demand: np.ndarray,
    wind_avail: np.ndarray,
    pv_avail: np.ndarray,
    storage: StorageSpec,
    dt: float = 1.0,
) -> BaldickResult:
    """Solves GUIDE.md section 4's LP for one scenario (one realised wind/PV path).
    `thermal` is `data.load_units()`'s frame (columns `Block`, `PMax_MW`, `MC_eur_MWh`).
    Wind and PV are dispatchable *down* to zero (curtailable) but free -- this matters
    only in the rare hour where net load would otherwise go negative.
    """
    T = len(demand)
    if len(wind_avail) != T or len(pv_avail) != T:
        raise ValueError("demand, wind_avail, pv_avail must all have length T")

    blocks = thermal.reset_index(drop=True)
    n_blocks = len(blocks)
    p_max = blocks["PMax_MW"].to_numpy()
    mc = blocks["MC_eur_MWh"].to_numpy()

    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model("baldick", env=env) as model:
        g = model.addVars(n_blocks, T, lb=0.0, name="g")
        for b in range(n_blocks):
            for t in range(T):
                g[b, t].ub = float(p_max[b])
        w = model.addVars(T, lb=0.0, ub=[float(x) for x in wind_avail], name="wind")
        p = model.addVars(T, lb=0.0, ub=[float(x) for x in pv_avail], name="pv")
        c = model.addVars(T, lb=0.0, ub=storage.c_max, name="charge")
        d = model.addVars(T, lb=0.0, ub=storage.d_max, name="discharge")
        s = model.addVars(T, lb=storage.s_min, ub=storage.s_max, name="soc")

        balance = {}
        soc_dyn = {}
        for t in range(T):
            balance[t] = model.addConstr(
                gp.quicksum(g[b, t] for b in range(n_blocks)) + w[t] + p[t] + d[t] - c[t]
                == float(demand[t]),
                name=f"balance[{t}]",
            )
            prev = storage.s0 if t == 0 else s[t - 1]
            soc_dyn[t] = model.addConstr(
                s[t] == prev + storage.alpha * c[t] - d[t] / storage.beta,
                name=f"soc_dyn[{t}]",
            )

        gen_cost_expr = dt * gp.quicksum(mc[b] * g[b, t] for b in range(n_blocks) for t in range(T))
        degradation_expr = dt * gp.quicksum(storage.zeta * d[t] + storage.epsilon * c[t] for t in range(T))
        terminal_value_expr = storage.b_slope * s[T - 1]

        model.setObjective(gen_cost_expr + degradation_expr - terminal_value_expr, GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Baldick LP failed to solve: Gurobi status {model.Status}")

        thermal_arr = np.array([[g[b, t].X for t in range(T)] for b in range(n_blocks)])
        wind_arr = np.array([w[t].X for t in range(T)])
        pv_arr = np.array([p[t].X for t in range(T)])
        charge_arr = np.array([c[t].X for t in range(T)])
        discharge_arr = np.array([d[t].X for t in range(T)])
        soc_arr = np.array([s[t].X for t in range(T)])

        price_arr = np.array([balance[t].Pi / dt + 0.0 for t in range(T)])
        # Sign verified empirically (see tests/test_equivalence.py): with the SOC
        # dynamics constraint written as `s[t] == prev + alpha*c[t] - d[t]/beta`,
        # Gurobi's raw dual must be negated to match GUIDE's mu_t (marginal value of
        # stored energy, positive, equal to b_slope at t=T-1 when SOC is interior).
        mu_arr = np.array([-soc_dyn[t].Pi + 0.0 for t in range(T)])

        return BaldickResult(
            thermal=thermal_arr,
            wind=wind_arr,
            pv=pv_arr,
            charge=charge_arr,
            discharge=discharge_arr,
            soc=soc_arr,
            price=price_arr,
            mu=mu_arr,
            thermal_names=tuple(blocks["Block"]),
            generation_cost=float(gen_cost_expr.getValue()),
            degradation_cost=float(degradation_expr.getValue()),
            terminal_value=float(storage.b_slope * soc_arr[T - 1]),
            dt=dt,
        )


def merit_order_price(thermal: pd.DataFrame, net_load: np.ndarray) -> np.ndarray:
    """Independent, non-LP ground truth for GUIDE.md section 9.1's validation test:
    sorts thermal blocks by cost and finds the marginal block for each hour's net load
    (= demand - wind - pv, i.e. with the battery and any uncertainty absent). Used only
    to check `solve_baldick`'s LP-derived prices against, not in the main pipeline.
    """
    blocks = thermal.sort_values("MC_eur_MWh").reset_index(drop=True)
    cum_capacity = blocks["PMax_MW"].cumsum().to_numpy()
    costs = blocks["MC_eur_MWh"].to_numpy()
    total_capacity = cum_capacity[-1]

    prices = np.empty(len(net_load))
    for t, nl in enumerate(net_load):
        if nl <= 0:
            prices[t] = 0.0
            continue
        if nl > total_capacity:
            raise ValueError(f"hour {t}: net load {nl:.1f} MW exceeds total thermal capacity {total_capacity:.1f} MW")
        idx = int(np.searchsorted(cum_capacity, nl, side="left"))
        prices[t] = costs[idx]
    return prices
