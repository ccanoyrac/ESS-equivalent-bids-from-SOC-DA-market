"""The conventional market: the same thermal/wind/PV supply stack as `baldick.py`, but
storage participates as an ordinary price-quantity curve (`curves.Block` stacks) instead
of having its SOC trajectory jointly optimised. Each period is cleared independently --
that lack of a joint SOC constraint *is* the model, not an omission (see
`Archive/baldick_soc_reproduction/experiments/current_market.py`'s docstring for the
same point made about the earlier, narrower version of this experiment). The resulting
SOC path is tracked afterwards purely as bookkeeping (GUIDE.md section 8/11): nothing
here enforces `s_min <= s_t <= s_max`.
"""

from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from src.baldick import StorageSpec
from src.curves import Block


@dataclass
class PeriodResult:
    thermal: np.ndarray  # (n_blocks,) MW
    wind: float
    pv: float
    discharge: float  # MWh delivered this period
    charge: float  # MWh bought this period
    price: float  # EUR/MWh


def clear_period(
    thermal: pd.DataFrame,
    demand_t: float,
    wind_avail_t: float,
    pv_avail_t: float,
    discharge_blocks: list[Block],
    charge_blocks: list[Block],
    dt: float = 1.0,
) -> PeriodResult:
    blocks = thermal.reset_index(drop=True)
    n_blocks = len(blocks)
    p_max = blocks["PMax_MW"].to_numpy()
    mc = blocks["MC_eur_MWh"].to_numpy()

    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model("period", env=env) as model:
        g = model.addVars(n_blocks, lb=0.0, ub=[float(x) for x in p_max], name="g")
        w = model.addVar(lb=0.0, ub=float(wind_avail_t), name="wind")
        p = model.addVar(lb=0.0, ub=float(pv_avail_t), name="pv")
        disc = [
            model.addVar(lb=0.0, ub=blk.quantity, name=f"disc[{k}]") for k, blk in enumerate(discharge_blocks)
        ]
        chg = [model.addVar(lb=0.0, ub=blk.quantity, name=f"chg[{k}]") for k, blk in enumerate(charge_blocks)]

        balance = model.addConstr(
            gp.quicksum(g[b] for b in range(n_blocks)) + w + p
            + gp.quicksum(disc) - gp.quicksum(chg) == float(demand_t),
            name="balance",
        )

        model.setObjective(
            dt * gp.quicksum(mc[b] * g[b] for b in range(n_blocks))
            + dt * gp.quicksum(blk.price * var for blk, var in zip(discharge_blocks, disc))
            - dt * gp.quicksum(blk.price * var for blk, var in zip(charge_blocks, chg)),
            GRB.MINIMIZE,
        )
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"conventional-market period LP failed to solve: Gurobi status {model.Status}")

        return PeriodResult(
            thermal=np.array([g[b].X for b in range(n_blocks)]),
            wind=w.X,
            pv=p.X,
            discharge=sum(v.X for v in disc),
            charge=sum(v.X for v in chg),
            price=balance.Pi / dt + 0.0,
        )


@dataclass
class ConventionalResult:
    thermal: np.ndarray  # (n_blocks, T) MW
    wind: np.ndarray  # (T,)
    pv: np.ndarray  # (T,)
    discharge: np.ndarray  # (T,)
    charge: np.ndarray  # (T,)
    price: np.ndarray  # (T,)
    soc: np.ndarray  # (T,) MWh, derived bookkeeping, not enforced
    soc_within_bounds: np.ndarray  # (T,) bool
    thermal_names: tuple[str, ...]
    dt: float

    def to_frame(self) -> pd.DataFrame:
        T = len(self.price)
        return pd.DataFrame(
            {
                "Thermal (MW)": self.thermal.sum(axis=0).round(6),
                "Wind (MW)": self.wind.round(6),
                "PV (MW)": self.pv.round(6),
                "Discharge (MW)": self.discharge.round(6),
                "Charge (MW)": self.charge.round(6),
                "SOC (MWh)": self.soc.round(6),
                "SOC within limits": self.soc_within_bounds,
                "Price (EUR/MWh)": self.price.round(6),
            },
            index=pd.RangeIndex(1, T + 1, name="Interval"),
        )

    def true_cost(self, thermal: pd.DataFrame, storage: StorageSpec) -> float:
        """Real net cost of the dispatch (fuel + degradation), evaluated at the
        storage's actual parameters -- not the committed curve prices, which are only
        the mechanism used to decide dispatch."""
        mc = thermal.reset_index(drop=True)["MC_eur_MWh"].to_numpy()
        gen_cost = float((mc[:, None] * self.thermal).sum())
        degradation = storage.epsilon * self.charge.sum() + storage.zeta * self.discharge.sum()
        return self.dt * (gen_cost + degradation)


def clear_conventional(
    thermal: pd.DataFrame,
    demand: np.ndarray,
    wind_avail: np.ndarray,
    pv_avail: np.ndarray,
    storage: StorageSpec,
    discharge_curve: list[list[Block]],
    charge_curve: list[list[Block]],
    dt: float = 1.0,
) -> ConventionalResult:
    """`discharge_curve[t]`/`charge_curve[t]` are the committed block stacks for period
    `t` (from `curves.build_curve` or `curves.single_price_curve`) -- fixed *before*
    `wind_avail`/`pv_avail` (the realised scenario) is known, which is the entire point
    of the comparison against `baldick.solve_baldick`.
    """
    T = len(demand)
    blocks = thermal.reset_index(drop=True)
    n_blocks = len(blocks)

    thermal_arr = np.zeros((n_blocks, T))
    wind_arr = np.zeros(T)
    pv_arr = np.zeros(T)
    discharge_arr = np.zeros(T)
    charge_arr = np.zeros(T)
    price_arr = np.zeros(T)
    soc_arr = np.zeros(T)

    soc_prev = storage.s0
    for t in range(T):
        r = clear_period(thermal, demand[t], wind_avail[t], pv_avail[t], discharge_curve[t], charge_curve[t], dt)
        thermal_arr[:, t] = r.thermal
        wind_arr[t] = r.wind
        pv_arr[t] = r.pv
        discharge_arr[t] = r.discharge
        charge_arr[t] = r.charge
        price_arr[t] = r.price

        soc_prev = soc_prev + storage.alpha * r.charge - r.discharge / storage.beta
        soc_arr[t] = soc_prev

    within_bounds = (soc_arr >= storage.s_min - 1e-6) & (soc_arr <= storage.s_max + 1e-6)

    return ConventionalResult(
        thermal=thermal_arr,
        wind=wind_arr,
        pv=pv_arr,
        discharge=discharge_arr,
        charge=charge_arr,
        price=price_arr,
        soc=soc_arr,
        soc_within_bounds=within_bounds,
        thermal_names=tuple(blocks["Block"]),
        dt=dt,
    )
