"""Many renewable, thermal, and storage participants competing in one day-ahead market,
built on top of Baldick (2026) Section 4.7's system. Run as
`python -m experiments.Multi_Participant_Baldick`.

Modeling choices (the prompt underspecifies exact numbers, so these are documented so
they're easy to change):

- Thermal (N_PARTICIPANTS owners, each capacity ~ Baldick's single 100 MW unit) and
  storage (each s_max = Baldick's single 800 MWh unit) are both built at roughly
  Baldick's *original per-unit scale*, so with N_PARTICIPANTS=4 of each, total thermal
  and storage capacity are both ~4x Baldick's single-participant system. Demand and the
  shared renewable forecast are scaled by the same DEMAND_SCALE factor to match --
  scaling demand alone while leaving renewable flat would leave renewable comically
  undersized relative to the new demand (thermal+renewable capacity would fall short of
  demand in some intervals even with storage straining to cover the gap) and would break
  the renewable-penetration story that makes Baldick's example worth reproducing.
  Scaling both keeps demand:renewable:thermal:storage in Baldick's original proportions,
  just uniformly bigger.
- Renewables all share one base forecast -- Baldick's RAC profile, scaled by
  DEMAND_SCALE and then split evenly across N_PARTICIPANTS owners (their "fair share")
  -- each with its own small, independent, zero-mean deviation from that shared shape,
  clipped so capacity never goes negative.
- Thermal owners each get a capacity drawn uniformly in [100, 150] MW and a cost that's
  Baldick's $50/MWh plus a small, fixed (non-intertemporal -- constant across t, only
  varying owner to owner) deviation, which is what creates a merit-order competition.
- Storage owners share identical technical characteristics (Baldick's base-case values:
  s_max=800 MWh, alpha=0.9, beta=1, degradation $1/MWh) but each posts a different
  constant willingness-to-pay. The four values deliberately straddle Baldick's own
  w=40 and w=55 examples so the spread's effect on terminal SOC and surplus is directly
  comparable to reproduce_baldick.py.
"""

from __future__ import annotations

import numpy as np

from core.generator import RenewableGenerator, ThermalGenerator
from core.storage import StorageParams
from core.system import System
from markets.day_ahead import DispatchResult, solve_day_ahead
from terminal_value.constant import constant_willingness_to_pay

SEED = 7

N_PARTICIPANTS = 4  # owners of each type: renewable, thermal, storage
DEMAND_SCALE = float(N_PARTICIPANTS)

BASE_DEMAND = (100.0, 200.0, 300.0, 300.0, 200.0, 100.0)
DEMAND = tuple(d * DEMAND_SCALE for d in BASE_DEMAND)
BASE_RAC = (200.0, 100.0, 500.0, 500.0, 25.0, 0.0)
DT = 4.0

RENEWABLE_NOISE_FRACTION = 0.08  # std dev of each owner's deviation, as a fraction of its fair share

THERMAL_BASE_COST = 50.0
THERMAL_COST_SPREAD = 15.0  # +/- $/MWh around the base cost, fixed per owner (not intertemporal)
THERMAL_P_MAX_RANGE = (100.0, 150.0)

ALPHA, BETA, GAMMA = 0.9, 1.0, 1.0
EPSILON = ZETA = 1.0
S_MAX = 800.0

# Straddles Baldick's own base case (w=40) and high-willingness-to-pay case (w=55).
STORAGE_WTP = (25.0, 40.0, 55.0, 70.0)


def _renewables(rng: np.random.Generator) -> tuple[RenewableGenerator, ...]:
    fair_share = np.array(BASE_RAC) * DEMAND_SCALE / N_PARTICIPANTS
    generators = []
    for i in range(N_PARTICIPANTS):
        noise = rng.normal(loc=0.0, scale=RENEWABLE_NOISE_FRACTION * fair_share)
        capacity = tuple(np.clip(fair_share + noise, 0.0, None).round(3))
        generators.append(RenewableGenerator(f"renewable_{i + 1}", capacity=capacity))
    return tuple(generators)


def _thermal_units(rng: np.random.Generator) -> tuple[ThermalGenerator, ...]:
    generators = []
    for i in range(N_PARTICIPANTS):
        p_max = round(float(rng.uniform(*THERMAL_P_MAX_RANGE)), 3)
        cost = round(THERMAL_BASE_COST + float(rng.uniform(-THERMAL_COST_SPREAD, THERMAL_COST_SPREAD)), 3)
        generators.append(ThermalGenerator(f"thermal_{i + 1}", cost=cost, p_max=p_max))
    return tuple(generators)


def _storages() -> tuple[StorageParams, ...]:
    return tuple(
        StorageParams(
            alpha=ALPHA, beta=BETA, gamma=GAMMA,
            s_min=0.0, s_max=S_MAX, s0=0.0,
            epsilon=EPSILON, zeta=ZETA,
            name=f"storage_{i + 1}",
        )
        for i in range(N_PARTICIPANTS)
    )


def build_system() -> tuple[System, tuple]:
    rng = np.random.default_rng(SEED)
    system = System(
        demand=DEMAND,
        thermal=_thermal_units(rng),
        renewables=_renewables(rng),
        storages=_storages(),
        dt=DT,
    )
    terminal_values = tuple(constant_willingness_to_pay(w) for w in STORAGE_WTP)
    return system, terminal_values


def run() -> DispatchResult:
    system, terminal_values = build_system()
    return solve_day_ahead(system, terminal_values)


def main() -> None:
    result = run()
    print(result.to_frame().round(3).to_string())
    print(f"objective (net system cost, $): {result.objective:,.2f}")
    print(f"terminal value b(.) total ($):  {result.terminal_value.sum():,.2f}")
    print(f"ESS surplus total ($):          {result.ess_surplus.sum():,.2f}")


if __name__ == "__main__":
    main()
