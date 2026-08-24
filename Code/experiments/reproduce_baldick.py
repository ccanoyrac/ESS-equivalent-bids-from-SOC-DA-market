"""Reproduce the base-case, high-willingness-to-pay, and increased-max-SOC examples of
Baldick (2026), Section 4.7 (Tables 1-4). Run as `python -m experiments.reproduce_baldick`.
"""

from __future__ import annotations

from core.generator import RenewableGenerator, ThermalGenerator
from core.storage import StorageParams
from core.system import System
from markets.day_ahead import DispatchResult, solve_day_ahead
from terminal_value.constant import constant_willingness_to_pay

DEMAND = (100.0, 200.0, 300.0, 300.0, 200.0, 100.0)
RAC = (200.0, 100.0, 500.0, 500.0, 25.0, 0.0)
DT = 4.0
THERMAL_COST = 50.0
THERMAL_P_MAX = 100.0
ALPHA, BETA, GAMMA = 0.9, 1.0, 1.0
EPSILON = ZETA = 1.0


def _system(s_max: float) -> System:
    return System(
        demand=DEMAND,
        thermal=(ThermalGenerator("thermal", cost=THERMAL_COST, p_max=THERMAL_P_MAX),),
        renewables=(RenewableGenerator("renewable", capacity=RAC),),
        storages=(
            StorageParams(
                alpha=ALPHA, beta=BETA, gamma=GAMMA,
                s_min=0.0, s_max=s_max, s0=0.0,
                epsilon=EPSILON, zeta=ZETA,
            ),
        ),
        dt=DT,
    )


CASES = {
    "base_case": (_system(s_max=800.0), (constant_willingness_to_pay(40.0),)),
    "high_willingness_to_pay": (_system(s_max=800.0), (constant_willingness_to_pay(55.0),)),
    "increased_max_soc": (_system(s_max=1200.0), (constant_willingness_to_pay(40.0),)),
}


def run_case(name: str) -> DispatchResult:
    system, terminal_values = CASES[name]
    return solve_day_ahead(system, terminal_values)


def main() -> None:
    for name in CASES:
        result = run_case(name)
        print(f"\n=== {name} ===")
        print(result.to_frame().round(3).to_string())
        print(f"objective (net cost, $): {result.objective:,.2f}")
        print(f"terminal value b(.) ($): {result.terminal_value.sum():,.2f}")
        print(f"ESS surplus ($): {result.ess_surplus.sum():,.2f}")


if __name__ == "__main__":
    main()
