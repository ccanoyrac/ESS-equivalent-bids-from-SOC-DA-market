"""Two sequential day-ahead markets, for the same multi-participant fleet as
Multi_Participant_Baldick.py: day 1 clears the ordinary system from `build_system()`,
then day 2 clears as an INDEPENDENT market that starts each storage owner exactly where
day 1 left it (its terminal SOC becomes day 2's `s0`) -- not one connected multi-day
optimization. This is meant to model a realistic day-ahead operator: it never sees
tomorrow's market while clearing today's, only today's own data plus whatever physical
state (SOC) storage happens to carry into the next clearing.

Day 2's renewable profile is day 1's own per-owner shape rescaled by a factor, isolating
a one-day-ahead renewable shock (a sharp drop or a sharp surge) that the market has no
foresight into. Storage owners also re-price their willingness-to-pay for day 2 by a
second, independent multiplier -- valuing stored energy *more* when they expect day 2 to
be renewable-scarce, and *less* when they expect it to be renewable-abundant, the same
scarcity logic that sets w in the first place. Run as
`python -m experiments.Multi_Time_Baldick`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from core.generator import RenewableGenerator
from core.storage import StorageParams
from core.system import System
from experiments.Multi_Participant_Baldick import build_system
from markets.day_ahead import DispatchResult, solve_day_ahead
from terminal_value.base import TerminalValueFunction


class MultiTimeCase(NamedTuple):
    renewable_factor: float
    wtp_multiplier: float


# Day 1 is always the ordinary, unscaled system from build_system(); day 2's renewable
# is rescaled by `renewable_factor` relative to its own day-1 shape, and every storage
# owner's willingness-to-pay is rescaled by `wtp_multiplier` -- a sharp renewable drop
# paired with a *higher* valuation of stored energy, and a sharp surge paired with a
# *lower* one, to see how the same fleet responds when the next clearing can only
# inherit day 1's ending SOC, not plan jointly for the shock ahead of time.
MULTI_TIME_CASES = {
    "multi_time_low_renewable": MultiTimeCase(renewable_factor=0.5, wtp_multiplier=1.2),
    "multi_time_high_renewable": MultiTimeCase(renewable_factor=1.5, wtp_multiplier=0.7),
}


@dataclass
class MultiTimeResult:
    """Two sequential, independently-cleared day-ahead markets sharing one fleet of
    owners. `day1` is the ordinary single-day market; `day2` starts each storage at
    `day1`'s own terminal SOC (see `next_day_system`) and clears against a rescaled
    renewable profile and rescaled willingness-to-pay bids (`day2_terminal_values`).
    `day1_system`/`day2_system` are the `System`s that produced each, kept around for
    labelling and dual/profit lookups in the notebook."""

    day1_system: System
    day1: DispatchResult
    day2_system: System
    day2_terminal_values: tuple[TerminalValueFunction, ...]
    day2: DispatchResult


def _carry_over_storages(storages: tuple[StorageParams, ...], terminal_soc: np.ndarray) -> tuple[StorageParams, ...]:
    """A copy of `storages` with `s0` replaced by each owner's terminal SOC from the
    previous clearing; every other technical characteristic is unchanged."""
    return tuple(
        StorageParams(
            alpha=s.alpha, beta=s.beta, gamma=s.gamma,
            s_min=s.s_min, s_max=s.s_max, s0=float(soc),
            p_max=s.p_max, g_max=s.g_max, epsilon=s.epsilon, zeta=s.zeta,
            name=s.name,
        )
        for s, soc in zip(storages, terminal_soc)
    )


def next_day_system(system: System, renewable_factor: float, terminal_soc: np.ndarray) -> System:
    """The next day's system: same demand and thermal fleet as `system`, each
    renewable's capacity is *its own* (this same) day's profile rescaled by
    `renewable_factor`, and each storage starts (`s0`) from `terminal_soc` -- the
    previous day's own ending SOC, carried over as a physical state rather than
    jointly optimized with the previous day. `renewable_factor=1.0` reproduces an
    ordinary day with no shock, just the SOC carryover.
    """
    scaled_renewables = tuple(
        RenewableGenerator(gen.name, capacity=tuple(c * renewable_factor for c in gen.capacity), cost=gen.cost)
        for gen in system.renewables
    )
    return System(
        demand=system.demand,
        thermal=system.thermal,
        renewables=scaled_renewables,
        storages=_carry_over_storages(system.storages, terminal_soc),
        dt=system.dt,
    )


def run_multi_time_case(name: str) -> MultiTimeResult:
    case = MULTI_TIME_CASES[name]
    system, terminal_values = build_system()
    day1 = solve_day_ahead(system, terminal_values)
    day2_system = next_day_system(system, case.renewable_factor, day1.terminal_soc)
    day2_terminal_values = tuple(tv.scaled(case.wtp_multiplier) for tv in terminal_values)
    day2 = solve_day_ahead(day2_system, day2_terminal_values)
    return MultiTimeResult(
        day1_system=system, day1=day1, day2_system=day2_system,
        day2_terminal_values=day2_terminal_values, day2=day2,
    )


def main() -> None:
    for name, case in MULTI_TIME_CASES.items():
        mt = run_multi_time_case(name)
        print(f"\n=== {name} (day 2 RAC x{case.renewable_factor}, day 2 WTP x{case.wtp_multiplier}) ===")
        print("-- Day 1 --")
        print(mt.day1.to_frame().round(3).to_string())
        print("-- Day 2 --")
        print(mt.day2.to_frame().round(3).to_string())
        print(f"day 1 objective ($): {mt.day1.objective:,.2f}   day 2 objective ($): {mt.day2.objective:,.2f}")
        print(f"day 1 -> day 2 carried-over SOC: {mt.day1.terminal_soc.round(2)}")


if __name__ == "__main__":
    main()
