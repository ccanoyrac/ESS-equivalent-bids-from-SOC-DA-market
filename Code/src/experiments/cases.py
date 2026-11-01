"""Test systems -- experiments.md section 3's `cases.py`.

`pilot/pilot2.py` (the file that was meant to define `day_case()`'s actual thermal
fleet / demand / ESS parameters) is not present in this repository -- confirmed absent
by search, not merely misplaced (see Code/README.md "Known gaps"). `day_case()`
therefore reuses `verify_equivalence.baldick_case`, the single-bus system from Baldick
(2026) Sec. 4.7 that `verify_equivalence.py`, `verify_price_based.py` and
`numerical_check.py` already validate the SOC/CD equivalence against. This is a
deliberate substitution confirmed with the user, not a guess: it means the stage-0
gate's *specific* published numbers (11.1% of V, $875, V=$7,866.6/day, ...) are not
targeted here, only its structural checks -- see config.py's stage-0 section and the
stage-0 notebook.
"""

from __future__ import annotations

import verify_equivalence as ve


def day_case(smax: float = 800.0, w: float = 40.0) -> dict:
    """The 'pilot day case', single ESS. See module docstring for the substitution note.
    `smax` (MWh) is the ESS energy capacity, `w` ($/MWh) its terminal willingness to pay
    -- Baldick (2026) Sec. 4.7's own base case is `smax=800, w=40`.
    """
    return ve.baldick_case(smax, w)


def stage1_case(smax_each: float = 400.0, w: tuple[float, ...] = (40.0, 55.0)) -> dict:
    """Stage-1 system: `day_case`'s single bus, generators and demand, with the 800 MWh
    battery split into `len(w)` batteries of `smax_each` MWh that differ only in their
    terminal value `w[e]` ($/MWh). Every other battery parameter is Baldick (2026) Sec.
    4.7's. The split (2 x 400 MWh, w = 40 / 55) is a modelling choice confirmed with the
    user (2026-10-28), not sourced data: total storage stays 800 MWh, so stage-1 numbers
    are comparable with stage 0.
    """
    case = ve.baldick_case(smax_each, w[0])
    base = case["ess"][0]
    case["ess"] = [dict(base, w=float(we)) for we in w]
    case["name"] = f"Stage 1: Baldick 4.7 system, {len(w)} x {smax_each:g} MWh, w = {'/'.join(f'{we:g}' for we in w)}"
    return case


def baldick_cases() -> list[dict]:
    """The three published single-bus variants, for cross-checking against
    `verify_equivalence.py` / `numerical_check.py`'s own results."""
    return [ve.baldick_case(800, 40), ve.baldick_case(800, 55), ve.baldick_case(1200, 40)]


def fleet_case(n_ess: int, share: float) -> dict:
    """Stage-2 system: a day case with `n_ess` heterogeneous ESS totalling `share` of
    peak load. Not implemented: `pilot/multi_ess.py`'s heterogeneous-ESS data is not
    present in this repository, and inventing fleet parameters would mean the stage-2
    results could not be compared to anything -- see experiments.md section 8, "never
    invent data".
    """
    raise NotImplementedError(
        "fleet_case() needs pilot/multi_ess.py's heterogeneous-ESS data, which is not "
        "present in this repository. Stage 2 is out of scope until that file (or "
        "equivalent, explicitly-sourced fleet data) is supplied -- see Code/README.md."
    )
