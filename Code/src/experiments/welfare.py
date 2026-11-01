"""Realised welfare -- experiments.md section 4.6.

    SW_real = -( generation cost + psi*redispatched MWh + VOLL*shed MWh
                 + ESS degradation (eps*p + zeta*g)*dT  -  sum_e w_e * s_e[T] )

Always evaluated at the TRUE ESS parameters, never the bids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WelfareResult:
    SW_real: float  # $
    generation_cost: float  # $
    redispatch_cost: float  # $
    shed_cost: float  # $
    degradation_cost: float  # $
    terminal_value: float  # $


def realised_welfare(
    case: dict,
    q_rt: np.ndarray,  # (n_blocks, T) MW
    ess: list[dict],
    p_rt: list[np.ndarray],  # per ESS, (T,) MW
    g_rt: list[np.ndarray],  # per ESS, (T,) MW
    s_T: list[float],  # per ESS, MWh (terminal SOC after real-time repair)
    shed: np.ndarray,  # (T,) MW
    redispatched_mw: np.ndarray,  # (T,) MW, sum|q_rt-q_da| this hour
    psi: float,
    voll: float,
) -> WelfareResult:
    dT = case["dT"]
    costs = [cost for G in case["gens"] for cost, _ in G["blocks"]]
    # q_rt is (n_blocks, T); cost is per block, constant across t.
    generation_cost = dT * float(sum(costs[j] * q_rt[j].sum() for j in range(len(costs))))

    redispatch_cost = psi * dT * float(redispatched_mw.sum())
    shed_cost = voll * dT * float(shed.sum())
    degradation_cost = dT * float(
        sum(S["eps"] * p_rt[e].sum() + S["zet"] * g_rt[e].sum() for e, S in enumerate(ess))
    )
    terminal_value = float(sum(S["w"] * s_T[e] for e, S in enumerate(ess)))

    SW_real = -(generation_cost + redispatch_cost + shed_cost + degradation_cost - terminal_value)
    return WelfareResult(
        SW_real=SW_real,
        generation_cost=generation_cost,
        redispatch_cost=redispatch_cost,
        shed_cost=shed_cost,
        degradation_cost=degradation_cost,
        terminal_value=terminal_value,
    )


def ess_profit_day_ahead(S: dict, p_da: np.ndarray, g_da: np.ndarray, lam_cd: np.ndarray, dT: float) -> float:
    """Day-ahead settlement profit: revenue from discharging minus cost of charging,
    both at the clearing price `lam_cd`, minus degradation -- the real-time adjustment
    cost an ESS itself causes is not separately settled in this market design, so this
    is reported as the day-ahead profit with that assumption flagged (experiments.md
    4.6: "otherwise report day-ahead profit and flag the assumption")."""
    revenue = dT * float(np.sum(lam_cd * g_da - lam_cd * p_da))
    degradation = dT * float(S["eps"] * p_da.sum() + S["zet"] * g_da.sum())
    return revenue - degradation
