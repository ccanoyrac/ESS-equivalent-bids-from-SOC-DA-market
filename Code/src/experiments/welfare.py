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


@dataclass
class ProfitResult:
    profit: float  # $, = energy_payments - imbalance_penalty - degradation + terminal_value
    energy_payments: float  # $, sum_t lam_cd[t]*(g_rt[t] - p_rt[t])*dT
    imbalance_penalty: float  # $, psi * deviation_mwh
    degradation: float  # $, (eps*p_rt + zeta*g_rt)*dT
    terminal_value: float  # $, w * s_T
    deviation_mwh: float  # MWh, sum_t |(g_rt - p_rt) - (g_da - p_da)|*dT


def ess_profit(
    S: dict,
    p_da: np.ndarray, g_da: np.ndarray,  # (T,) MW, sold/bought day-ahead
    p_rt: np.ndarray, g_rt: np.ndarray,  # (T,) MW, what the battery actually did (after repair)
    s_T: float,  # MWh, terminal SOC after repair
    lam_cd: np.ndarray,  # (T,) $/MWh, day-ahead clearing price
    psi: float,  # $/MWh, real-time redispatch premium
    dT: float,
) -> ProfitResult:
    """The battery owner's profit for the day, at its TRUE parameters:

        profit = dT*sum_t lam_cd[t]*(g_da - p_da)                      day-ahead settlement
               + dT*sum_t lam_cd[t]*((g_rt - g_da) - (p_rt - p_da))    deviations at the day-ahead price
               - psi * dT*sum_t |(g_rt - p_rt) - (g_da - p_da)|        imbalance penalty
               - dT*(eps*sum p_rt + zeta*sum g_rt)                     degradation on what it did
               + w*s_T                                                  energy kept, at the true w

    The first two lines add up to dT*sum lam_cd*(g_rt - p_rt). The penalty charges the
    battery the redispatch premium psi that welfare already counts for every MWh it deviates
    from its day-ahead position, so a battery that sells what it cannot deliver pays for it
    (choice confirmed with the user, 2026-11-11). The terminal value makes the profit
    comparable with welfare, which values s_T at w too."""
    net_rt = g_rt - p_rt
    net_da = g_da - p_da
    energy_payments = dT * float(np.sum(lam_cd * net_rt))
    deviation_mwh = dT * float(np.sum(np.abs(net_rt - net_da)))
    imbalance_penalty = psi * deviation_mwh
    degradation = dT * float(S["eps"] * p_rt.sum() + S["zet"] * g_rt.sum())
    terminal_value = float(S["w"] * s_T)
    assert deviation_mwh >= 0 and degradation >= 0
    return ProfitResult(
        profit=energy_payments - imbalance_penalty - degradation + terminal_value,
        energy_payments=energy_payments,
        imbalance_penalty=imbalance_penalty,
        degradation=degradation,
        terminal_value=terminal_value,
        deviation_mwh=deviation_mwh,
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
