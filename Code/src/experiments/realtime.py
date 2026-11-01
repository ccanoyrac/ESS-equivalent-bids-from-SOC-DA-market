"""Real-time repair and redispatch -- experiments.md section 4.5.

`pilot/pilot2.feasible_schedule` is not present in this repository; `feasible_schedule`
below is re-derived directly from the ESS's own physical constraints -- the same
(A8)-(A12) rows `verify_price_based.self_schedule` already uses -- not invented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class RepairResult:
    p: np.ndarray  # (T,) MW, SOC-feasible charge
    g: np.ndarray  # (T,) MW, SOC-feasible discharge
    s: np.ndarray  # (T,) MWh, resulting SOC
    undeliverable_mwh: float  # dT * sum(|p-p_da| + |g-g_da|)


def feasible_schedule(S: dict, p_da: np.ndarray, g_da: np.ndarray, T: int, dT: float) -> RepairResult:
    """The SOC-feasible (p, g) closest to (p_da, g_da) in L1 distance -- experiments.md
    4.5 point 1. Standard L1-as-LP trick: p = p_da + up_p - dn_p (up_p, dn_p >= 0), so
    |p - p_da| = up_p + dn_p at the optimum; same for g. Subject to (A8)-(A10): SOC
    dynamics, energy limits, power limits.
    """
    a, b, gam = S["a"], S["b"], S["g"]
    n = 7 * T
    ip = lambda t: t
    ig = lambda t: T + t
    iS = lambda t: 2 * T + t
    iup = lambda t: 3 * T + t
    idn = lambda t: 4 * T + t
    iug = lambda t: 5 * T + t
    idg = lambda t: 6 * T + t

    c = np.zeros(n)
    c[3 * T : 7 * T] = 1.0  # minimise sum(up_p+dn_p+up_g+dn_g)

    Aeq, beq, Aub, bub = [], [], [], []
    for t in range(T):
        r = np.zeros(n)
        r[ip(t)] = 1.0
        r[iup(t)] = -1.0
        r[idn(t)] = 1.0
        Aeq.append(r)
        beq.append(p_da[t])
        r = np.zeros(n)
        r[ig(t)] = 1.0
        r[iug(t)] = -1.0
        r[idg(t)] = 1.0
        Aeq.append(r)
        beq.append(g_da[t])

        r = np.zeros(n)
        r[iS(t)] = 1.0
        r[ip(t)] = -dT * a
        r[ig(t)] = dT * b
        if t > 0:
            r[iS(t - 1)] = -gam
            beq.append(0.0)
        else:
            beq.append(gam * S["s0"])
        Aeq.append(r)

        r = np.zeros(n)
        r[ip(t)] = dT * a
        if t > 0:
            r[iS(t - 1)] = gam
            bub.append(S["smax"])
        else:
            bub.append(S["smax"] - gam * S["s0"])
        Aub.append(r)

        r = np.zeros(n)
        r[ig(t)] = dT * b
        if t > 0:
            r[iS(t - 1)] = -gam
            bub.append(-S["smin"])
        else:
            bub.append(-S["smin"] + gam * S["s0"])
        Aub.append(r)

    bnd = (
        [(0, S["pbar"])] * T
        + [(0, S["gbar"])] * T
        + [(None, None)] * T
        + [(0, None)] * (4 * T)
    )
    res = linprog(c, A_ub=np.array(Aub), b_ub=bub, A_eq=np.array(Aeq), b_eq=beq, bounds=bnd, method="highs")
    assert res.status == 0, res.message

    p = res.x[:T]
    g = res.x[T : 2 * T]
    s = res.x[2 * T : 3 * T]
    undeliverable = dT * float(np.sum(res.x[3 * T :]))
    return RepairResult(p=p, g=g, s=s, undeliverable_mwh=undeliverable)


@dataclass
class RedispatchResult:
    q_rt: np.ndarray  # (n_blocks,) MW, real-time generator dispatch
    shed: float  # MW
    redispatched_mw: float  # sum|q_rt - q_da| this hour


def redispatch_hour(case: dict, t: int, q_da_t: np.ndarray, net_supply_needed: float, psi: float, voll: float) -> RedispatchResult:
    """Generators move from `q_da_t` to meet `net_supply_needed` (= D[t] - g_rt[t] +
    p_rt[t], the ESS's real-time net injection already netted out), each MWh moved
    (either direction) costing its own offer price plus the premium `psi`; any
    remaining shortfall is shed at `voll` -- experiments.md 4.5 point 2.
    """
    costs, caps = [], []
    for G in case["gens"]:
        for cost, qb in G["blocks"]:
            costs.append(cost)
            caps.append(qb[t])
    n = len(costs)
    costs = np.array(costs)
    caps = np.array(caps)
    q0 = np.asarray(q_da_t)

    # q_j = q0_j + up_j - dn_j; minimise (cost_j+psi)*up_j + (psi-cost_j)*dn_j + voll*shed
    # (the constant sum(cost_j*q0_j) is dropped -- it doesn't affect the optimal up/dn).
    c = np.concatenate([costs + psi, psi - costs, [voll]])
    bounds = [(0, caps[j] - q0[j]) for j in range(n)] + [(0, q0[j]) for j in range(n)] + [(0, None)]
    delta_needed = net_supply_needed - float(q0.sum())
    A_eq = [np.concatenate([np.ones(n), -np.ones(n), [1.0]])]
    b_eq = [delta_needed]

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    assert res.status == 0, res.message

    up = res.x[:n]
    dn = res.x[n : 2 * n]
    shed = float(res.x[2 * n])
    q_rt = q0 + up - dn
    return RedispatchResult(q_rt=q_rt, shed=shed, redispatched_mw=float(np.sum(up) + np.sum(dn)))
