"""Solve Baldick's SOC design once per instance -- experiments.md section 4.1.

Returns `lam_star`, each ESS's schedule, `SW_star`, `SW_noESS` and the storage value
`V = SW_star - SW_noESS`. Built entirely on `verify_equivalence.build`/`duals` (never
re-implements the SOC LP).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

import verify_equivalence as ve


def with_shedding_generator(case: dict, voll: float) -> dict:
    """A copy of `case` with one extra generator block per bus, priced at `voll`
    ($/MWh) with capacity equal to that bus's peak demand. Load shedding is not a
    separate variable anywhere in `verify_equivalence.build` (which has none), so it is
    represented the standard way: an arbitrarily expensive but always-available supply
    block. Dispatching this block by `x` MWh in hour `t` is economically identical to
    shedding `x` MWh of load at value `voll` -- experiments.md section 4.4's "load
    shedding at VOLL keeps it feasible".
    """
    out = copy.deepcopy(case)
    T = out["T"]
    for n in range(out["N"]):
        peak = float(out["D"][n].max()) if out["D"].size else 0.0
        cap = max(peak, 1.0) * 2.0  # headroom: never the binding constraint itself
        out["gens"].append(dict(n=n, blocks=[(voll, [cap] * T)], Ru=1e9, Rd=1e9))
    return out


@dataclass
class Benchmark:
    case: dict
    lam_star: np.ndarray  # (N, T) $/MWh
    pi_star: list[np.ndarray]  # per ESS, (T,) $/MWh
    nuU_star: list[np.ndarray]  # per ESS, (T,) $/MWh
    nuL_star: list[np.ndarray]  # per ESS, (T,) $/MWh
    p_star: list[np.ndarray]  # per ESS, (T,) MW charge
    g_star: list[np.ndarray]  # per ESS, (T,) MW discharge
    s_star: list[np.ndarray]  # per ESS, (T,) MWh SOC
    q_star: np.ndarray  # (n_blocks, T) MW, generator blocks in case["gens"] order
    SW_star: float  # $
    SW_noESS: float  # $
    V: float  # $, = SW_star - SW_noESS


def solve_benchmark(case: dict, voll: float = 1000.0) -> Benchmark:
    """experiments.md section 4.1. `SW_star` uses the same welfare function as the
    realised outcome (4.6) evaluated at Baldick's own dispatch (no redispatch, no
    shedding by construction); `SW_noESS` clears the same system with every ESS removed
    and shedding allowed at `voll`.
    """
    T, N = case["T"], case["N"]
    lp, res, _ = ve.build(case, "SOC")
    y, ub, lb = ve.duals(case, lp, res)

    lam_star = np.array([[y["lam", n, t] for t in range(T)] for n in range(N)])
    pi_star, nuU_star, nuL_star, p_star, g_star, s_star = [], [], [], [], [], []
    for e, S in enumerate(case["ess"]):
        pi_star.append(np.array([y["pi", e, t] for t in range(T)]))
        nuU_star.append(np.array([y["nuU", e, t] for t in range(T)]))
        nuL_star.append(np.array([y["nuL", e, t] for t in range(T)]))
        p_star.append(np.array([res.x[lp.idx["p", e, t]] for t in range(T)]))
        g_star.append(np.array([res.x[lp.idx["g", e, t]] for t in range(T)]))
        s_star.append(np.array([res.x[lp.idx["s", e, t]] for t in range(T)]))
    q_star = np.array([
        [res.x[lp.idx["q", i, k, t]] for t in range(T)]
        for i, G in enumerate(case["gens"]) for k in range(len(G["blocks"]))
    ])

    # res.fun = dT*(gen cost + degradation) - sum_e w_e*s_e[T], dropping the terminal
    # value's constant w_e*gamma_e^T*s0_e (see verify_equivalence.build's own comment on
    # the "SOC" objective) -- add it back to get the true `-SW_star`.
    sbb = sum(S["w"] * S["g"] ** T * S["s0"] for S in case["ess"])
    SW_star = -(res.fun + sbb)

    case_noess = copy.deepcopy(case)
    case_noess["ess"] = []
    case_noess_shed = with_shedding_generator(case_noess, voll)
    _, res_noess, _ = ve.build(case_noess_shed, "SOC")
    SW_noESS = -res_noess.fun

    return Benchmark(
        case=case,
        lam_star=lam_star,
        pi_star=pi_star,
        nuU_star=nuU_star,
        nuL_star=nuL_star,
        p_star=p_star,
        g_star=g_star,
        s_star=s_star,
        q_star=q_star,
        SW_star=SW_star,
        SW_noESS=SW_noESS,
        V=SW_star - SW_noESS,
    )
