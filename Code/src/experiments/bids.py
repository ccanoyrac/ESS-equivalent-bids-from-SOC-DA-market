"""Bid families -- experiments.md section 4.3.

Reuses `verify_price_based.self_schedule` / `prop5_polyhedron` / `pi_range` (copied
unedited from `Archive/ess_equivalence_code/`, see `Code/src/verify_price_based.py`'s
own note) for the ESS self-scheduling LP and its dual-multiplier polyhedron; this
module only selects a point in that polyhedron and converts it to bid prices/blocks.

Prices from multipliers (same for every family except B_priv):
    u = alpha*(pi - nu_up) - eps     (charge bid)
    c = zeta  + beta*(pi + nu_lo)    (discharge offer)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

import verify_price_based as vpb


@dataclass
class Bid:
    u: np.ndarray  # (T,) $/MWh, charge bid price
    c: np.ndarray  # (T,) $/MWh, discharge offer price
    p0: np.ndarray  # (T,) MW, planned charge quantity
    g0: np.ndarray  # (T,) MW, planned discharge quantity


def _solver_vertex(poly: dict, T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """experiments.md 4.3's 'solver' rule: the vertex of the Prop. 5 polyhedron
    minimising sum(pi)."""
    e = np.zeros(3 * T)
    e[:T] = 1.0
    r = linprog(
        e, A_ub=poly["Aub"], b_ub=poly["bub"], A_eq=poly["Aeq"], b_eq=poly["beq"],
        bounds=poly["bnd"], method="highs",
    )
    assert r.status == 0, r.message
    return r.x[:T], r.x[T : 2 * T], r.x[2 * T : 3 * T]


def _prelimit_vertex(poly: dict, gamma: float, T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """experiments.md 4.3's 'prelimit' rule: pi[t] = clip(pi[t-1]/gamma, lo_t, hi_t),
    first hour = hi_1 -- the same sequential projection already used (for the chart, not
    for bid construction) by `numerical_check.py`'s own "pre-limit rule" marker.

    pi alone doesn't pin down nu_up/nu_lo: at every hour where neither SOC bound is
    tight (poly['U'][t] and poly['L'][t] both False) both are forced to exactly 0 by
    the polyhedron's own bounds, so there's nothing to choose. At hour 0 they come
    directly from the vertex `pi_range` already solved to get `hi_1` (that vertex
    satisfies every constraint, nu_up[0]/nu_lo[0] included). For hours 1..T-1, the
    polyhedron's own recursion equality pi[t] = gamma*(pi[t+1] - nu_up[t+1] + nu_lo[t+1])
    has at most one free unknown once pi is fixed (the other is forced to 0), so it is
    solved for directly -- this is a derivation from the polyhedron's own constraints,
    not an independent choice.
    """
    rng, pts = vpb.pi_range(poly, T)
    pi = np.empty(T)
    pi[0] = rng[0][1]  # hi_1
    for t in range(1, T):
        lo, hi = rng[t]
        pi[t] = min(max(pi[t - 1] / gamma, lo), hi)

    nu_up = np.zeros(T)
    nu_lo = np.zeros(T)
    v0 = pts[1]  # the vertex pi_range solved to obtain hi_1 (t=0, "max" direction)
    nu_up[0], nu_lo[0] = v0[T + 0], v0[2 * T + 0]
    for t in range(T - 1):
        diff = pi[t + 1] - pi[t] / gamma
        if poly["U"][t + 1]:
            nu_up[t + 1] = max(0.0, diff)
        elif poly["L"][t + 1]:
            nu_lo[t + 1] = max(0.0, -diff)
    return pi, nu_up, nu_lo


def _select_vertex(poly: dict, S: dict, T: int, rule: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rule == "prelimit":
        return _prelimit_vertex(poly, S["g"], T)
    if rule == "solver":
        return _solver_vertex(poly, T)
    raise ValueError(f"unknown pi-selection rule {rule!r} (expected 'prelimit' or 'solver')")


def bid_star(S: dict, pi_star: np.ndarray, nuU_star: np.ndarray, nuL_star: np.ndarray, p_star: np.ndarray, g_star: np.ndarray) -> Bid:
    """B_star: prices from `lam_star`'s own Baldick-solve duals, quantity = the
    benchmark schedule. Must give dSW=0 exactly (the stage-0 gate's first check)."""
    u = S["a"] * (pi_star - nuU_star) - S["eps"]
    c = S["zet"] + S["b"] * (pi_star + nuL_star)
    return Bid(u=u, c=c, p0=p_star, g0=g_star)


def self_schedule(S: dict, lam: np.ndarray, T: int, dT: float, kappa: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P^SS(lam) with a throughput tie-breaker (BidRule.tie_break_reward):

        min  sum_t dT[(eps+lam_t)p_t + (zeta-lam_t)g_t] - w s_T  -  kappa * sum_t dT (p_t+g_t)
        s.t. (A8)-(A12)

    The extra term only enters the objective, so it equals `vpb.self_schedule` run with
    eps -> eps-kappa and zeta -> zeta-kappa (those two enter nothing else) -- no copy of
    its constraints. For small enough kappa the solution is, among the plans optimal for
    the true profit, the one with the most throughput (lexicographic optimum). That is
    enforced: kappa is halved until the plan's true objective matches the true optimum,
    so Prop. 5 (which needs an optimal plan) still applies. kappa=0 returns the solver's
    own vertex. Returns (p, g, s).
    """
    assert 0.0 <= 2.0 * kappa < S["eps"] + S["zet"], "kappa must not make simultaneous charge+discharge free"
    res, c_true = vpb.self_schedule(S, lam, T, dT)
    assert res.status == 0, res.message
    x = res.x
    tol = 1e-9 * max(1.0, abs(res.fun))
    k = kappa
    for _ in range(40):
        if k <= 0.0:
            break
        res_k, _ = vpb.self_schedule({**S, "eps": S["eps"] - k, "zet": S["zet"] - k}, lam, T, dT)
        assert res_k.status == 0, res_k.message
        if c_true @ res_k.x - res.fun <= tol:
            x = res_k.x
            break
        k /= 2.0
    else:
        raise AssertionError(f"tie-breaker: no kappa <= {kappa} keeps the plan optimal for the true profit")
    return x[:T], x[T : 2 * T], x[2 * T :]


def bid_star_p(S: dict, lam_star: np.ndarray, T: int, dT: float, rule: str, kappa: float) -> Bid:
    """B_star_p: prices from `lam_star`, quantity = the ESS's own self-schedule at
    `lam_star` (isolates the quantity-information loss from using a self-schedule
    instead of the benchmark's own joint schedule). `kappa`: tie-breaker, see
    `self_schedule`."""
    p, g, s = self_schedule(S, lam_star, T, dT, kappa)
    poly = vpb.prop5_polyhedron(S, lam_star, p, g, s, T, dT)
    pi, nu_up, nu_lo = _select_vertex(poly, S, T, rule)
    u = S["a"] * (pi - nu_up) - S["eps"]
    c = S["zet"] + S["b"] * (pi + nu_lo)
    return Bid(u=u, c=c, p0=p, g0=g)


def bid_fcst(S: dict, lam_hat_e: np.ndarray, T: int, dT: float, rule: str, kappa: float) -> Bid:
    """B_fcst: identical construction to B_star_p but at the ESS's own *forecast*
    `lam_hat_e` instead of the true `lam_star` -- the main experiment."""
    p, g, s = self_schedule(S, lam_hat_e, T, dT, kappa)
    poly = vpb.prop5_polyhedron(S, lam_hat_e, p, g, s, T, dT)
    pi, nu_up, nu_lo = _select_vertex(poly, S, T, rule)
    u = S["a"] * (pi - nu_up) - S["eps"]
    c = S["zet"] + S["b"] * (pi + nu_lo)
    return Bid(u=u, c=c, p0=p, g0=g)


def bid_priv(S: dict, T: int) -> Bid:
    """B_priv: pi[t] = gamma^(T-t)*w, no rents (nu_up=nu_lo=0), full capacity each hour
    -- the lower bound, using only the ESS's technical data (no price information)."""
    t = np.arange(1, T + 1)
    pi = S["g"] ** (T - t) * S["w"]
    u = S["a"] * pi - S["eps"]
    c = S["zet"] + S["b"] * pi
    return Bid(u=u, c=c, p0=np.full(T, S["pbar"]), g0=np.full(T, S["gbar"]))


def quantity_blocks(bid: Bid, S: dict, delta: float, full_capacity: bool) -> dict:
    """experiments.md 4.3's quantity rule: two blocks per side, split at the planned
    quantity with a +/-`delta` ($/MWh) price spread. `full_capacity=True` is B_priv's
    own rule (one full-capacity block each side, no spread -- it already bids `pbar`,
    `gbar` as `p0`, `g0`, so the second block would be empty anyway)."""
    T = len(bid.u)
    if full_capacity:
        return dict(
            charge=[[(bid.u[t], S["pbar"])] for t in range(T)],
            discharge=[[(bid.c[t], S["gbar"])] for t in range(T)],
        )
    charge = [
        [(bid.u[t] + delta, bid.p0[t]), (bid.u[t] - delta, S["pbar"] - bid.p0[t])] for t in range(T)
    ]
    discharge = [
        [(bid.c[t] - delta, bid.g0[t]), (bid.c[t] + delta, S["gbar"] - bid.g0[t])] for t in range(T)
    ]
    return dict(charge=charge, discharge=discharge)
