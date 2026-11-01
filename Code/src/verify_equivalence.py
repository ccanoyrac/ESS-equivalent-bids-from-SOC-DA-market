"""
Numerical verification of the KKT equivalence note (complete formulation).

What it does, for each test case:
  1. Solves Baldick's SOC-based clearing problem  P^SOC  (A1)-(A12)  as an LP.
  2. Recovers the multipliers from the solver and checks the KKT stationarity
     rows (S1)-(S8) and the unrolled costate, using the sign conventions of the note.
  3. Builds the equivalent bids (E):
         u^p*_{e,t} = alpha_e (pi*_{e,t} - nubar*_{e,t}) - eps_e
         c^g*_{e,t} = zeta_e + beta_e (pi*_{e,t} + nuund*_{e,t})
     with one block per ESS and interval, block sizes = physical limits.
  4. Solves the current design  P^CD  (A1)-(A7), (C1)-(C2) with those bids and checks
       Prop. 1: the SOC-design dispatch is CD-optimal, with the same nodal prices;
       Prop. 2: f^SOC* = f^CD* + K.
  5. Prints the data used in the chart of slide 13 (ESS 1 in the 1-bus test).

Assumptions (as in the note): convex LP, p/g exclusivity relaxed, linear terminal bid
b_e(D) = w_e * D, ramps for t >= 2.

Sign conventions.  scipy's linprog returns marginals = d f*/d rhs. The note writes the
Lagrangian as  L = f + y * (lhs - rhs)  for every constraint, with multipliers of MW rows
scaled by dT. Hence
    equality / <= rows :  y = -marginal            (divide by dT for MW rows)
    variable upper bound: ybar = -upper.marginal / dT
    variable lower bound: yund =  lower.marginal / dT
"""
import numpy as np
from scipy.optimize import linprog


# ----------------------------------------------------------------------------------
# Test cases
# ----------------------------------------------------------------------------------
def baldick_case(smax, w):
    """Single-bus example of Baldick (2026), Sec. 4.7. p_bar, g_bar not binding (1000 MW)."""
    T = 6
    return dict(
        name=f"Baldick 4.7 (s_max={smax}, w={w})", T=T, dT=4.0,
        N=1, lines=[], y=[], fbar=[], ref=0,
        gens=[dict(n=0, blocks=[(0.0, [200, 100, 500, 500, 25, 0])], Ru=1e6, Rd=1e6),   # renewable
              dict(n=0, blocks=[(50.0, [100] * T)], Ru=1e6, Rd=1e6)],                  # thermal
        loads=[],                                            # no elastic demand
        D=np.array([[100, 200, 300, 300, 200, 100.]]),       # inelastic demand
        ess=[dict(n=0, a=0.9, b=1.0, g=1.0, eps=1.0, zet=1.0, smin=0.0, smax=smax,
                  s0=0.0, pbar=1000.0, gbar=1000.0, w=w)])


def one_bus_case():
    """Single-bus (single-node) test: same generators, ramp limits, elastic demand blocks,
    and two structurally different ESS (gamma<1, s_min>0) as the old 3-bus network test,
    just collapsed onto one node -- no lines, no DC power flow, no reference-angle rows.
    Demand is the sum of the three original buses' profiles, so the aggregate system is
    unchanged; only the network (and hence (S3)/(S4)'s flow/angle content) is removed.
    """
    T = 6
    return dict(
        name="1-bus test (ramps, elastic demand, 2 ESS)", T=T, dT=4.0,
        N=1, lines=[], y=[], fbar=[], ref=0,
        gens=[dict(n=0, blocks=[(20, [80] * T), (45, [60] * T)], Ru=40, Rd=40),
              dict(n=0, blocks=[(35, [50] * T), (70, [80] * T)], Ru=30, Rd=30),
              dict(n=0, blocks=[(0, [200, 100, 500, 500, 25, 0])], Ru=1e4, Rd=1e4)],  # renewable
        loads=[dict(n=0, blocks=[(90, 40), (60, 30)]), dict(n=0, blocks=[(55, 50)])],
        D=np.array([[100, 200, 300, 300, 200, 100.]]),
        ess=[dict(n=0, a=0.9, b=1.0, g=0.99, eps=1., zet=1., smin=10., smax=500., s0=100.,
                  pbar=120., gbar=120., w=40.),                       # "ESS 1" on slide 13
             dict(n=0, a=0.85, b=1.05, g=1.0, eps=2., zet=1.5, smin=0., smax=300., s0=0.,
                  pbar=80., gbar=80., w=55.)])


# ----------------------------------------------------------------------------------
# LP builder (shared by both designs)
# ----------------------------------------------------------------------------------
class LP:
    def __init__(self):
        self.idx, self.nv = {}, 0
        self.Aeq, self.beq, self.tagE = [], [], []
        self.Aub, self.bub, self.tagU = [], [], []

    def var(self, key):
        self.idx[key] = self.nv
        self.nv += 1

    def row(self, coefs):
        r = np.zeros(self.nv)
        for key, v in coefs:
            r[self.idx[key]] += v
        return r

    def eq(self, coefs, rhs, tag):
        self.Aeq.append(self.row(coefs)); self.beq.append(rhs); self.tagE.append(tag)

    def le(self, coefs, rhs, tag):
        self.Aub.append(self.row(coefs)); self.bub.append(rhs); self.tagU.append(tag)


def build(case, design, bids=None):
    """design = 'SOC' (Baldick) or 'CD' (current design with bids)."""
    T, dT, N = case["T"], case["dT"], case["N"]
    L = len(case["lines"])
    A = np.zeros((N, L))                                  # bus-line incidence
    for l, (fr, to) in enumerate(case["lines"]):
        A[fr, l], A[to, l] = 1, -1
    lp = LP()
    for i, G in enumerate(case["gens"]):
        for k in range(len(G["blocks"])):
            for t in range(T): lp.var(("q", i, k, t))
    for m, Ld in enumerate(case["loads"]):
        for j in range(len(Ld["blocks"])):
            for t in range(T): lp.var(("d", m, j, t))
    for l in range(L):
        for t in range(T): lp.var(("f", l, t))
    for n in range(N):
        for t in range(T): lp.var(("th", n, t))
    for e in range(len(case["ess"])):
        for t in range(T):
            lp.var(("p", e, t)); lp.var(("g", e, t))
            if design == "SOC": lp.var(("s", e, t))

    c = np.zeros(lp.nv)
    bnd = [(None, None)] * lp.nv
    # objective and bounds: rest of market  (A3), (A5), (A7)
    for i, G in enumerate(case["gens"]):
        for k, (ck, qb) in enumerate(G["blocks"]):
            for t in range(T):
                c[lp.idx["q", i, k, t]] = dT * ck
                bnd[lp.idx["q", i, k, t]] = (0, qb[t])
    for m, Ld in enumerate(case["loads"]):
        for j, (u, db) in enumerate(Ld["blocks"]):
            for t in range(T):
                c[lp.idx["d", m, j, t]] = -dT * u
                bnd[lp.idx["d", m, j, t]] = (0, db)
    for l in range(L):
        for t in range(T): bnd[lp.idx["f", l, t]] = (-case["fbar"][l], case["fbar"][l])
    # ESS objective and power bounds: (A11)-(A12) or (C1)-(C2)
    for e, S in enumerate(case["ess"]):
        for t in range(T):
            bnd[lp.idx["p", e, t]] = (0, S["pbar"])
            bnd[lp.idx["g", e, t]] = (0, S["gbar"])
            if design == "SOC":
                c[lp.idx["p", e, t]] = dT * S["eps"]
                c[lp.idx["g", e, t]] = dT * S["zet"]
            else:
                c[lp.idx["p", e, t]] = -dT * bids["u"][e][t]
                c[lp.idx["g", e, t]] = dT * bids["c"][e][t]
        if design == "SOC":
            c[lp.idx["s", e, T - 1]] -= S["w"]        # -b_e(s_T - sbb) = -w s_T + w sbb (constant dropped)

    # (A1) nodal balance:  sum d + D + sum A f - sum q - sum (g - p) = 0
    for n in range(N):
        for t in range(T):
            co = []
            for m, Ld in enumerate(case["loads"]):
                if Ld["n"] == n: co += [(("d", m, j, t), 1) for j in range(len(Ld["blocks"]))]
            co += [(("f", l, t), A[n, l]) for l in range(L)]
            for i, G in enumerate(case["gens"]):
                if G["n"] == n: co += [(("q", i, k, t), -1) for k in range(len(G["blocks"]))]
            for e, S in enumerate(case["ess"]):
                if S["n"] == n: co += [(("g", e, t), -1), (("p", e, t), 1)]
            lp.eq(co, -case["D"][n, t], ("lam", n, t))
    # (A2) DC flow and (A4) reference angle
    for l in range(L):
        for t in range(T):
            lp.eq([(("f", l, t), 1)] + [(("th", n, t), -case["y"][l] * A[n, l]) for n in range(N)], 0, ("kap", l, t))
    for t in range(T):
        lp.eq([(("th", case["ref"], t), 1)], 0, ("xi", t))
    # (A6) ramps, t >= 2
    for i, G in enumerate(case["gens"]):
        K = len(G["blocks"])
        for t in range(1, T):
            up = [(("q", i, k, t), 1) for k in range(K)] + [(("q", i, k, t - 1), -1) for k in range(K)]
            lp.le(up, G["Ru"], ("tauU", i, t))
            lp.le([(key, -v) for key, v in up], G["Rd"], ("tauL", i, t))
    # (A8)-(A10) SOC rows, SOC design only
    if design == "SOC":
        for e, S in enumerate(case["ess"]):
            for t in range(T):
                prev = [(("s", e, t - 1), -S["g"])] if t > 0 else []
                rhs0 = S["g"] * S["s0"] if t == 0 else 0.0
                lp.eq([(("s", e, t), 1), (("p", e, t), -dT * S["a"]), (("g", e, t), dT * S["b"])] + prev,
                      rhs0, ("pi", e, t))                                                   # (A8)
                prevU = [(("s", e, t - 1), S["g"])] if t > 0 else []
                lp.le([(("p", e, t), dT * S["a"])] + prevU,
                      S["smax"] - (S["g"] * S["s0"] if t == 0 else 0), ("nuU", e, t))       # (A9)
                prevL = [(("s", e, t - 1), -S["g"])] if t > 0 else []
                lp.le([(("g", e, t), dT * S["b"])] + prevL,
                      -S["smin"] + (S["g"] * S["s0"] if t == 0 else 0), ("nuL", e, t))      # (A10)

    res = linprog(c, A_ub=np.array(lp.Aub) if lp.Aub else None, b_ub=lp.bub or None,
                  A_eq=np.array(lp.Aeq), b_eq=lp.beq, bounds=bnd, method="highs")
    assert res.status == 0, res.message
    return lp, res, A


def duals(case, lp, res):
    """Multipliers in the note's convention (all in $/MWh)."""
    dT = case["dT"]
    mw_rows = ("lam", "kap", "xi", "tauU", "tauL")
    y = {}
    for k, tag in enumerate(lp.tagE):
        y[tag] = -res.eqlin.marginals[k] / (dT if tag[0] in mw_rows else 1.0)
    for k, tag in enumerate(lp.tagU):
        y[tag] = -res.ineqlin.marginals[k] / (dT if tag[0] in mw_rows else 1.0)
    ub = -res.upper.marginals / dT          # upper-bound multipliers  (rho_bar, sigma_bar, delta_bar, ...)
    lb = res.lower.marginals / dT           # lower-bound multipliers  (rho_und, sigma_und, delta_und, ...)
    return y, ub, lb


# ----------------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------------
def verify(case, show_chart_data=False):
    T, dT, N = case["T"], case["dT"], case["N"]
    lp, res, A = build(case, "SOC")
    y, ub, lb = duals(case, lp, res)
    x, I = res.x, lp.idx
    L = len(case["lines"])
    get = lambda key: y.get(key, 0.0)
    resid = {}

    # (S1) generators, with ramp wedge  rho_it = (tauU_t - tauL_t) - (tauU_{t+1} - tauL_{t+1})
    r = []
    for i, G in enumerate(case["gens"]):
        for k, (ck, _) in enumerate(G["blocks"]):
            for t in range(T):
                wedge = (get(("tauU", i, t)) - get(("tauL", i, t))) - (get(("tauU", i, t + 1)) - get(("tauL", i, t + 1)))
                j = I["q", i, k, t]
                r.append(ck - y["lam", G["n"], t] + ub[j] - lb[j] + wedge)
    resid["S1"] = r
    # (S2) elastic demand
    resid["S2"] = [-u + y["lam", Ld["n"], t] + ub[I["d", m, j, t]] - lb[I["d", m, j, t]]
                   for m, Ld in enumerate(case["loads"]) for j, (u, _) in enumerate(Ld["blocks"]) for t in range(T)]
    # (S3) flows, (S4) angles
    resid["S3"] = [sum(A[n, l] * y["lam", n, t] for n in range(N)) + y["kap", l, t] + ub[I["f", l, t]] - lb[I["f", l, t]]
                   for l in range(L) for t in range(T)]
    resid["S4"] = [-sum(case["y"][l] * A[n, l] * y["kap", l, t] for l in range(L)) + (y["xi", t] if n == case["ref"] else 0)
                   for n in range(N) for t in range(T)]

    # ESS rows (S5)-(S8), unrolled costate, equivalent bids (E), constant K
    bids, K, S58, unroll, chart = {"u": [], "c": []}, 0.0, [], [], None
    for e, S in enumerate(case["ess"]):
        pi = np.array([y["pi", e, t] for t in range(T)])
        nU = np.array([y["nuU", e, t] for t in range(T)])
        nL = np.array([y["nuL", e, t] for t in range(T)])
        lam = np.array([y["lam", S["n"], t] for t in range(T)])
        rhoU = np.array([ub[I["p", e, t]] for t in range(T)]); rhoL = np.array([lb[I["p", e, t]] for t in range(T)])
        sigU = np.array([ub[I["g", e, t]] for t in range(T)]); sigL = np.array([lb[I["g", e, t]] for t in range(T)])
        S58 += list(S["eps"] + lam - S["a"] * (pi - nU) + rhoU - rhoL)                       # (S5)
        S58 += list(S["zet"] - lam + S["b"] * (pi + nL) + sigU - sigL)                       # (S6)
        S58 += [pi[t] - S["g"] * (pi[t + 1] - nU[t + 1] + nL[t + 1]) for t in range(T - 1)]   # (S7)
        S58 += [pi[T - 1] - S["w"]]                                                          # (S8), linear b
        unroll += [S["g"] ** (T - 1 - t) * pi[T - 1]
                   + sum(S["g"] ** (tau - t) * (nL[tau] - nU[tau]) for tau in range(t + 1, T)) - pi[t]
                   for t in range(T)]
        u_star = S["a"] * (pi - nU) - S["eps"]                                               # (E)
        c_star = S["zet"] + S["b"] * (pi + nL)
        bids["u"].append(u_star); bids["c"].append(c_star)
        # K_e = -pi_0 s_0 - s_max sum nuU + s_min sum nuL + min_s{pi_T s - w (s - sbb)};  last term = w*sbb
        sbb = S["g"] ** T * S["s0"]
        pi0 = S["g"] * (pi[0] - nU[0] + nL[0])
        K += -pi0 * S["s0"] - S["smax"] * nU.sum() + S["smin"] * nL.sum() + S["w"] * sbb
        if e == 0:
            chart = dict(lam=lam, u=u_star, c=c_star,
                         p=[x[I["p", e, t]] for t in range(T)], g=[x[I["g", e, t]] for t in range(T)],
                         s=[x[I["s", e, t]] for t in range(T)])
    resid["S5-S8"] = S58
    resid["unroll"] = unroll

    # Current design with bids (E)
    lp2, res2, _ = build(case, "CD", bids)
    c2 = res2.fun
    # CD objective evaluated at the SOC-design dispatch
    x_soc_in_cd = np.zeros(lp2.nv)
    for key, j in lp2.idx.items():
        x_soc_in_cd[j] = x[I[key]]
    c_cd = np.zeros(lp2.nv)  # rebuild the CD cost vector to evaluate f^CD at the SOC dispatch
    for i, G in enumerate(case["gens"]):
        for k, (ck, _) in enumerate(G["blocks"]):
            for t in range(T): c_cd[lp2.idx["q", i, k, t]] = dT * ck
    for m, Ld in enumerate(case["loads"]):
        for j, (u, _) in enumerate(Ld["blocks"]):
            for t in range(T): c_cd[lp2.idx["d", m, j, t]] = -dT * u
    for e in range(len(case["ess"])):
        for t in range(T):
            c_cd[lp2.idx["p", e, t]] = -dT * bids["u"][e][t]
            c_cd[lp2.idx["g", e, t]] = dT * bids["c"][e][t]
    f_cd_at_soc = c_cd @ x_soc_in_cd
    lam_cd = np.array([-res2.eqlin.marginals[k] / dT for k, tag in enumerate(lp2.tagE) if tag[0] == "lam"])
    lam_soc = np.array([y[tag] for tag in lp.tagE if tag[0] == "lam"])
    f_soc_true = res.fun + sum(S["w"] * S["g"] ** T * S["s0"] for S in case["ess"])   # add back constant w*sbb

    print(f"\n=== {case['name']} ===")
    for k, v in resid.items():
        if v: print(f"  max |{k:6s}| residual : {np.max(np.abs(v)):.2e}")
    print(f"  Prop. 1: f^CD* = {c2:.4f}   f^CD(SOC dispatch) = {f_cd_at_soc:.4f}   (equal => SOC dispatch is CD-optimal)")
    print(f"           max |lambda^CD - lambda^SOC| = {np.max(np.abs(lam_cd - lam_soc)):.2e}")
    print(f"  Prop. 2: f^SOC* = {f_soc_true:.4f}   f^CD* + K = {c2 + K:.4f}")
    if show_chart_data:
        print("\n  Slide 13 data: ESS 1")
        print("    t    lambda*    u^p*      c^g*      p       g       s")
        for t in range(T):
            print(f"    {t + 1}  {chart['lam'][t]:8.2f}  {chart['u'][t]:8.2f}  {chart['c'][t]:8.2f}"
                  f"  {chart['p'][t]:6.1f}  {chart['g'][t]:6.1f}  {chart['s'][t]:6.1f}")


if __name__ == "__main__":
    for smax, w in [(800, 40), (800, 55), (1200, 40)]:
        verify(baldick_case(smax, w))
    verify(one_bus_case(), show_chart_data=True)
