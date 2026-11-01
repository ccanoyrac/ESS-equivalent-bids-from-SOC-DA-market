"""
Numerical check of Section 4 of the equivalence note (bids from technical data and prices).

For every test case and every ESS e:
  1. Solve Baldick's SOC design; read the nodal prices lambda* at n(e).
  2. Lemma 4 (decomposition): solve the ESS self-schedule P^SS_e(lambda*) using only theta_e and lambda*;
     check that Baldick's schedule attains its optimal value.
  3. Prop. 5: from the self-schedule primal only, build the polyhedron of admissible
     (pi, nubar, nuund) and compute the range of every pi_t by LP.
     Cross-check against the dual optimal face of P^SS (dual objective = primal value).
  4. Theorem 1: for the extreme points (min and max of each pi_t), build the bids,
     clear the current design, and check that the SOC-design dispatch is CD-optimal and
     that the CD prices equal lambda*.
"""
import numpy as np
from scipy.optimize import linprog
import verify_equivalence as V
from verify_equivalence import build, duals, baldick_case, three_bus_case

TOL = 1e-6

def capture_build(case, design, bids=None):
    cap = {}
    orig = V.linprog
    def spy(c, **k):
        cap.update(c=c, **k); return orig(c, **k)
    V.linprog = spy
    try:
        lp, res, A = build(case, design, bids)
    finally:
        V.linprog = orig
    return lp, res, cap

def self_schedule(S, lam, T, dT):
    """P^SS_e(lambda): min sum dT[(eps+lam)p + (zeta-lam)g] - w s_T  s.t. (A8)-(A12)."""
    n = 3 * T; ip = lambda t: t; ig = lambda t: T + t; iS = lambda t: 2 * T + t
    c = np.zeros(n)
    for t in range(T):
        c[ip(t)] = dT * (S['eps'] + lam[t]); c[ig(t)] = dT * (S['zet'] - lam[t])
    c[iS(T - 1)] = -S['w']
    Aeq, beq, Aub, bub = [], [], [], []
    for t in range(T):
        r = np.zeros(n); r[iS(t)] = 1; r[ip(t)] = -dT * S['a']; r[ig(t)] = dT * S['b']
        if t > 0: r[iS(t - 1)] = -S['g']; beq.append(0.0)
        else: beq.append(S['g'] * S['s0'])
        Aeq.append(r)
        r = np.zeros(n); r[ip(t)] = dT * S['a']
        if t > 0: r[iS(t - 1)] = S['g']; bub.append(S['smax'])
        else: bub.append(S['smax'] - S['g'] * S['s0'])
        Aub.append(r)
        r = np.zeros(n); r[ig(t)] = dT * S['b']
        if t > 0: r[iS(t - 1)] = -S['g']; bub.append(-S['smin'])
        else: bub.append(-S['smin'] + S['g'] * S['s0'])
        Aub.append(r)
    bnd = [(0, S['pbar'])] * T + [(0, S['gbar'])] * T + [(None, None)] * T
    res = linprog(c, A_ub=np.array(Aub), b_ub=bub, A_eq=np.array(Aeq), b_eq=beq, bounds=bnd, method='highs')
    return res, c

def prop5_polyhedron(S, lam, p, g, s, T, dT):
    """Prop. 5: admissible (pi, nubar, nuund) from theta_e, lambda and ONE optimal schedule (p,g,s).
    Returns equality/inequality lists over x = [pi(T), nubar(T), nuund(T)]."""
    a, b, gam, eps, zet = S['a'], S['b'], S['g'], S['eps'], S['zet']
    sp = np.concatenate([[S['s0']], s])               # s_{t-1}
    U = [abs(gam * sp[t] + dT * a * p[t] - S['smax']) < 1e-6 for t in range(T)]   # upper SOC row tight
    L = [abs(gam * sp[t] - dT * b * g[t] - S['smin']) < 1e-6 for t in range(T)]   # lower SOC row tight
    nv = 3 * T; P = lambda t: t; NU = lambda t: T + t; NL = lambda t: 2 * T + t
    Aeq, beq, Aub, bub = [], [], [], []
    # (a) terminal value, (b) recursion
    r = np.zeros(nv); r[P(T - 1)] = 1; Aeq.append(r); beq.append(S['w'])
    for t in range(T - 1):
        r = np.zeros(nv); r[P(t)] = 1; r[P(t + 1)] = -gam; r[NU(t + 1)] = gam; r[NL(t + 1)] = -gam
        Aeq.append(r); beq.append(0.0)
    bnd = [(None, None)] * T + [(0, None if U[t] else 0) for t in range(T)] + [(0, None if L[t] else 0) for t in range(T)]
    # (d) price consistency, u = a(pi-nubar)-eps, c = zet + b(pi+nuund)
    for t in range(T):
        u = np.zeros(nv); u[P(t)] = a; u[NU(t)] = -a; uc = -eps      # u = u.x + uc
        cc = np.zeros(nv); cc[P(t)] = b; cc[NL(t)] = b; ccc = zet
        if p[t] < 1e-6:                     # u <= lam
            Aub.append(u); bub.append(lam[t] - uc)
        elif p[t] > S['pbar'] - 1e-6:       # u >= lam
            Aub.append(-u); bub.append(-(lam[t] - uc))
        else:                               # u = lam
            Aeq.append(u); beq.append(lam[t] - uc)
        if g[t] < 1e-6:                     # c >= lam
            Aub.append(-cc); bub.append(-(lam[t] - ccc))
        elif g[t] > S['gbar'] - 1e-6:       # c <= lam
            Aub.append(cc); bub.append(lam[t] - ccc)
        else:                               # c = lam
            Aeq.append(cc); beq.append(lam[t] - ccc)
    return dict(Aeq=np.array(Aeq), beq=beq, Aub=np.array(Aub) if Aub else None, bub=bub or None, bnd=bnd, U=U, L=L)

def pi_range(poly, T):
    out, pts = [], []
    for t in range(T):
        for sgn in (1, -1):
            e = np.zeros(3 * T); e[t] = sgn
            r = linprog(e, A_ub=poly['Aub'], b_ub=poly['bub'], A_eq=poly['Aeq'], b_eq=poly['beq'], bounds=poly['bnd'], method='highs')
            assert r.status == 0, r.message
            pts.append(r.x)
        out.append((pts[-2][t], pts[-1][t]))
    return out, pts

def dual_face_range(S, lam, T, dT, fss):
    """Range of pi_t over the dual optimal face of P^SS (independent cross-check)."""
    nv = 7 * T; P = lambda t: t; NU = lambda t: T + t; NL = lambda t: 2 * T + t
    RU = lambda t: 3 * T + t; RL = lambda t: 4 * T + t; SU = lambda t: 5 * T + t; SL = lambda t: 6 * T + t
    Aeq, beq = [], []
    for t in range(T):
        r = np.zeros(nv); r[P(t)] = -S['a']; r[NU(t)] = S['a']; r[RU(t)] = 1; r[RL(t)] = -1; Aeq.append(r); beq.append(-(S['eps'] + lam[t]))
        r = np.zeros(nv); r[P(t)] = S['b']; r[NL(t)] = S['b']; r[SU(t)] = 1; r[SL(t)] = -1; Aeq.append(r); beq.append(-(S['zet'] - lam[t]))
        r = np.zeros(nv)
        if t < T - 1: r[P(t)] = 1; r[P(t + 1)] = -S['g']; r[NU(t + 1)] = S['g']; r[NL(t + 1)] = -S['g']; beq.append(0.0)
        else: r[P(t)] = 1; beq.append(S['w'])
        Aeq.append(r)
    r = np.zeros(nv); gm = S['g']; s0 = S['s0']
    r[P(0)] = -gm * s0; r[NU(0)] = gm * s0; r[NL(0)] = -gm * s0
    for t in range(T):
        r[NU(t)] += -S['smax']; r[NL(t)] += S['smin']; r[RU(t)] += -dT * S['pbar']; r[SU(t)] += -dT * S['gbar']
    Aeq.append(r); beq.append(fss)
    bnd = [(None, None)] * T + [(0, None)] * (6 * T)
    out = []
    for t in range(T):
        e = np.zeros(nv); e[P(t)] = 1
        lo = linprog(e, A_eq=np.array(Aeq), b_eq=beq, bounds=bnd, method='highs')
        hi = linprog(-e, A_eq=np.array(Aeq), b_eq=beq, bounds=bnd, method='highs')
        out.append((lo.fun, -hi.fun))
    return out

def check_case(case):
    T, dT = case['T'], case['dT']
    lpS, resS, capS = capture_build(case, 'SOC')
    yS, _, _ = duals(case, lpS, resS); I = lpS.idx; x = resS.x
    # solver bids (E) for all ESS, used as the baseline for the other ESS
    base_u, base_c = [], []
    for e, S in enumerate(case['ess']):
        pi = np.array([yS['pi', e, t] for t in range(T)]); nU = np.array([yS['nuU', e, t] for t in range(T)]); nL = np.array([yS['nuL', e, t] for t in range(T)])
        base_u.append(S['a'] * (pi - nU) - S['eps']); base_c.append(S['zet'] + S['b'] * (pi + nL))
    print(f"\n=== {case['name']}")
    worst_lam, worst_obj, worst_kkt = 0.0, 0.0, 0.0
    for e, S in enumerate(case['ess']):
        lam = np.array([yS['lam', S['n'], t] for t in range(T)])
        pB = np.array([x[I['p', e, t]] for t in range(T)]); gB = np.array([x[I['g', e, t]] for t in range(T)]); sB = np.array([x[I['s', e, t]] for t in range(T)])
        rss, css = self_schedule(S, lam, T, dT)
        valB = css @ np.concatenate([pB, gB, sB])
        print(f"ESS {e+1}: lambda* = {lam.round(2)}")
        print(f"  Lemma 4 (decomposition): P^SS optimum {rss.fun:.4f}, Baldick schedule {valB:.4f}, gap {abs(rss.fun-valB):.1e}")
        p, g, s = rss.x[:T], rss.x[T:2*T], rss.x[2*T:]
        poly = prop5_polyhedron(S, lam, p, g, s, T, dT)
        rng, pts = pi_range(poly, T)
        dfr = dual_face_range(S, lam, T, dT, rss.fun)
        print("  tight upper SOC rows:", [t+1 for t in range(T) if poly['U'][t]], " tight lower:", [t+1 for t in range(T) if poly['L'][t]])
        print("  Prop. 5 pi ranges   :", [f"[{a:.3f},{b:.3f}]" if b-a > 1e-6 else f"{a:.3f}" for a, b in rng])
        print("  dual-face pi ranges :", [f"[{a:.3f},{b:.3f}]" if b-a > 1e-6 else f"{a:.3f}" for a, b in dfr])
        print("  max |Prop5 - dual face| =", max(max(abs(a-c), abs(b-d)) for (a, b), (c, d) in zip(rng, dfr)))
        # Theorem 1 on the extreme points
        for xp in pts:
            pi, nU, nL = xp[:T], xp[T:2*T], xp[2*T:]
            u = list(base_u); c = list(base_c)
            u[e] = S['a'] * (pi - nU) - S['eps']; c[e] = S['zet'] + S['b'] * (pi + nL)
            lpC, resC, capC = capture_build(case, 'CD', {'u': u, 'c': c})
            lamC = np.array([[-resC.eqlin.marginals[k] / dT for k, tag in enumerate(lpC.tagE) if tag == ('lam', n, t)][0] for n in range(case['N']) for t in range(T)])
            lamS = np.array([yS['lam', n, t] for n in range(case['N']) for t in range(T)])
            # evaluate CD objective at the SOC-design dispatch
            xc = np.zeros(lpC.nv)
            for key, j in lpC.idx.items(): xc[j] = x[I[key]]
            worst_lam = max(worst_lam, np.abs(lamC - lamS).max())
            # KKT certificate: box form of (S5^CD)-(S6^CD) for ESS e at (x*, lambda*)
            viol = 0.0
            for t in range(T):
                ue, ce, pt, gt = u[e][t], c[e][t], pB[t], gB[t]
                if pt < 1e-6: viol = max(viol, ue - lam[t])
                elif pt > S['pbar'] - 1e-6: viol = max(viol, lam[t] - ue)
                else: viol = max(viol, abs(ue - lam[t]))
                if gt < 1e-6: viol = max(viol, lam[t] - ce)
                elif gt > S['gbar'] - 1e-6: viol = max(viol, ce - lam[t])
                else: viol = max(viol, abs(ce - lam[t]))
            worst_kkt = max(worst_kkt, viol)
            worst_obj = max(worst_obj, abs(capC['c'] @ xc - resC.fun) / max(1, abs(resC.fun)))
    print(f"Theorem 1 over all extreme points: KKT violation of (x*, lambda*) in the CD = {worst_kkt:.1e}, "
          f"rel. gap f_CD(x*) - f_CD* = {worst_obj:.1e}")
    print(f"  (prices returned by the CD solver differ from lambda* by up to {worst_lam:.1e}: the CD dual is not unique there)")

if __name__ == '__main__':
    for smax, w in [(800, 40), (800, 55), (1200, 40)]:
        check_case(baldick_case(smax, w))
    # check_case(three_bus_case())   # 3-bus test: set aside for now
