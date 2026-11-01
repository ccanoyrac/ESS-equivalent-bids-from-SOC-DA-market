"""
Computational burden: SOC-based design (Baldick) vs current design (interval bids).

Single-bus LP market, T intervals, G generator blocks with ramp limits (intertemporal in BOTH
designs, as in a real day-ahead market), E batteries. The current design is cleared with the
equivalent bids (E) computed from the SOC design, so both problems give the same dispatch and prices.

Reports, for each (T, E): rows, columns, nonzeros, solve time (HiGHS dual simplex and IPM, median of 3).
Sparse matrices are used throughout (scipy.sparse).
"""
import time, sys, numpy as np, scipy.sparse as sp
from scipy.optimize import linprog

rng = np.random.default_rng(1)


def instance(T, E, G=20):
    h = np.arange(T) * 24 / T
    D = 1000 + 400 * np.sin((h - 9) / 24 * 2 * np.pi) + 30 * rng.standard_normal(T)
    ren = np.clip(600 * np.sin((h - 6) / 12 * np.pi), 0, None) * (0.8 + 0.4 * rng.random(T))
    cost = np.sort(rng.uniform(10, 120, G)); cap = np.full(G, 1600 / G); ramp = cap * 0.5
    smax = rng.uniform(100, 400, E) * (2000 / max(E, 1)) / 250
    ess = dict(smax=smax, pbar=smax / 4, gbar=smax / 4, a=np.full(E, 0.9), b=np.full(E, 1.05),
               g=np.full(E, 0.999), eps=np.ones(E), zet=np.ones(E), s0=smax / 2, w=np.full(E, 45.0))
    return dict(T=T, E=E, G=G, dT=24 / T, D=D, ren=ren, cost=cost, cap=cap, ramp=ramp, ess=ess)


def build(I, design, u=None, c=None):
    T, E, G, dT = I["T"], I["E"], I["G"], I["dT"]; S = I["ess"]
    # variable layout: q[G,T], r[T] (renewable), p[E,T], g[E,T], (s[E,T] for SOC)
    nq, nr, npg = G * T, T, E * T
    iq = lambda k, t: k * T + t; ir = lambda t: nq + t
    ip = lambda e, t: nq + nr + e * T + t; ig = lambda e, t: nq + nr + npg + e * T + t
    isx = lambda e, t: nq + nr + 2 * npg + e * T + t
    nv = nq + nr + 2 * npg + (npg if design == "SOC" else 0)
    cvec = np.zeros(nv); lb = np.zeros(nv); ub = np.zeros(nv)
    for k in range(G):
        cvec[k * T:(k + 1) * T] = dT * I["cost"][k]; ub[k * T:(k + 1) * T] = I["cap"][k]
    ub[nq:nq + T] = I["ren"]
    for e in range(E):
        sl_p = slice(ip(e, 0), ip(e, 0) + T); sl_g = slice(ig(e, 0), ig(e, 0) + T)
        ub[sl_p] = S["pbar"][e]; ub[sl_g] = S["gbar"][e]
        if design == "SOC":
            cvec[sl_p] = dT * S["eps"][e]; cvec[sl_g] = dT * S["zet"][e]
            sl_s = slice(isx(e, 0), isx(e, 0) + T); lb[sl_s] = -np.inf; ub[sl_s] = np.inf
            cvec[isx(e, T - 1)] = -S["w"][e]
        else:
            cvec[sl_p] = -dT * u[e]; cvec[sl_g] = dT * c[e]
    Er, Ec, Ev, beq = [], [], [], []           # equalities
    Ur, Uc, Uv, bub = [], [], [], []           # inequalities
    def add(R, C, V, row, cols, vals): R += [row] * len(cols); C += cols; V += vals
    # (A1) balance:  sum q + r + sum (g - p) = D
    for t in range(T):
        cols = [iq(k, t) for k in range(G)] + [ir(t)] + [ig(e, t) for e in range(E)] + [ip(e, t) for e in range(E)]
        vals = [1.0] * (G + 1 + E) + [-1.0] * E
        add(Er, Ec, Ev, len(beq), cols, vals); beq.append(I["D"][t])
    # (A6) ramps
    for k in range(G):
        for t in range(1, T):
            add(Ur, Uc, Uv, len(bub), [iq(k, t), iq(k, t - 1)], [1.0, -1.0]); bub.append(I["ramp"][k])
            add(Ur, Uc, Uv, len(bub), [iq(k, t - 1), iq(k, t)], [1.0, -1.0]); bub.append(I["ramp"][k])
    if design == "SOC":
        for e in range(E):
            a, b, gm, s0 = S["a"][e], S["b"][e], S["g"][e], S["s0"][e]
            for t in range(T):
                # (A8) s_t - g s_{t-1} - dT a p_t + dT b g_t = 0
                cols = [isx(e, t), ip(e, t), ig(e, t)] + ([isx(e, t - 1)] if t else [])
                vals = [1.0, -dT * a, dT * b] + ([-gm] if t else [])
                add(Er, Ec, Ev, len(beq), cols, vals); beq.append(gm * s0 if t == 0 else 0.0)
                # (A9) g s_{t-1} + dT a p_t <= smax
                cols = [ip(e, t)] + ([isx(e, t - 1)] if t else []); vals = [dT * a] + ([gm] if t else [])
                add(Ur, Uc, Uv, len(bub), cols, vals); bub.append(S["smax"][e] - (gm * s0 if t == 0 else 0))
                # (A10) -g s_{t-1} + dT b g_t <= -smin   (smin = 0)
                cols = [ig(e, t)] + ([isx(e, t - 1)] if t else []); vals = [dT * b] + ([-gm] if t else [])
                add(Ur, Uc, Uv, len(bub), cols, vals); bub.append(gm * s0 if t == 0 else 0.0)
    Aeq = sp.csr_matrix((Ev, (Er, Ec)), shape=(len(beq), nv))
    Aub = sp.csr_matrix((Uv, (Ur, Uc)), shape=(len(bub), nv))
    return cvec, Aub, np.array(bub), Aeq, np.array(beq), list(zip(lb, ub)), (ip, ig, isx)


def solve(prob, method, reps=3):
    cvec, Aub, bub, Aeq, beq, bnd, _ = prob
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = linprog(cvec, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bnd, method=method)
        ts.append(time.perf_counter() - t0)
        assert r.status == 0, r.message
    return r, float(np.median(ts))


def size(prob):
    _, Aub, _, Aeq, _, bnd, _ = prob
    return Aub.shape[0] + Aeq.shape[0], len(bnd), Aub.nnz + Aeq.nnz


FULL_GRID = [(24, 1), (24, 10), (24, 50), (24, 200), (96, 10), (96, 50), (96, 200), (288, 50), (288, 200)]


def run(grid=FULL_GRID, reps=3, verbose=True):
    """Solve both designs for every (T, E) in grid. Returns a list of dicts."""
    if verbose:
        print(f"{'T':>4} {'E':>4} | {'rows SOC/CD':>14} {'cols SOC/CD':>14} {'nnz SOC/CD':>15} | "
              f"{'simplex s SOC/CD':>18} {'IPM s SOC/CD':>16} | {'max|dlam|':>9}")
    out = []
    for T, E in grid:
        I = instance(T, E)
        P_soc = build(I, "SOC")
        r_soc, ts_soc = solve(P_soc, "highs-ds", reps)
        _, ti_soc = solve(P_soc, "highs-ipm", reps)
        # equivalent bids (E) from the SOC duals
        dT = I["dT"]; S = I["ess"]
        lam = r_soc.eqlin.marginals[:T] / dT          # balance written as supply - demand = D
        pi = -r_soc.eqlin.marginals[T:].reshape(E, T)
        nmu = -r_soc.ineqlin.marginals[2 * I["G"] * (T - 1):].reshape(E, T, 2)
        nU, nL = nmu[:, :, 0], nmu[:, :, 1]
        u = S["a"][:, None] * (pi - nU) - S["eps"][:, None]; c = S["zet"][:, None] + S["b"][:, None] * (pi + nL)
        P_cd = build(I, "CD", u, c)
        r_cd, ts_cd = solve(P_cd, "highs-ds", reps)
        _, ti_cd = solve(P_cd, "highs-ipm", reps)
        dlam = np.abs(r_cd.eqlin.marginals[:T] / dT - lam).max()
        (ms, ns, zs), (mc, nc, zc) = size(P_soc), size(P_cd)
        out.append(dict(T=T, E=E, rows_soc=ms, rows_cd=mc, cols_soc=ns, cols_cd=nc, nnz_soc=zs, nnz_cd=zc,
                        simplex_soc=ts_soc, simplex_cd=ts_cd, ipm_soc=ti_soc, ipm_cd=ti_cd, dlam=dlam))
        if verbose:
            print(f"{T:>4} {E:>4} | {ms:>6}/{mc:<7} {ns:>6}/{nc:<7} {zs:>7}/{zc:<7} | "
                  f"{ts_soc:8.3f}/{ts_cd:<8.3f} {ti_soc:7.3f}/{ti_cd:<7.3f} | {dlam:9.1e}")
            sys.stdout.flush()
    return out


if __name__ == "__main__":
    import json
    res = run()
    json.dump(res, open("computational_burden_results.json", "w"), indent=1)
