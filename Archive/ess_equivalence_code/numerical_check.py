"""
Numerical check of the equivalence note on Baldick's single-bus cases [1, Sec. 4.7].

For each case (s_max, w) in {(800, 40), (800, 55), (1200, 40)}:
  - solves the SOC design P^SOC, reads lambda*, the schedule and the multipliers;
  - builds the equivalent bids (E) and clears the current design P^CD with them;
  - checks Props 1-2, the KKT residuals, Prop. 4 (pi from prices), Lemma 4,
    Prop. 5 (vs the dual optimal face) and Theorem 1 (all extreme points);
  - writes figures/fig_check_outcome.(pdf|png), figures/fig_check_pi.(pdf|png) and the
    LaTeX rows of the summary table (figures/check_table.tex).
Uses verify_equivalence.py and verify_price_based.py (same folder).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_equivalence as V
import verify_price_based as P
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
CASES = [(800, 40), (800, 55), (1200, 40)]
TITLES = [r"Base case ($\bar s=800$, $w=40$)", r"Higher terminal value ($w=55$)", r"Larger battery ($\bar s=1200$)"]
BLUE, ORANGE, BLUE_L, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#cde2fb", "#1f2328", "#5b6068", "#e3e5e8"
TOL = 1e-7


def price_based_pi(S, lam, p, g, nU, nL, T):
    """Proposition 4: pi from prices at marginal intervals, segments from the binding set."""
    H = [t for t in range(T) if nU[t] > TOL or nL[t] > TOL]
    starts = [0] + [h for h in H if h > 0]
    pi = np.full(T, np.nan)
    for k, a in enumerate(starts):
        b = starts[k + 1] if k + 1 < len(starts) else T
        seg = range(a, b)
        if b == T:                                            # last segment: terminal value
            for t in seg: pi[t] = S["g"] ** (T - 1 - t) * S["w"]
            continue
        for tt in seg:                                        # find a marginal interval
            if 1e-6 < g[tt] < S["gbar"] - 1e-6 and nL[tt] <= TOL:
                v = (lam[tt] - S["zet"]) / S["b"]; break
            if 1e-6 < p[tt] < S["pbar"] - 1e-6 and nU[tt] <= TOL:
                v = (lam[tt] + S["eps"]) / S["a"]; break
        else:
            continue                                          # no marginal interval: leave NaN
        for t in seg: pi[t] = S["g"] ** (tt - t) * v
    return pi


def run_case(smax, w):
    case = V.baldick_case(smax, w); T, dT = case["T"], case["dT"]; S = case["ess"][0]
    lp, res, _ = V.build(case, "SOC"); y, ub, lb = V.duals(case, lp, res); I, x = lp.idx, res.x
    get = lambda key: y.get(key, 0.0)
    lam = np.array([y["lam", 0, t] for t in range(T)])
    pi = np.array([y["pi", 0, t] for t in range(T)])
    nU = np.array([y["nuU", 0, t] for t in range(T)]); nL = np.array([y["nuL", 0, t] for t in range(T)])
    p = np.array([x[I["p", 0, t]] for t in range(T)]); g = np.array([x[I["g", 0, t]] for t in range(T)])
    s = np.array([x[I["s", 0, t]] for t in range(T)])
    rU = np.array([ub[I["p", 0, t]] for t in range(T)]); rL = np.array([lb[I["p", 0, t]] for t in range(T)])
    sU = np.array([ub[I["g", 0, t]] for t in range(T)]); sL = np.array([lb[I["g", 0, t]] for t in range(T)])

    # KKT residuals: (S1) generators, (S5)-(S8) ESS
    r = []
    for i, G in enumerate(case["gens"]):
        for k, (ck, _) in enumerate(G["blocks"]):
            for t in range(T):
                wedge = (get(("tauU", i, t)) - get(("tauL", i, t))) - (get(("tauU", i, t + 1)) - get(("tauL", i, t + 1)))
                j = I["q", i, k, t]; r.append(ck - lam[t] + ub[j] - lb[j] + wedge)
    r += list(S["eps"] + lam - S["a"] * (pi - nU) + rU - rL)
    r += list(S["zet"] - lam + S["b"] * (pi + nL) + sU - sL)
    r += [pi[t] - S["g"] * (pi[t + 1] - nU[t + 1] + nL[t + 1]) for t in range(T - 1)] + [pi[-1] - S["w"]]
    kkt = np.max(np.abs(r))

    # Equivalent bids (E) and the current design
    u = S["a"] * (pi - nU) - S["eps"]; c = S["zet"] + S["b"] * (pi + nL)
    lpC, resC, capC = P.capture_build(case, "CD", {"u": [u], "c": [c]})
    lamC = np.array([-resC.eqlin.marginals[k] / dT for k, tag in enumerate(lpC.tagE) if tag[0] == "lam"])
    xc = np.zeros(lpC.nv)
    for key, j in lpC.idx.items(): xc[j] = x[I[key]]
    obj_gap = abs(capC["c"] @ xc - resC.fun) / max(1.0, abs(resC.fun))
    sbb = S["g"] ** T * S["s0"]; pi0 = S["g"] * (pi[0] - nU[0] + nL[0])
    K = -pi0 * S["s0"] - S["smax"] * nU.sum() + S["smin"] * nL.sum() + S["w"] * sbb
    f_soc = res.fun + S["w"] * sbb
    p2_gap = abs(f_soc - (resC.fun + K)) / max(1.0, abs(f_soc))

    # Prop. 4: pi from prices
    pi_p4 = price_based_pi(S, lam, p, g, nU, nL, T)
    p4_err = np.nanmax(np.abs(pi_p4 - pi))

    # Lemma 4, Prop. 5, dual face, Theorem 1 (extreme points)
    rss, css = P.self_schedule(S, lam, T, dT)
    l4_gap = abs(rss.fun - css @ np.concatenate([p, g, s]))
    poly = P.prop5_polyhedron(S, lam, rss.x[:T], rss.x[T:2 * T], rss.x[2 * T:], T, dT)
    rng, pts = P.pi_range(poly, T)
    dfr = P.dual_face_range(S, lam, T, dT, rss.fun)
    p5_err = max(max(abs(a - c_), abs(b - d)) for (a, b), (c_, d) in zip(rng, dfr))
    t1_kkt, t1_lam = 0.0, 0.0
    for xp in pts:
        pe, nUe, nLe = xp[:T], xp[T:2 * T], xp[2 * T:]
        ue = S["a"] * (pe - nUe) - S["eps"]; ce = S["zet"] + S["b"] * (pe + nLe)
        for t in range(T):
            if p[t] < 1e-6: t1_kkt = max(t1_kkt, ue[t] - lam[t])
            elif p[t] > S["pbar"] - 1e-6: t1_kkt = max(t1_kkt, lam[t] - ue[t])
            else: t1_kkt = max(t1_kkt, abs(ue[t] - lam[t]))
            if g[t] < 1e-6: t1_kkt = max(t1_kkt, lam[t] - ce[t])
            elif g[t] > S["gbar"] - 1e-6: t1_kkt = max(t1_kkt, ce[t] - lam[t])
            else: t1_kkt = max(t1_kkt, abs(ce[t] - lam[t]))
        lpE, resE, _ = P.capture_build(case, "CD", {"u": [ue], "c": [ce]})
        lamE = np.array([-resE.eqlin.marginals[k] / dT for k, tag in enumerate(lpE.tagE) if tag[0] == "lam"])
        t1_lam = max(t1_lam, np.abs(lamE - lam).max())

    # pre-limit rule (projection onto the admissible range)
    piR = np.zeros(T)
    for t in range(T):
        lo, hi = rng[t]
        cand = piR[t - 1] / S["g"] if t > 0 else hi
        piR[t] = min(max(cand, lo), hi)

    return dict(case=case, S=S, T=T, lam=lam, lamC=lamC, p=p, g=g, s=s, u=u, c=c, pi=pi, pi_p4=pi_p4,
                rng=rng, piR=piR,
                checks=dict(kkt=kkt, lam=np.abs(lamC - lam).max(), obj=obj_gap, p2=p2_gap, p4=p4_err,
                            l4=l4_gap, p5=p5_err, t1=t1_kkt, t1lam=t1_lam))


def style(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#9aa0a6")
    ax.tick_params(colors=MUTED, labelsize=7.5, length=2.5)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xticks(range(1, 7))


def fig_outcome(R):
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, axs = plt.subplots(3, 3, figsize=(7.1, 5.6), sharex=True)
    t = np.arange(1, 7)
    for j, r in enumerate(R):
        # row 1: prices
        ax = axs[0, j]; style(ax)
        ax.bar(t, r["lam"], width=0.55, color=BLUE, label=r"SOC design $\lambda^\star$", zorder=2)
        ax.plot(t, r["lamC"], ls="none", marker="D", ms=5, mfc=ORANGE, mec="white", mew=0.8,
                label=r"current design with bids (E)", zorder=3)
        ax.set_ylim(0, 62); ax.set_title(TITLES[j], fontsize=8.5, color=INK)
        # row 2: bid/offer band and lambda*
        ax = axs[1, j]; style(ax)
        for tt in range(6):
            ax.add_patch(plt.Rectangle((t[tt] - 0.22, r["u"][tt]), 0.44, r["c"][tt] - r["u"][tt],
                                       fc=BLUE_L, ec=BLUE, lw=0.8, zorder=2))
        ax.plot(t, r["lam"], ls="none", marker="o", ms=5, mfc=INK, mec="white", mew=0.8, zorder=3)
        ax.set_ylim(-3, 62)
        # row 3: SOC
        ax = axs[2, j]; style(ax)
        smax = r["S"]["smax"]
        ax.axhline(smax, color=MUTED, ls=(0, (3, 2)), lw=0.8)
        ax.text(0.75, smax * 1.03, r"$\bar s$", color=MUTED, fontsize=7.5, va="bottom")
        ax.plot(np.r_[0, t], np.r_[r["S"]["s0"], r["s"]], color=BLUE, lw=1.6, marker="o", ms=3.5, zorder=3)
        ax.set_ylim(-60, 1300); ax.set_xlabel("hour $t$", fontsize=8, color=MUTED)
        ax.set_xticks(range(0, 7))
    axs[0, 0].set_ylabel("nodal price\n(\\$/MWh)", fontsize=8, color=MUTED)
    axs[1, 0].set_ylabel("bid $u_t$ / offer $c_t$\n(\\$/MWh)", fontsize=8, color=MUTED)
    axs[2, 0].set_ylabel("SOC at end of $t$\n(MWh)", fontsize=8, color=MUTED)
    h1, l1 = axs[0, 0].get_legend_handles_labels()
    band = plt.Rectangle((0, 0), 1, 1, fc=BLUE_L, ec=BLUE, lw=0.8)
    dot = plt.Line2D([], [], ls="none", marker="o", ms=5, mfc=INK, mec="white")
    fig.legend([h1[1], h1[0], band, dot], [l1[1], l1[0], r"band $[u_t, c_t]$ of bids (E)", r"$\lambda^\star_t$ (on an edge of the band)"],
               loc="lower center", ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=0.6, w_pad=0.8)
    fig.savefig(os.path.join(OUT, "fig_check_outcome.pdf")); fig.savefig(os.path.join(OUT, "fig_check_outcome.png"), dpi=200)
    plt.close(fig)


def fig_pi(R):
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.6), sharey=True)
    t = np.arange(1, 7)
    for j, r in enumerate(R):
        ax = axs[j]; style(ax)
        for tt, (lo, hi) in enumerate(r["rng"]):
            if hi - lo > 1e-6:
                ax.add_patch(plt.Rectangle((t[tt] - 0.17, lo), 0.34, hi - lo, fc=BLUE_L, ec=BLUE, lw=0.8, zorder=1))
        ax.plot(t, r["pi"], ls="none", marker="o", ms=6, mfc=BLUE, mec="white", mew=0.8, zorder=3)
        ax.plot(t, r["pi_p4"], ls="none", marker="x", ms=4.5, mew=1.1, color=INK, zorder=4)
        ax.plot(t, r["piR"], ls="none", marker="o", ms=10, mfc="none", mec=ORANGE, mew=1.4, zorder=5)
        ax.set_ylim(-3, 60); ax.set_title(TITLES[j], fontsize=8.5, color=INK); ax.set_xlabel("hour $t$", fontsize=8, color=MUTED)
    axs[0].set_ylabel(r"$\pi_t$ (\$/MWh)", fontsize=8, color=MUTED)
    hs = [plt.Rectangle((0, 0), 1, 1, fc=BLUE_L, ec=BLUE, lw=0.8),
          plt.Line2D([], [], ls="none", marker="o", ms=6, mfc=BLUE, mec="white"),
          plt.Line2D([], [], ls="none", marker="x", ms=4.5, mew=1.1, color=INK),
          plt.Line2D([], [], ls="none", marker="o", ms=9, mfc="none", mec=ORANGE, mew=1.4)]
    fig.legend(hs, ["admissible range (Prop. 5 = dual optimal face)", "solver multiplier", "from prices (Prop. 4)",
                    "pre-limit rule"], loc="lower center", ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.2, 1, 1), w_pad=0.8)
    fig.savefig(os.path.join(OUT, "fig_check_pi.pdf")); fig.savefig(os.path.join(OUT, "fig_check_pi.png"), dpi=200)
    plt.close(fig)


def sci(v):
    if v < 1e-15: return r"$0$"
    m, e = f"{v:.0e}".split("e"); return rf"${m}\cdot10^{{{int(e)}}}$"


if __name__ == "__main__":
    R = [run_case(*c) for c in CASES]
    for r in R:
        print(r["case"]["name"], {k: f"{v:.1e}" for k, v in r["checks"].items()})
        print("   ranges", [(round(a, 2), round(b, 2)) for a, b in r["rng"]], " piR", r["piR"].round(2))
    fig_outcome(R); fig_pi(R)
    rows = [("kkt", r"KKT residual of $\mathcal P^{\mathrm{SOC}}$, (S1)--(S8)", "solver"),
            ("lam", r"$\max_t|\lambda^{\mathrm{CD}}_t-\lambda^\star_t|$ with bids \eqref{E}", r"Prop.~\ref{p1}"),
            ("obj", r"$f^{\mathrm{CD}}(x^\star)-f^{\mathrm{CD}\star}$ (relative)", r"Prop.~\ref{p1}"),
            ("p2", r"$f^{\mathrm{SOC}\star}-(f^{\mathrm{CD}\star}+K)$ (relative)", r"Prop.~\ref{p2}"),
            ("p4", r"$\max_t|\pi^{\text{prices}}_t-\pi^\star_t|$", r"Prop.~\ref{p4}"),
            ("l4", r"self-schedule optimum minus value of Baldick's schedule", r"Lemma~\ref{l1}"),
            ("p5", r"range of $\pi_t$: polyhedron vs.\ dual optimal face", r"Prop.~\ref{p5}"),
            ("t1", r"CD KKT violation at $(x^\star,\lambda^\star)$, all extreme bids", r"Thm.~\ref{th1}")]
    lines = []
    for key, desc, res_ in rows:
        vals = " & ".join(sci(r["checks"][key]) for r in R)
        lines.append(f"{desc} & {res_} & {vals} & $\\checkmark$\\\\")
    open(os.path.join(OUT, "check_table.tex"), "w").write("\n".join(lines) + "\n")
    print("wrote figures and check_table.tex")
