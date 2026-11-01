"""Exploratory prototype for docs/regimes_design.tex -- NOT part of the experiments package.

Groups B_fcst draws by their day-ahead battery schedule ("regime"), with common random numbers (the same
seeds, i.e. the same error shapes z, for every sigma), and writes the illustration figures and numbers used
in the design document. The real implementation is specified in that document (experiments/regimes.py).

Usage (from Code/): python docs/prototypes/regimes_prototype.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import benchmark, bids, cases, forecasts, metrics, runner
from experiments.config import BidRule, ErrorModel, RealTime

OUT = ROOT / "docs" / "figures"
N_DRAWS = 400
SIGMAS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)  # $/MWh
SIGMA_TABLE = 5.0
RULE, PSI = "prelimit", 10.0
REGIME_COLOURS = ["#2a78d6", "#eb6834", "#5aa469", "#7b4bb7", "#c0392b", "#d4a017", "#1f8a8a"]
OTHER = "#b8bcc2"
INK, MUTED, GRID = "#1f2328", "#5b6068", "#e3e5e8"

case = cases.day_case()
bench = benchmark.solve_benchmark(case)
S, T, dT = case["ess"][0], case["T"], case["dT"]
lam_star = bench.lam_star[0]
free = np.where(np.isclose(lam_star, 0.0, atol=1e-9))[0]


def outcome_of(lam_hat):
    blk = bids.quantity_blocks(bids.bid_fcst(S, lam_hat, T, dT, RULE, BidRule().tie_break_reward), S, BidRule().delta, full_capacity=False)
    return runner.evaluate_blocks(case, [blk], RealTime(psi=PSI))


def signature(out):
    """Regime key: the day-ahead battery schedule (charge and discharge per interval, MW)."""
    return tuple(np.round(np.r_[out.da.p_da[0], out.da.g_da[0]], 3))


def pattern(sig):
    p, g = np.array(sig[:T]), np.array(sig[T:])
    return "".join("C" if p[t] > 1e-6 else ("D" if g[t] > 1e-6 else "-") for t in range(T))


def features(out):
    s_da = np.cumsum(dT * (S["a"] * out.da.p_da[0] - S["b"] * out.da.g_da[0])) + S["s0"]
    return {"bought_free_MWh": float(dT * out.da.p_da[0][free].sum()),
            "soc_violations": int(np.sum((s_da < S["smin"] - 1e-6) | (s_da > S["smax"] + 1e-6))),
            "undeliverable_MWh": float(out.undeliverable_mwh),
            "shed_MWh": float(dT * out.shed_rt.sum())}


def run_grid():
    rows = []
    for sigma in SIGMAS:
        err = ErrorModel(sigma_abs=sigma)
        for n in range(N_DRAWS):          # seed n = draw n for every sigma: common random numbers
            lh = forecasts.forecast_prices(lam_star, err, 1, T, np.random.default_rng(n))[0]
            out = outcome_of(lh)
            d = metrics.dsw(bench.SW_star, out.welfare.SW_real)
            rows.append({"sigma": sigma, "draw": n, "sig": signature(out),
                         "loss": metrics.dsw_pct(d, bench.V), **features(out)})
    return rows


MAP_BASE_DRAW = 2   # a typical draw (in the most frequent regime at sigma = 5); its z fixes the other intervals


def z_of(n):
    """The error shape z of draw n (the same random numbers forecast_prices uses, unscaled and unclipped)."""
    return forecasts.forecast_prices(np.zeros(T), ErrorModel(sigma_abs=1.0, price_floor=None), 1, T,
                                     np.random.default_rng(n))[0]


def regime_map(sigma, k1, k2, n=41, lim=3.0):
    """Regime key over a 2-D slice of z-space: z[k1], z[k2] vary, the other z_t are those of MAP_BASE_DRAW."""
    grid = np.linspace(-lim, lim, n)
    keys = np.empty((n, n), dtype=object)
    z0 = z_of(MAP_BASE_DRAW)
    for i, a in enumerate(grid):
        for j, b in enumerate(grid):
            z = z0.copy()
            z[k1], z[k2] = a, b
            keys[j, i] = signature(outcome_of(np.maximum(lam_star + sigma * z, 0.0)))
    return grid, keys


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    rows = run_grid()

    # regime ids, ordered by overall frequency
    counts = {}
    for r in rows:
        counts[r["sig"]] = counts.get(r["sig"], 0) + 1
    order = sorted(counts, key=lambda s: -counts[s])
    rid = {s: k + 1 for k, s in enumerate(order)}

    # is the loss constant within a regime? (it should be: everything after the day-ahead schedule is deterministic)
    spread = max(np.ptp([r["loss"] for r in rows if r["sig"] == s]) for s in order)

    summary = {"n_draws": N_DRAWS, "sigmas": SIGMAS, "n_regimes_total": len(order), "max_within_regime_loss_spread": spread}
    per_sigma = {}
    for sigma in SIGMAS:
        rs = [r for r in rows if r["sigma"] == sigma]
        mc_mean = float(np.mean([r["loss"] for r in rs]))
        tab = {}
        for r in rs:
            k = rid[r["sig"]]
            t = tab.setdefault(k, {"count": 0, "loss": r["loss"], "pattern": pattern(r["sig"]),
                                   **{f: r[f] for f in ("bought_free_MWh", "soc_violations", "undeliverable_MWh", "shed_MWh")}})
            t["count"] += 1
        for k, t in tab.items():
            t["p"] = t["count"] / N_DRAWS
            t["se"] = float(np.sqrt(t["p"] * (1 - t["p"]) / N_DRAWS))
            t["contribution"] = t["p"] * t["loss"]
        decomposed = sum(t["contribution"] for t in tab.values())
        per_sigma[str(sigma)] = {"mc_mean": mc_mean, "decomposed_mean": decomposed, "n_regimes": len(tab),
                                 "regimes": {str(k): tab[k] for k in sorted(tab)}}
    summary["per_sigma"] = per_sigma

    # transitions between sigma = 1 and sigma = 5 (same draws)
    a = {r["draw"]: rid[r["sig"]] for r in rows if r["sigma"] == 1.0}
    b = {r["draw"]: rid[r["sig"]] for r in rows if r["sigma"] == SIGMA_TABLE}
    trans = {}
    for n in range(N_DRAWS):
        trans[(a[n], b[n])] = trans.get((a[n], b[n]), 0) + 1
    summary["transitions_1_to_5"] = {f"{i}->{j}": c for (i, j), c in sorted(trans.items())}
    summary["stay_share_1_to_5"] = sum(c for (i, j), c in trans.items() if i == j) / N_DRAWS

    top = [k for k in range(1, min(7, len(order)) + 1)]

    # figure 1: stacked regime probabilities vs sigma
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    style(ax)
    xs = np.arange(len(SIGMAS))
    bottom = np.zeros(len(SIGMAS))
    for k in top + ["other"]:
        if k == "other":
            vals = np.array([sum(t["p"] for kk, t in per_sigma[str(s)]["regimes"].items() if int(kk) not in top)
                             for s in SIGMAS])
            col, lab = OTHER, "other regimes"
        else:
            vals = np.array([per_sigma[str(s)]["regimes"].get(str(k), {"p": 0})["p"] for s in SIGMAS])
            loss_k = next(r["loss"] for r in rows if rid[r["sig"]] == k)
            col, lab = REGIME_COLOURS[k - 1], f"R{k}  ({loss_k:.1f}% of V)"
        ax.bar(xs, vals, 0.62, bottom=bottom, color=col, edgecolor="white", linewidth=1, label=lab)
        bottom += vals
    means = [per_sigma[str(s)]["mc_mean"] for s in SIGMAS]
    for x, m in zip(xs, means):
        ax.text(x, 1.02, f"{m:.1f}%", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels([f"{s:g}" for s in SIGMAS])
    ax.set_xlabel(r"$\sigma$ (\$/MWh)", color=MUTED); ax.set_ylabel("probability of the regime", color=MUTED)
    ax.set_ylim(0, 1.1)
    ax.set_title("Regime probabilities (bars) and mean loss = $\\sum_k p_k L_k$ (numbers on top)", fontsize=9.5)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "reg_probabilities.pdf")
    plt.close(fig)

    # figure 2: regime map on the slice (z_1, z_2), other z_t from a typical draw; sigma = 5
    k1, k2 = 0, 1
    grid, keys = regime_map(SIGMA_TABLE, k1, k2)
    ids = np.vectorize(lambda s: rid.get(s, 0))(keys)
    present = sorted(set(ids.ravel()))
    cmap_cols = [REGIME_COLOURS[k - 1] if 1 <= k <= len(REGIME_COLOURS) else OTHER for k in present]
    lut = {k: i for i, k in enumerate(present)}
    img = np.vectorize(lambda k: lut[k])(ids)
    loss_of = {rid[r["sig"]]: r["loss"] for r in rows}
    fig, ax = plt.subplots(figsize=(4.9, 3.9))
    ax.imshow(img, origin="lower", extent=(grid[0], grid[-1], grid[0], grid[-1]), aspect="equal",
              cmap=matplotlib.colors.ListedColormap(cmap_cols), interpolation="nearest")
    z0 = z_of(MAP_BASE_DRAW)
    ax.plot(z0[k1], z0[k2], "o", mfc="white", mec=INK, ms=7)
    ax.annotate(f"draw {MAP_BASE_DRAW}", (z0[k1], z0[k2]), xytext=(6, 6), textcoords="offset points", fontsize=7,
                color=INK)
    ax.set_xlabel(r"$z_1$ (error shape, 00--04h)", color=MUTED); ax.set_ylabel(r"$z_2$ (error shape, 04--08h)", color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap_cols]
    ax.legend(handles, [f"R{k}  ({loss_of[k]:.1f}% of V)" if k else "not in the sample" for k in present],
              fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f"Regimes on a slice of $z$-space ($\\sigma$={SIGMA_TABLE:g})", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "reg_map.pdf")
    plt.close(fig)
    summary["map_regimes"] = [int(k) for k in present]

    # figure 3: transition matrix sigma 1 -> 5, top regimes + "other"
    group = lambda k: k if k in top else 0
    ks = top + [0]
    M = np.zeros((len(ks), len(ks)))
    for (i, j), c in trans.items():
        M[ks.index(group(i)), ks.index(group(j))] += c / N_DRAWS
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max())
    for i in range(len(ks)):
        for j in range(len(ks)):
            if M[i, j] >= 0.0025:
                ax.text(j, i, f"{100 * M[i, j]:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 0.5 * M.max() else INK)
    names = [f"R{k}" if k else "other" for k in ks]
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(names, fontsize=7.5)
    ax.set_yticks(range(len(ks))); ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel(r"regime at $\sigma=5$", color=MUTED); ax.set_ylabel(r"regime at $\sigma=1$ (same $z$)", color=MUTED)
    ax.set_title("Where each draw goes when $\\sigma$ grows (% of draws)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "reg_transitions.pdf")
    plt.close(fig)
    summary["transition_matrix_top"] = {"labels": names, "percent": (100 * M).round(2).tolist()}

    summary["regime_patterns"] = {str(rid[s]): pattern(s) for s in order}
    (OUT / "reg_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sigma"}, indent=1))
    for s in SIGMAS:
        ps = per_sigma[str(s)]
        print(f"sigma {s:>4}: regimes {ps['n_regimes']:>2}, MC mean {ps['mc_mean']:.3f}, sum p*L {ps['decomposed_mean']:.3f}")
    print(json.dumps(per_sigma[str(SIGMA_TABLE)]["regimes"], indent=1))


if __name__ == "__main__":
    main()
