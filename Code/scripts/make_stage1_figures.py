"""Figures for docs/stage_1_structure.pdf: the stage-1 case (two batteries, one bus), Baldick's
benchmark on it, the two errors (price and terminal value) on one draw, and the zero-error check.

Writes PNGs to docs/figures/stage1/ and prints every number the PDF quotes. Same plot
conventions as notebooks/stage0_experiments.ipynb.

    python scripts/make_stage1_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import benchmark, bids, cases, forecasts, runner
from experiments.config import BidRule, ErrorModel, RealTime, RunSettings

OUT = ROOT / "docs" / "figures" / "stage1"
OUT.mkdir(parents=True, exist_ok=True)

COL = dict(renewable="#5aa469", thermal="#8a8f98", discharge="#eb6834", charge="#2a78d6", shed="#9b1c1c",
           price="#1f2328", forecast="#7b4bb7", bench="#9aa0a6", bound="#5b6068", bad="#c0392b")
ESS_COL = ["#c2410c", "#0b7285"]  # battery A, battery B (lines)
ESS_HATCH = ["", "////"]  # battery A solid, battery B hatched (stacked bars)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e3e5e8", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "font.size": 9, "axes.titlesize": 10, "hatch.color": "white"})

case = cases.stage1_case()
T, dT = case["T"], case["dT"]
NAMES = ["A", "B"]
t = np.arange(1, T + 1)
RULE, KAPPA, DELTA, PSI = "prelimit", BidRule().tie_break_reward, BidRule().delta, 10.0


def interval_labels():
    return [f"{int(k * dT):02d}-{int((k + 1) * dT):02d}h" for k in range(T)]


def set_interval_axis(ax):
    ax.set_xticks(t)
    ax.set_xticklabels(interval_labels(), fontsize=7.5)
    ax.set_xlim(0.4, T + 0.6)


def block_names():
    return ["renewable" if cost == 0 else f"thermal ({cost:g} $/MWh)" for G in case["gens"] for cost, _ in G["blocks"]]


def soc_axis(ax, smax):
    ax.axhspan(0, smax, color="#eef4fb", zorder=0)
    ax.axhline(smax, color=COL["bound"], ls=":", lw=1)
    ax.axhline(0, color=COL["bound"], ls=":", lw=1)
    ax.set_ylabel("SOC (MWh)")
    ax.set_xticks(np.arange(0.5, T + 1.5))
    ax.set_xticklabels([f"{int(k * dT)}h" for k in range(T + 1)])
    ax.set_xlim(0.4, T + 0.6)


def plot_soc(ax, S, s, **kw):
    ax.plot(np.arange(0.5, len(s) + 1), np.r_[S["s0"], s], marker="o", ms=4, **kw)


def soc_path(S, p, g):
    s, out = S["s0"], []
    for pt, gt in zip(p, g):
        s = S["g"] * s + dT * (S["a"] * pt - S["b"] * gt)
        out.append(s)
    return np.array(out)


def energy_balance(ax, q, p, g, shed, title):
    """Supply above zero (renewable, thermal, discharge A, discharge B, shed), charge below zero."""
    kw = dict(width=0.7, edgecolor="white", linewidth=0.8)
    bottom = np.zeros(T)
    for name, row in zip(block_names(), q):
        ax.bar(t, row, bottom=bottom, color=COL["renewable" if name == "renewable" else "thermal"], label=name, **kw)
        bottom += row
    for e in range(len(g)):
        ax.bar(t, g[e], bottom=bottom, color=COL["discharge"], hatch=ESS_HATCH[e], label=f"discharge {NAMES[e]}", **kw)
        bottom += g[e]
    if np.any(shed > 1e-9):
        ax.bar(t, shed, bottom=bottom, color=COL["shed"], label="load shed", **kw)
    below = np.zeros(T)
    for e in range(len(p)):
        ax.bar(t, -np.asarray(p[e]), bottom=below, color=COL["charge"], hatch=ESS_HATCH[e],
               label=f"charge {NAMES[e]} (< 0)", **kw)
        below -= np.asarray(p[e])
    ax.hlines(case["D"][0], t - 0.42, t + 0.42, color=COL["price"], lw=2.2, label="demand")
    ax.axhline(0, color=COL["bound"], lw=0.8)
    top = max(bottom.max(), case["D"][0].max())
    ax.set_ylim(min(1.15 * below.min(), -0.05 * top), 1.6 * top)
    ax.set_ylabel("MW")
    ax.set_title(title)
    set_interval_axis(ax)
    ax.legend(fontsize=7, ncol=4, loc="upper left")


# ---------------------------------------------------------------------------------------------
# 1. The system
# ---------------------------------------------------------------------------------------------
caps = np.array([qb for G in case["gens"] for _, qb in G["blocks"]], dtype=float)
fig, ax = plt.subplots(figsize=(9, 3.0))
bottom = np.zeros(T)
for name, cap in zip(block_names(), caps):
    ax.bar(t, cap, bottom=bottom, width=0.7, edgecolor="white",
           color=COL["renewable" if name == "renewable" else "thermal"], label=f"{name} capacity")
    bottom += cap
ax.hlines(case["D"][0], t - 0.42, t + 0.42, color=COL["price"], lw=2.2, label="demand")
for k in np.where(caps.sum(axis=0) < case["D"][0])[0]:
    ax.annotate(f"shortfall {case['D'][0][k] - caps[:, k].sum():.0f} MW\nwithout storage",
                (k + 1, case["D"][0][k]), xytext=(k + 1.45, case["D"][0][k] + 120), fontsize=8,
                arrowprops=dict(arrowstyle="->", color=COL["bad"]), color=COL["bad"])
set_interval_axis(ax)
ax.set_ylabel("MW"); ax.set_title("Generation capacity vs. demand (same system as stage 0)")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "s1_system.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------------------------------------
# 2. Baldick's benchmark
# ---------------------------------------------------------------------------------------------
bench = benchmark.solve_benchmark(case)
bench0 = benchmark.solve_benchmark(cases.day_case())
print(f"SW* = {bench.SW_star:,.2f}  SW_noESS = {bench.SW_noESS:,.2f}  V = {bench.V:,.2f}")
print(f"stage 0: SW* = {bench0.SW_star:,.2f}  V = {bench0.V:,.2f}")
print("lam*", bench.lam_star[0].round(3))
for e in range(2):
    print(NAMES[e], "p", bench.p_star[e].round(2), "g", bench.g_star[e].round(2), "s", bench.s_star[e].round(2),
          "pi", bench.pi_star[e].round(3), "nuU", bench.nuU_star[e].round(3), "nuL", bench.nuL_star[e].round(3))
print("q*", bench.q_star.round(2))

fig, axes = plt.subplots(3, 1, figsize=(9, 8.6))
energy_balance(axes[0], bench.q_star, bench.p_star, bench.g_star, np.zeros(T),
               "Benchmark dispatch (Baldick's SOC design, both batteries)")
ax = axes[1]
x = np.r_[t, T + 1] - 0.5
ax.step(x, np.r_[bench.lam_star[0], bench.lam_star[0][-1]], where="post", color=COL["price"], lw=2.0, label="lambda* (price)")
for e, S in enumerate(case["ess"]):
    ax.step(x, np.r_[bench.pi_star[e], bench.pi_star[e][-1]], where="post", color=ESS_COL[e], lw=1.4, ls="--",
            label=f"pi*_{NAMES[e]} (value of stored energy)")
    ax.axhline(S["w"], color=ESS_COL[e], ls=":", lw=1, label=f"w_{NAMES[e]} = {S['w']:g}")
set_interval_axis(ax); ax.set_ylabel("$/MWh"); ax.set_title("Prices and each battery's value of stored energy")
ax.set_ylim(-3, 62)
ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5)
ax = axes[2]
soc_axis(ax, case["ess"][0]["smax"])
for e, S in enumerate(case["ess"]):
    plot_soc(ax, S, bench.s_star[e], color=ESS_COL[e], lw=1.8, label=f"s*_{NAMES[e]}")
ax.set_title("State of charge (end of each interval)"); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "s1_benchmark.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------------------------------------
# 3. The two errors on one draw
# ---------------------------------------------------------------------------------------------
SIGMA, SIGMA_W, OMEGA, DRAW = 5.0, 5.0, 0.5, 0
error = ErrorModel(sigma_abs=SIGMA, sigma_w=SIGMA_W, omega=OMEGA)
settings = RunSettings(error=error, bid=BidRule(), realtime=RealTime(psi=PSI), n_ess=2, draws=1, setting_id="s1_fig")
lam_star = bench.lam_star[0]
rng = np.random.default_rng(settings.seed(DRAW))
chi = forecasts.ar1_unit_variance(T, error.rho, rng)
etas = [forecasts.ar1_unit_variance(T, error.rho, rng) for _ in range(2)]
zs = [np.sqrt(OMEGA) * chi + np.sqrt(1 - OMEGA) * eta for eta in etas]
lam_hat = forecasts.forecast_prices(lam_star, error, 2, T, np.random.default_rng(settings.seed(DRAW)))
assert np.allclose(lam_hat, [np.maximum(lam_star + SIGMA * z, 0.0) for z in zs])
w_true = np.array([S["w"] for S in case["ess"]])
w_hat = forecasts.terminal_values(w_true, error, np.random.default_rng(settings.seed_w(DRAW)))
print(f"draw {DRAW}: seed {settings.seed(DRAW)}, seed_w {settings.seed_w(DRAW)}")
print("lam_hat_A", lam_hat[0].round(2), "lam_hat_B", lam_hat[1].round(2), "w_hat", w_hat.round(2))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), gridspec_kw=dict(width_ratios=[1.1, 1.1, 0.8]))
ax = axes[0]
ax.axhline(0, color=COL["bound"], lw=0.8)
ax.plot(t, chi, "s--", color=COL["bench"], lw=1, ms=4, label="chi (common to A and B)")
for e in range(2):
    ax.plot(t, zs[e], "o-", color=ESS_COL[e], lw=1.5, label=f"z_{NAMES[e]} = sqrt(w)chi + sqrt(1-w)eta_{NAMES[e]}".replace("w)", "omega)"))
set_interval_axis(ax); ax.set_ylabel("standard units")
ax.set_title(f"1. Price-error shapes (omega = {OMEGA:g})"); ax.legend(fontsize=7)
ax = axes[1]
ax.plot(t, lam_star, "o-", color=COL["price"], lw=1.8, label="lambda* (truth)")
for e in range(2):
    ax.plot(t, lam_hat[e], "D-", color=ESS_COL[e], lw=1.3, ms=4, label=f"lambda_hat_{NAMES[e]}")
set_interval_axis(ax); ax.set_ylabel("$/MWh")
ax.set_title(f"2. Each owner's price forecast (sigma = {SIGMA:g} $/MWh)"); ax.legend(fontsize=7.5)
ax = axes[2]
grid = np.linspace(20, 75, 400)
for e in range(2):
    dens = np.exp(-0.5 * ((grid - w_true[e]) / SIGMA_W) ** 2)
    ax.fill_between(grid, e, e + 0.8 * dens, color=ESS_COL[e], alpha=0.18, lw=0)
    ax.plot([w_true[e]] * 2, [e, e + 0.8], color=ESS_COL[e], lw=2, label=f"true w_{NAMES[e]} = {w_true[e]:g}")
    ax.plot(w_hat[e], e + 0.35, "D", color=ESS_COL[e], ms=7, mec="white", label=f"w_hat_{NAMES[e]} = {w_hat[e]:.1f}")
ax.axvline(49, color=COL["bad"], ls="--", lw=1.2, label="49: sell-or-hold threshold")
ax.set_yticks([0.4, 1.4]); ax.set_yticklabels(NAMES); ax.set_xlabel("$/MWh"); ax.grid(axis="y", visible=False)
ax.set_title(f"3. Terminal-value error (sigma_w = {SIGMA_W:g})"); ax.legend(fontsize=6.8, loc="upper left")
ax.set_ylim(-0.1, 2.6)
fig.tight_layout(); fig.savefig(OUT / "s1_errors.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------------------------------------
# 4. Zero error: bids from each battery's own self-schedule at lambda* (B_star_p)
# ---------------------------------------------------------------------------------------------
bids0 = [bids.bid_star_p(S, lam_star, T, dT, RULE, KAPPA) for S in case["ess"]]
blocks0 = [bids.quantity_blocks(b, S, DELTA, full_capacity=False) for b, S in zip(bids0, case["ess"])]
out0 = runner.evaluate_blocks(case, blocks0, RealTime(psi=PSI))
loss0 = 100 * (bench.SW_star - out0.welfare.SW_real) / bench.V
bids_star = [bids.bid_star(S, bench.pi_star[e], bench.nuU_star[e], bench.nuL_star[e], bench.p_star[e], bench.g_star[e])
             for e, S in enumerate(case["ess"])]
out_star = runner.evaluate_blocks(case, [bids.quantity_blocks(b, S, DELTA, False) for b, S in zip(bids_star, case["ess"])],
                                  RealTime(psi=PSI))
print(f"B_star loss = {100 * (bench.SW_star - out_star.welfare.SW_real) / bench.V:.6f} %")
print(f"B_star_p loss = {loss0:.3f} %, shed = {dT * out0.shed_rt.sum():.1f} MWh, undeliverable = {out0.undeliverable_mwh:.1f} MWh")
print("lam_cd", out0.da.lam_cd.round(2), "welfare", out0.welfare)
for e in range(2):
    print(NAMES[e], "plan p0", bids0[e].p0.round(2), "g0", bids0[e].g0.round(2), "u", bids0[e].u.round(2), "c", bids0[e].c.round(2))
    print("   blocks charge", [[(round(pr, 2), round(q, 1)) for pr, q in blk] for blk in blocks0[e]["charge"]])
    print("   blocks discharge", [[(round(pr, 2), round(q, 1)) for pr, q in blk] for blk in blocks0[e]["discharge"]])
    print("   DA p", out0.da.p_da[e].round(2), "g", out0.da.g_da[e].round(2), "| RT p", out0.p_rt[e].round(2),
          "g", out0.g_rt[e].round(2), "s", out0.s_rt[e].round(1))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.9]))
ax = axes[0]
for e in range(2):
    off = -0.17 if e == 0 else 0.17
    for k in range(T):
        x0 = k + 1 + off - 0.14
        ax.add_patch(plt.Rectangle((x0, bids0[e].u[k]), 0.28, bids0[e].c[k] - bids0[e].u[k], fc=ESS_COL[e], alpha=0.13, ec="none"))
        ax.hlines(bids0[e].u[k], x0, x0 + 0.28, color=COL["charge"], lw=2.2)
        ax.hlines(bids0[e].c[k], x0, x0 + 0.28, color=COL["discharge"], lw=2.2)
    ax.text(1 + off, -6, NAMES[e], ha="center", fontsize=8, color=ESS_COL[e], fontweight="bold")
ax.plot(t, lam_star, "o", color=COL["price"], ms=6, mec="white", zorder=5)
ax.annotate("A sells at 41,\nB buys at 43.1:\nthe bids cross", (6, 42), xytext=(4.1, 18), fontsize=7.5, color=COL["bad"],
            arrowprops=dict(arrowstyle="->", color=COL["bad"]))
handles = [plt.Line2D([], [], color=COL["charge"], lw=2.2), plt.Line2D([], [], color=COL["discharge"], lw=2.2),
           plt.Line2D([], [], ls="none", marker="o", color=COL["price"])]
ax.legend(handles, ["u: charge bid", "c: discharge offer", "lambda*"], fontsize=7, loc="upper left", ncol=3)
set_interval_axis(ax); ax.set_ylabel("$/MWh"); ax.set_ylim(-9, 66)
ax.set_title("Bids at zero error (left bar A, right bar B)")


def merit_order(ax, k):
    D = case["D"][0][k]
    supply = [(c, cap[k]) for c, cap in zip([cost for G in case["gens"] for cost, _ in G["blocks"]], caps)]
    for e in range(2):
        supply += [(pr, mw, e) for pr, mw in blocks0[e]["discharge"][k]]
    supply = sorted([s if len(s) == 3 else (*s, None) for s in supply if s[1] > 1e-9])
    demand = [(1000.0, D, None)] + sorted([(pr, mw, e) for e in range(2) for pr, mw in blocks0[e]["charge"][k] if mw > 1e-9],
                                          reverse=True)
    for curve, col, lab in [(supply, COL["discharge"], "supply"), (demand, COL["charge"], "demand")]:
        xs = np.r_[0, np.cumsum([mw for _, mw, _ in curve])]
        ys = [min(pr, 70) for pr, _, _ in curve]
        ax.step(xs, np.r_[ys, ys[-1]], where="post", color=col, lw=1.6, label=lab)
        for (pr, mw, e), x0 in zip(curve, xs[:-1]):
            if e is not None and mw > 50:
                ax.text(x0 + mw / 2, min(pr, 70) + 1.5, f"{NAMES[e]}", ha="center", fontsize=7.5, color=ESS_COL[e])
    cleared = D + sum(out0.da.p_da[e][k] for e in range(2))
    ax.plot(cleared, out0.da.lam_cd[k], "o", color=COL["price"], ms=7, zorder=5, label="cleared point")
    ax.axhline(lam_star[k], color=COL["bench"], ls=":", lw=1.2, label="lambda*")
    ax.set_xlim(0, 2300); ax.set_ylim(-3, 73)
    ax.set_title(f"Merit order {interval_labels()[k]}: {out0.da.lam_cd[k]:.2f} $/MWh (lambda* {lam_star[k]:g})")
    ax.set_xlabel("MW"); ax.set_ylabel("$/MWh"); ax.legend(fontsize=7, loc="lower right")


merit_order(axes[1], T - 1)
ax = axes[2]
soc_axis(ax, case["ess"][0]["smax"])
for e, S in enumerate(case["ess"]):
    s_da = soc_path(S, out0.da.p_da[e], out0.da.g_da[e])
    plot_soc(ax, S, s_da, color=ESS_COL[e], lw=1.6, label=f"{NAMES[e]}: if day-ahead delivered")
    plot_soc(ax, S, out0.s_rt[e], color=ESS_COL[e], lw=1.0, ls=":", label=f"{NAMES[e]}: real time")
ax.set_ylim(-3700, 4300)
ax.set_title("SOC: the day-ahead schedule is impossible"); ax.legend(fontsize=6.8, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "s1_zero_error.png", dpi=170); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.3))
energy_balance(ax, out0.q_rt, out0.p_rt, out0.g_rt, out0.shed_rt, "Real time at zero error (B_star_p): what was physically delivered")
fig.tight_layout(); fig.savefig(OUT / "s1_zero_error_rt.png", dpi=170); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
