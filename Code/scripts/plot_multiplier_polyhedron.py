"""Figures of the multiplier polyhedron Pi (Prop. 5) for docs/model_structure.tex.

Pi(lam) is the set of (pi, nu_up, nu_lo) in R^{3T} from which valid bids
    u = alpha*(pi - nu_up) - eps,   c = zeta + beta*(pi + nu_lo)
can be built, given a price vector `lam` and the ESS's own self-schedule at `lam`
(verify_price_based.prop5_polyhedron). It is unbounded in the nu directions (at an hour
where the ESS is idle any less aggressive bid is still consistent), but its projection
on pi is bounded, so every figure is drawn in pi-coordinates. Projections are exact:
each point is an LP optimum of a direction over Pi (support function), and the shape is
the convex hull of those optima.

Writes, to docs/figures/:
  fig_poly_true.pdf      Pi(lam_star): the (pi_2, pi_4) face and the pi-path view
  fig_poly_forecast.pdf  Pi(lam_hat) for forecast draws vs Pi(lam_star), 2-D projections
  fig_poly_3d.pdf        one forecast draw, projection on (pi_2, pi_3, pi_4)

Usage: python scripts/plot_multiplier_polyhedron.py
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
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

import verify_price_based as vpb
from experiments import benchmark, bids, cases, config, forecasts, metrics, runner
from experiments.config import BidRule, ErrorModel, RealTime

OUT = ROOT / "docs" / "figures"
BLUE, ORANGE, INK, MUTED, GRID, FACE = "#2a78d6", "#eb6834", "#1f2328", "#5b6068", "#e3e5e8", "#cde2fb"
CMAP = matplotlib.colormaps["Oranges"]
SIGMAS_ABS = (1.0, 5.0)  # $/MWh, two of the stage-0 error sizes
N_DRAWS = 8
DRAW_SEED = 1000
RT = RealTime(psi=config.STAGE0_PSI_FOR_STOCHASTIC)
DELTA = BidRule().delta
KAPPA = BidRule().tie_break_reward  # self-schedule tie-breaker (bids.self_schedule)


# ---------------------------------------------------------------------------
# Polyhedron geometry
# ---------------------------------------------------------------------------
def polyhedron_at(S: dict, lam: np.ndarray, T: int, dT: float) -> tuple[dict, np.ndarray, np.ndarray]:
    """Pi(lam) and the (tie-broken) self-schedule (p, g) it is built from."""
    p, g, s = bids.self_schedule(S, lam, T, dT, KAPPA)
    return vpb.prop5_polyhedron(S, lam, p, g, s, T, dT), p, g


def support_point(poly: dict, cost: np.ndarray) -> np.ndarray:
    """argmin of `cost` . x over Pi (x in R^{3T})."""
    r = linprog(cost, A_ub=poly["Aub"], b_ub=poly["bub"], A_eq=poly["Aeq"], b_eq=poly["beq"],
                bounds=poly["bnd"], method="highs")
    assert r.status == 0, r.message
    return r.x


def projection(poly: dict, coords: list[int], T: int, n_dir: int, rng: np.random.Generator) -> np.ndarray:
    """Vertices of the projection of Pi on the pi-coordinates `coords` (unique rows).
    2-D: evenly spaced directions; 3-D: random directions on the sphere."""
    k = len(coords)
    if k == 2:
        ang = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)
        dirs = np.c_[np.cos(ang), np.sin(ang)]
    else:
        dirs = rng.normal(size=(n_dir, k))
    pts = []
    for d in dirs:
        cost = np.zeros(3 * T)
        cost[coords] = -d  # maximise d . pi[coords]
        pts.append(support_point(poly, cost)[coords])
    # deduplicate on rounded keys but keep the exact LP values (they are re-used as
    # equality right-hand sides in minimal_rent_point)
    _, keep = np.unique(np.round(pts, 6), axis=0, return_index=True)
    return np.array(pts)[np.sort(keep)]


def polygon(pts: np.ndarray) -> np.ndarray:
    """Closed outline of a 2-D point set; a segment or a point is returned as is."""
    if len(pts) < 3 or np.linalg.matrix_rank(pts - pts[0], tol=1e-6) < 2:
        return pts
    h = ConvexHull(pts)
    return pts[np.r_[h.vertices, h.vertices[0]]]


def polytope_edges(hull: ConvexHull) -> list[tuple[int, int]]:
    """Edges of a 3-D hull, without the triangulation's diagonals: an edge is kept only
    if the two triangles sharing it lie on different planes."""
    owners: dict[tuple[int, int], list[int]] = {}
    for f, simplex in enumerate(hull.simplices):
        for i in range(3):
            e = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
            owners.setdefault(e, []).append(f)
    normals = hull.equations[:, :3]
    return [e for e, (f1, f2) in owners.items() if abs(normals[f1] @ normals[f2]) < 1 - 1e-9]


def minimal_rent_point(poly: dict, pi: np.ndarray, T: int) -> np.ndarray:
    """The point of Pi with the given pi and the smallest sum of nu -- the same choice
    `bids._prelimit_vertex` makes (nu only as large as the costate forces it)."""
    fixed = dict(poly)
    rows = np.zeros((T, 3 * T))
    rows[np.arange(T), np.arange(T)] = 1.0
    fixed["Aeq"] = np.vstack([poly["Aeq"], rows])
    fixed["beq"] = list(poly["beq"]) + list(pi)
    cost = np.r_[np.zeros(T), np.ones(2 * T)]
    return support_point(fixed, cost)


# ---------------------------------------------------------------------------
# Welfare of a point of Pi, through the experiment pipeline
# ---------------------------------------------------------------------------
def dsw_pct(case: dict, bench, bid: bids.Bid) -> float:
    S = case["ess"][0]
    blocks = bids.quantity_blocks(bid, S, DELTA, full_capacity=False)
    out = runner.evaluate_blocks(case, [blocks], RT)
    return metrics.dsw_pct(metrics.dsw(bench.SW_star, out.welfare.SW_real), bench.V)


def bid_from_point(S: dict, x: np.ndarray, p0: np.ndarray, g0: np.ndarray, T: int) -> bids.Bid:
    pi, nu_up, nu_lo = x[:T], x[T : 2 * T], x[2 * T :]
    return bids.Bid(u=S["a"] * (pi - nu_up) - S["eps"], c=S["zet"] + S["b"] * (pi + nu_lo), p0=p0, g0=g0)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def style(ax) -> None:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#9aa0a6")
    ax.tick_params(colors=MUTED, labelsize=8, length=2.5)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_true(case: dict, bench) -> dict:
    """Pi(lam_star): its free face in (pi_2, pi_4) and the pi paths of its corners."""
    S, T, dT = case["ess"][0], case["T"], case["dT"]
    lam = bench.lam_star[0]
    poly, p0, g0 = polyhedron_at(S, lam, T, dT)
    rng, _ = vpb.pi_range(poly, T)
    free = [t for t, (lo, hi) in enumerate(rng) if hi - lo > 1e-6]
    assert free == [1, 3], f"expected pi_2 and pi_4 free at lam_star, got {[t + 1 for t in free]}"

    corners = projection(poly, free, T, 64, np.random.default_rng(0))
    pi_of = lambda c: np.array([c[free.index(t)] if t in free else rng[t][0] for t in range(T)])
    named = {
        "Baldick dual": bench.pi_star[0],
        "prelimit": bids._select_vertex(poly, S, T, "prelimit")[0],
        "solver": bids._select_vertex(poly, S, T, "solver")[0],
    }
    rows = []
    for c in corners:
        pi = pi_of(c)
        bid = bid_from_point(S, minimal_rent_point(poly, pi, T), p0, g0, T)
        out = runner.evaluate_blocks(case, [bids.quantity_blocks(bid, S, DELTA, full_capacity=False)], RT)
        name = next((n for n, v in named.items() if np.allclose(v, pi, atol=1e-6)), "fourth corner")
        rows.append(dict(name=name, pi=pi, bid=bid, p_da=out.da.p_da[0], g_da=out.da.g_da[0],
                         dsw=metrics.dsw_pct(metrics.dsw(bench.SW_star, out.welfare.SW_real), bench.V)))
    order = ["Baldick dual", "prelimit", "solver", "fourth corner"]
    rows.sort(key=lambda r: order.index(r["name"]))
    for letter, r in zip("ABCD", rows):
        r["letter"] = letter
    same_da = all(np.allclose(r["p_da"], rows[0]["p_da"]) and np.allclose(r["g_da"], rows[0]["g_da"]) for r in rows)
    same_loss = max(r["dsw"] for r in rows) - min(r["dsw"] for r in rows) < 1e-9

    fig = plt.figure(figsize=(10.5, 5.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1], hspace=0.55, wspace=0.28)
    a = fig.add_subplot(gs[:, 0])
    style(a)
    ring = polygon(corners)
    a.fill(ring[:, 0], ring[:, 1], fc=FACE, ec=BLUE, lw=1.4, zorder=1)
    for r in rows:
        x, y = r["pi"][1], r["pi"][3]
        a.plot(x, y, "o", ms=17, mfc="white", mec=INK, mew=1.4, zorder=3)
        a.text(x, y, r["letter"], ha="center", va="center", fontsize=10, fontweight="bold", color=INK, zorder=4)
        dy = 6.5 if y > 25 else -6.5
        a.text(x, y + dy, r["name"], ha="center", va="center", fontsize=8, color=INK)
    a.text(25, 25, "every point of the square\nis a valid choice:\nall are consistent with\n" r"$\lambda^\star$ and the planned schedule",
           ha="center", va="center", fontsize=8, color=MUTED)
    a.set_xlim(-9, 59); a.set_ylim(-12, 61)
    a.set_xlabel(r"$\pi_2$: value of a stored MWh at the end of 04-08h (\$/MWh)", fontsize=8.5, color=MUTED)
    a.set_ylabel(r"$\pi_4$: value of a stored MWh at the end of 12-16h (\$/MWh)", fontsize=8.5, color=MUTED)
    a.set_title(r"(a) The choices: $\Pi(\lambda^\star)$ is a square", fontsize=10, color=INK)

    t = np.arange(1, T + 1)
    ref = rows[0]["bid"]
    for k, r in enumerate(rows):
        ax = fig.add_subplot(gs[k // 2, 1 + k % 2])
        style(ax)
        b = r["bid"]
        differs = (np.abs(b.u - ref.u) > 1e-6) | (np.abs(b.c - ref.c) > 1e-6)
        for tt in range(T):
            if differs[tt]:
                ax.axvspan(tt + 0.5, tt + 1.5, color="#fff4cc", zorder=0)
            ax.add_patch(plt.Rectangle((tt + 1 - 0.3, b.u[tt]), 0.6, b.c[tt] - b.u[tt], fc="#eef0f3", ec="none", zorder=1))
            ax.hlines(b.u[tt], tt + 0.7, tt + 1.3, color=BLUE, lw=2.2, zorder=2)
            ax.hlines(b.c[tt], tt + 0.7, tt + 1.3, color=ORANGE, lw=2.2, zorder=2)
        ax.plot(t, bench.lam_star[0], "o", color=INK, ms=5.5, mec="white", zorder=3)
        ax.set_xticks(t); ax.set_xticklabels(["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"], fontsize=6.5)
        ax.set_xlim(0.5, T + 0.5); ax.set_ylim(-5, 56)
        ax.set_title(f"{r['letter']}: {r['name']}   ($\\pi_2$={r['pi'][1]:.1f}, $\\pi_4$={r['pi'][3]:.1f})", fontsize=8.5)
        if k % 2 == 0:
            ax.set_ylabel("$/MWh", fontsize=8, color=MUTED)
    handles = [plt.Line2D([], [], color=BLUE, lw=2.2), plt.Line2D([], [], color=ORANGE, lw=2.2),
               plt.Line2D([], [], ls="none", marker="o", color=INK), plt.Rectangle((0, 0), 1, 1, fc="#fff4cc")]
    fig.legend(handles, ["charge bid u (buys if price <= u)", "discharge offer c (sells if price >= c)",
                         r"true price $\lambda^\star$", "bid differs from corner A"],
               loc="upper center", bbox_to_anchor=(0.66, 0.085), ncol=2, fontsize=7.5, frameon=False)
    verdict = ("same day-ahead schedule and same loss in all four: " f"{rows[0]['dsw']:.2f}% of V"
               if same_da and same_loss else "the corners give different outcomes")
    fig.text(0.655, 0.955, f"(b) The bids each corner produces:\n{verdict}", ha="center", va="center", fontsize=9.5,
             color=INK)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.18)
    fig.savefig(OUT / "fig_poly_true.pdf")
    plt.close(fig)
    return dict(free=[t + 1 for t in free], corners=rows, same_da=same_da, same_loss=same_loss,
                U=poly["U"], L=poly["L"])


def fig_forecast_ranges(case: dict, bench) -> dict:
    """For each interval: the range of valid pi_t at lam_star (wide bar) and for several forecast draws
    (narrow bars, coloured by the draw's loss), with the value the prelimit rule picks (dot)."""
    S, T, dT = case["ess"][0], case["T"], case["dT"]
    poly_star, _, _ = polyhedron_at(S, bench.lam_star[0], T, dT)
    rng_star, _ = vpb.pi_range(poly_star, T)
    data = {s: forecast_draws(case, bench, s) for s in SIGMAS_ABS}
    for ds in data.values():
        for d in ds:
            d["rng"], _ = vpb.pi_range(d["poly"], T)
    vmax = max(d["dsw"] for ds in data.values() for d in ds)
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = matplotlib.colors.ListedColormap(CMAP(np.linspace(0.3, 1, 256)))
    labels = ["00-04h", "04-08h", "08-12h", "12-16h", "16-20h", "20-24h"]
    fig, axes = plt.subplots(len(SIGMAS_ABS), 1, figsize=(10, 3.2 * len(SIGMAS_ABS)), sharex=True)
    for ax, s in zip(np.atleast_1d(axes), SIGMAS_ABS):
        style(ax)
        for tt, (lo, hi) in enumerate(rng_star):
            ax.add_patch(plt.Rectangle((tt + 1 - 0.42, lo), 0.84, max(hi - lo, 0.8), fc=FACE, ec=BLUE, lw=1.2, zorder=1))
        offs = np.linspace(-0.3, 0.3, len(data[s]))
        for off, d in zip(offs, data[s]):
            col = cmap(norm(d["dsw"]))
            for tt, (lo, hi) in enumerate(d["rng"]):
                ax.add_patch(plt.Rectangle((tt + 1 + off - 0.025, lo), 0.05, max(hi - lo, 0.8), fc=col, ec="none", zorder=2))
                ax.plot(tt + 1 + off, d["pi_pre"][tt], "o", ms=3.5, mfc=INK, mec="white", mew=0.5, zorder=3)
        ax.set_xticks(np.arange(1, T + 1)); ax.set_xticklabels(labels)
        ax.set_xlim(0.45, T + 0.55)
        ax.set_ylabel(r"$\pi_t$ (\$/MWh)", fontsize=8.5, color=MUTED)
        ax.set_title(rf"$\sigma$ = {s:g} \$/MWh: wide blue bar = valid range at $\lambda^\star$; "
                     rf"thin bars = valid range for {N_DRAWS} forecast draws; dot = value the prelimit rule picks",
                     fontsize=8.5, loc="left")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.93, bottom=0.08, hspace=0.3)
    cb = fig.colorbar(sm, cax=fig.add_axes((0.905, 0.2, 0.015, 0.6)))
    cb.set_label(r"$\Delta SW$ of the draw (% of $V$)", fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7.5, colors=MUTED)
    fig.savefig(OUT / "fig_poly_ranges.pdf")
    plt.close(fig)
    return {s: [round(d["dsw"], 1) for d in ds] for s, ds in data.items()}


def forecast_draws(case: dict, bench, sigma_abs: float) -> list[dict]:
    S, T, dT = case["ess"][0], case["T"], case["dT"]
    lam = bench.lam_star[0]
    draws = []
    for d in range(N_DRAWS):
        rng = np.random.default_rng(DRAW_SEED + d)
        lam_hat = forecasts.forecast_prices(lam, ErrorModel(sigma_abs=sigma_abs), 1, T, rng)[0]
        poly, _, _ = polyhedron_at(S, lam_hat, T, dT)
        bid = bids.bid_fcst(S, lam_hat, T, dT, "prelimit", KAPPA)
        pi_pre = bids._select_vertex(poly, S, T, "prelimit")[0]
        draws.append(dict(poly=poly, pi_pre=pi_pre, dsw=dsw_pct(case, bench, bid)))
    return draws


def fig_forecast(case: dict, bench) -> dict:
    """Pi(lam_hat) projected on (pi_2, pi_4) and (pi_3, pi_5), several draws per sigma."""
    S, T, dT = case["ess"][0], case["T"], case["dT"]
    poly_star, _, _ = polyhedron_at(S, bench.lam_star[0], T, dT)
    planes = [(1, 3), (2, 4)]
    data = {s: forecast_draws(case, bench, s) for s in SIGMAS_ABS}
    vmax = max(d["dsw"] for ds in data.values() for d in ds)
    norm = Normalize(vmin=0, vmax=vmax)

    fig, axs = plt.subplots(len(SIGMAS_ABS), len(planes), figsize=(7.2, 6.0))
    for i, s in enumerate(SIGMAS_ABS):
        for j, (ti, tj) in enumerate(planes):
            ax = axs[i, j]
            style(ax)
            for d in data[s]:
                ring = polygon(projection(d["poly"], [ti, tj], T, 72, None))
                ax.fill(ring[:, 0], ring[:, 1], fc=CMAP(0.15), ec=MUTED, lw=0.6, alpha=0.35, zorder=1)
            star = polygon(projection(poly_star, [ti, tj], T, 72, None))
            if len(star) >= 3:
                ax.fill(star[:, 0], star[:, 1], fc=FACE, ec=BLUE, lw=1.6, zorder=2)
            else:
                ax.plot(star[:, 0], star[:, 1], color=BLUE, marker="o", ms=7, lw=2, zorder=2)
            for d in data[s]:
                ax.plot(d["pi_pre"][ti], d["pi_pre"][tj], marker="s", ms=7, ls="none", mec=INK, mew=0.7,
                        mfc=CMAP(0.25 + 0.75 * norm(d["dsw"])), zorder=3)
            ax.set_xlabel(rf"$\pi_{ti + 1}$ (\$/MWh)", fontsize=8.5, color=MUTED)
            ax.set_ylabel(rf"$\pi_{tj + 1}$ (\$/MWh)", fontsize=8.5, color=MUTED)
            ax.set_title(rf"$\sigma$ = {s:g} \$/MWh: projection on $(\pi_{ti + 1},\pi_{tj + 1})$",
                         fontsize=9, color=INK)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=FACE, ec=BLUE, lw=1.6),
        plt.Rectangle((0, 0), 1, 1, fc=CMAP(0.15), ec=MUTED, lw=0.6, alpha=0.6),
        plt.Line2D([], [], marker="s", ms=7, ls="none", mec=INK, mfc=CMAP(0.6)),
    ]
    fig.legend(handles, [r"$\Pi(\lambda^\star)$", rf"$\Pi(\hat\lambda)$, {N_DRAWS} draws",
                         r"prelimit choice (fill = $\Delta SW\%$)"],
               loc="lower center", ncol=3, fontsize=8, frameon=False)
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(CMAP(np.linspace(0.25, 1, 256))), norm=norm)
    fig.subplots_adjust(left=0.09, right=0.85, bottom=0.12, top=0.95, hspace=0.45, wspace=0.32)
    cb = fig.colorbar(sm, cax=fig.add_axes((0.89, 0.25, 0.018, 0.55)))
    cb.set_label(r"$\Delta SW$ (% of $V$) of the prelimit bid", fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7.5, colors=MUTED)
    fig.savefig(OUT / "fig_poly_forecast.pdf")
    plt.close(fig)
    return {s: [round(d["dsw"], 1) for d in ds] for s, ds in data.items()}


def fig_3d(case: dict, bench, sigma_abs: float, draw: int) -> dict:
    """One forecast draw: the 3-D projection of Pi(lam_hat) on (pi_2, pi_3, pi_4), with the
    flat Pi(lam_star) (it lies in the plane pi_3 = const) drawn inside the same axes."""
    S, T, dT = case["ess"][0], case["T"], case["dT"]
    coords = [1, 2, 3]
    lam_hat = forecasts.forecast_prices(bench.lam_star[0], ErrorModel(sigma_abs=sigma_abs), 1, T,
                                        np.random.default_rng(DRAW_SEED + draw))[0]
    poly, _, _ = polyhedron_at(S, lam_hat, T, dT)
    V = projection(poly, coords, T, 600, np.random.default_rng(1))
    hull = ConvexHull(V)
    poly_star, _, _ = polyhedron_at(S, bench.lam_star[0], T, dT)
    Vs = projection(poly_star, coords, T, 200, np.random.default_rng(2))
    ring = polygon(Vs[:, [0, 2]])
    pi3_star = Vs[0, 1]
    pre = bids._select_vertex(poly, S, T, "prelimit")[0][coords]

    fig = plt.figure(figsize=(5.6, 4.6))
    ax = fig.add_subplot(projection="3d")
    faces = [V[s] for s in hull.simplices]
    ax.add_collection3d(Poly3DCollection(faces, fc=CMAP(0.2), ec="none", alpha=0.25))
    for a, b in polytope_edges(hull):
        ax.plot(*V[[a, b]].T, color=CMAP(0.7), lw=0.9)
    ax.add_collection3d(Poly3DCollection([np.c_[ring[:, 0], np.full(len(ring), pi3_star), ring[:, 1]]],
                                         fc=FACE, ec=BLUE, lw=1.4, alpha=0.9))
    ax.scatter(*pre, marker="s", s=45, c=ORANGE, edgecolors=INK, depthshade=False)
    ax.scatter(*bench.pi_star[0][coords], marker="o", s=45, c=INK, edgecolors="white", depthshade=False)
    lo, hi = np.minimum(V.min(0), Vs.min(0)), np.maximum(V.max(0), Vs.max(0))
    pad = 0.05 * (hi - lo + 1)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    ax.set_xlabel(r"$\pi_2$", fontsize=9)
    ax.set_ylabel(r"$\pi_3$", fontsize=9)
    ax.set_zlabel(r"$\pi_4$", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=22, azim=-58)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=CMAP(0.2), ec=CMAP(0.55), alpha=0.5),
        plt.Rectangle((0, 0), 1, 1, fc=FACE, ec=BLUE),
        plt.Line2D([], [], marker="s", ls="none", mfc=ORANGE, mec=INK),
        plt.Line2D([], [], marker="o", ls="none", mfc=INK, mec="white"),
    ]
    ax.legend(handles, [rf"$\Pi(\hat\lambda)$, $\sigma$={sigma_abs:g}, draw {draw}", r"$\Pi(\lambda^\star)$ (flat)",
                        "prelimit choice", "Baldick dual"], fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_poly_3d.pdf")
    plt.close(fig)
    return dict(lam_hat=lam_hat.round(2), n_vertices=len(hull.vertices), pi3_star=pi3_star)


def affine_dim(poly: dict, T: int, rng: np.random.Generator) -> int:
    """Dimension of the pi-projection of Pi (rank of support points over random directions)."""
    V = projection(poly, list(range(T)), T, 300, rng)
    return int(np.linalg.matrix_rank(V - V[0], tol=1e-6))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    case = cases.day_case()
    bench = benchmark.solve_benchmark(case)
    S, T, dT = case["ess"][0], case["T"], case["dT"]

    info = fig_true(case, bench)
    print("Pi(lam_star): free pi coordinates", info["free"], "| same day-ahead schedule at every corner:",
          info["same_da"], "| same loss:", info["same_loss"])
    for r in info["corners"]:
        print(f"  {r['letter']} {r['name']:14s} pi={np.round(r['pi'], 3)}  dSW={r['dsw']:.2f}% of V")
    print("forecast ranges, dSW% per draw:", fig_forecast_ranges(case, bench))
    poly_star, _, _ = polyhedron_at(S, bench.lam_star[0], T, dT)
    print("  dim of pi-projection at lam_star:", affine_dim(poly_star, T, np.random.default_rng(0)))
    for s in SIGMAS_ABS:
        dims = []
        for d in range(N_DRAWS):
            lam_hat = forecasts.forecast_prices(bench.lam_star[0], ErrorModel(sigma_abs=s), 1, T,
                                                np.random.default_rng(DRAW_SEED + d))[0]
            dims.append(affine_dim(polyhedron_at(S, lam_hat, T, dT)[0], T, np.random.default_rng(0)))
        print(f"  sigma={s:g}: dim of pi-projection per draw {dims}")

    print("forecast dSW% (prelimit) per draw:", fig_forecast(case, bench))
    print("3-D draw:", fig_3d(case, bench, sigma_abs=2.0, draw=0))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
