"""The experiment loop -- experiments.md section 4.8.

One CSV row per draw; `meta.json` per stage with the config, git commit, date, package
versions and the benchmark values. Settings whose CSV is already complete are skipped
(resume); nothing in `results/` is overwritten without the caller passing `overwrite=True`
explicitly (experiments.md section 8: "never overwrite results without asking").
"""

from __future__ import annotations

import datetime
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import bids, forecasts, market, metrics, realtime, welfare
from .benchmark import Benchmark
from .config import RealTime, RunSettings


@dataclass
class Outcome:
    da: market.DayAheadResult
    welfare: welfare.WelfareResult
    undeliverable_mwh: float
    shed_rt: np.ndarray  # (T,) MW
    p_rt: list[np.ndarray]  # per ESS, (T,) MW, SOC-feasible charge after repair
    g_rt: list[np.ndarray]  # per ESS, (T,) MW, SOC-feasible discharge after repair
    s_rt: list[np.ndarray]  # per ESS, (T,) MWh
    q_rt: np.ndarray  # (n_blocks, T) MW, generators after redispatch
    redispatched_mw: np.ndarray  # (T,) MW, sum_j |q_rt - q_da|


def evaluate_blocks(case: dict, ess_blocks: list[dict], rt: RealTime) -> Outcome:
    """Everything after bid construction: day-ahead clearing (4.4), real-time repair and
    redispatch (4.5), realised welfare (4.6). `ess_blocks[e]` is the output of
    `bids.quantity_blocks` for ESS `e`; batteries are taken from `case["ess"]` in order."""
    T, dT = case["T"], case["dT"]
    n_ess = len(ess_blocks)
    da = market.clear_cd(case, ess_blocks, voll=rt.voll)

    repairs = [realtime.feasible_schedule(case["ess"][e], da.p_da[e], da.g_da[e], T, dT) for e in range(n_ess)]
    p_rt = [r.p for r in repairs]
    g_rt = [r.g for r in repairs]
    s_T = [r.s[-1] for r in repairs]
    undeliverable_mwh = sum(r.undeliverable_mwh for r in repairs)

    n_blocks = da.q_da.shape[0]
    q_rt = np.zeros((n_blocks, T))
    shed_rt = np.zeros(T)
    redispatched = np.zeros(T)
    for t in range(T):
        net_needed = case["D"][0, t] - sum(g_rt[e][t] for e in range(n_ess)) + sum(p_rt[e][t] for e in range(n_ess))
        rd = realtime.redispatch_hour(case, t, da.q_da[:, t], net_needed, rt.psi, rt.voll)
        q_rt[:, t] = rd.q_rt
        shed_rt[t] = rd.shed
        redispatched[t] = rd.redispatched_mw

    w = welfare.realised_welfare(
        case, q_rt, case["ess"][:n_ess], p_rt, g_rt, s_T, shed_rt, redispatched, rt.psi, rt.voll,
    )
    return Outcome(
        da=da, welfare=w, undeliverable_mwh=undeliverable_mwh, shed_rt=shed_rt,
        p_rt=p_rt, g_rt=g_rt, s_rt=[r.s for r in repairs], q_rt=q_rt, redispatched_mw=redispatched,
    )


def run_draw(case: dict, bench: Benchmark, settings: RunSettings, draw: int) -> dict:
    """One row: one (setting, draw) combination."""
    T, dT = case["T"], case["dT"]
    n_ess = settings.n_ess
    rng = np.random.default_rng(settings.seed(draw))
    lam_star_e = bench.lam_star[0]  # single bus
    family = settings.bid.family

    # One draw for all batteries, so the common component chi is shared (omega) -- drawn
    # before the loop, not per battery.
    lam_hat = forecasts.forecast_prices(lam_star_e, settings.error, n_ess, T, rng) if family == "B_fcst" else None
    # Each owner's own estimate of its terminal value; only the owner-built bids use it.
    w_true = np.array([case["ess"][e]["w"] for e in range(n_ess)])
    w_hat = forecasts.terminal_values(w_true, settings.error, np.random.default_rng(settings.seed_w(draw)))

    ess_blocks = []
    for e in range(n_ess):
        S = case["ess"][e]
        S_own = dict(S, w=float(w_hat[e]))  # the battery as its owner believes it to be
        if family == "B_star":
            bid = bids.bid_star(S, bench.pi_star[e], bench.nuU_star[e], bench.nuL_star[e], bench.p_star[e], bench.g_star[e])
        elif family == "B_star_p":
            bid = bids.bid_star_p(S, lam_star_e, T, dT, settings.bid.rule, settings.bid.tie_break_reward)
        elif family == "B_fcst":
            bid = bids.bid_fcst(S_own, lam_hat[e], T, dT, settings.bid.rule, settings.bid.tie_break_reward)
        elif family == "B_priv":
            bid = bids.bid_priv(S_own, T)
        else:
            raise ValueError(f"unknown bid family {family!r}")
        ess_blocks.append(bids.quantity_blocks(bid, S, settings.bid.delta, full_capacity=(family == "B_priv")))

    out = evaluate_blocks(case, ess_blocks, settings.realtime)
    da, w = out.da, out.welfare
    dsw_val = metrics.dsw(bench.SW_star, w.SW_real)
    ess_profit_da = sum(
        welfare.ess_profit_day_ahead(case["ess"][e], da.p_da[e], da.g_da[e], da.lam_cd, dT) for e in range(n_ess)
    )
    pr = [welfare.ess_profit(case["ess"][e], da.p_da[e], da.g_da[e], out.p_rt[e], out.g_rt[e], out.s_rt[e][-1],
                             da.lam_cd, settings.realtime.psi, dT) for e in range(n_ess)]

    w_cols = {f"w_hat_{e}": float(w_hat[e]) for e in range(n_ess)} if settings.error.sigma_w > 0 else {}
    return dict(
        stage=settings.setting_id.split("_")[0],
        setting_id=settings.setting_id,
        family=settings.bid.family,
        rule=settings.bid.rule,
        tie_break_reward=settings.bid.tie_break_reward,
        sigma_pct=settings.error.sigma_pct,
        sigma_abs=settings.error.sigma_abs,
        mae_pct=settings.error.mae_pct,
        mae_abs=settings.error.mae_abs,
        sigma=settings.error.sigma_value(float(lam_star_e.mean())),
        rho=settings.error.rho,
        bias_pct=settings.error.bias_pct,
        omega=settings.error.omega,
        sigma_w=settings.error.sigma_w,
        **w_cols,
        psi=settings.realtime.psi,
        n_ess=n_ess,
        draw=draw,
        seed=settings.seed(draw),
        dSW=dsw_val,
        dSW_pct=metrics.dsw_pct(dsw_val, bench.V),
        undeliverable_MWh=out.undeliverable_mwh,
        shed_MWh=dT * float(out.shed_rt.sum()),
        price_dist=metrics.price_distortion(da.lam_cd, lam_star_e),
        ess_profit=sum(r.profit for r in pr),  # all batteries, welfare.ess_profit
        ess_energy_payments=sum(r.energy_payments for r in pr),
        ess_imbalance_penalty=sum(r.imbalance_penalty for r in pr),
        ess_degradation=sum(r.degradation for r in pr),
        ess_terminal_value=sum(r.terminal_value for r in pr),
        ess_deviation_MWh=sum(r.deviation_mwh for r in pr),
        ess_profit_da=ess_profit_da,  # day-ahead settlement only, ignores real time
        SW_real=w.SW_real,
    )


def run_setting(case: dict, bench: Benchmark, settings: RunSettings) -> pd.DataFrame:
    rows = [run_draw(case, bench, settings, draw) for draw in range(settings.draws)]
    return pd.DataFrame(rows)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def run_all(
    case: dict,
    bench: Benchmark,
    settings_list: list[RunSettings],
    out_dir: Path,
    stage_name: str,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Runs every setting in `settings_list`, writes `<out_dir>/<setting_id>.csv` and
    one `<out_dir>/meta.json` for the stage. Skips a setting whose CSV already exists
    unless `overwrite=True`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for settings in settings_list:
        path = out_dir / f"{settings.setting_id}.csv"
        if path.exists() and not overwrite:
            frames.append(pd.read_csv(path))
            continue
        df = run_setting(case, bench, settings)
        df.to_csv(path, index=False)
        frames.append(df)

    meta = dict(
        stage=stage_name,
        date=datetime.datetime.now().isoformat(timespec="seconds"),
        git_commit=_git_commit(),
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        SW_star=bench.SW_star,
        SW_noESS=bench.SW_noESS,
        V=bench.V,
        lam_star=bench.lam_star.tolist(),
        n_settings=len(settings_list),
    )
    meta_path = out_dir / "meta.json"
    if not meta_path.exists() or overwrite:
        meta_path.write_text(json.dumps(meta, indent=2))

    return pd.concat(frames, ignore_index=True)
