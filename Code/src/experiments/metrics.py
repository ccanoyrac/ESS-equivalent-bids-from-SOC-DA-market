"""Metrics -- experiments.md section 4.7."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def dsw(SW_star: float, SW_real: float, tol: float = 1e-6) -> float:
    """dSW = SW_star - SW_real. Must be >= -tol: the benchmark's own welfare can never
    be exceeded by a design that only knows a forecast or less (experiments.md 4.7)."""
    d = SW_star - SW_real
    assert d >= -tol, f"dSW={d} < -{tol}: SW_real exceeds the benchmark, which should be impossible"
    return d


def dsw_pct(dsw_value: float, V: float) -> float:
    return 100.0 * dsw_value / V if V != 0 else float("nan")


@dataclass
class Decomposition:
    quantity_part: float  # dSW(B_star_p): loss from using a self-schedule quantity plan
    price_part: float  # dSW(B_fcst) - dSW(B_star_p): additional loss from forecast error


def decompose(dsw_star_p: float, dsw_fcst: float) -> Decomposition:
    """Accounting split (experiments.md 4.7) -- NOT a causal decomposition, just
    `dSW(B_fcst)` partitioned into the piece already present at zero forecast error
    (the quantity-information loss of self-scheduling) and the rest (attributed to the
    forecast)."""
    return Decomposition(quantity_part=dsw_star_p, price_part=dsw_fcst - dsw_star_p)


def price_distortion(lam_cd: np.ndarray, lam_star: np.ndarray) -> float:
    return float(np.mean(np.abs(lam_cd - lam_star)))


def loss_by_soc_binding(
    hourly_cost_diff: np.ndarray, soc_binding: np.ndarray
) -> dict[str, float]:
    """Splits a per-hour cost-difference series by whether an SOC limit binds in the
    benchmark at that hour (experiments.md 4.7's "loss by hour type")."""
    soc_binding = np.asarray(soc_binding, dtype=bool)
    return dict(
        binding=float(hourly_cost_diff[soc_binding].sum()) if soc_binding.any() else 0.0,
        non_binding=float(hourly_cost_diff[~soc_binding].sum()) if (~soc_binding).any() else 0.0,
    )


def sigma_star(sigma_grid: np.ndarray, mean_loss_fcst: np.ndarray, mean_loss_priv: float) -> float | None:
    """The error size where B_fcst's mean loss first reaches B_priv's (constant) mean
    loss, by linear interpolation on `sigma_grid` -- experiments.md 4.7. `None` if the
    two series never cross on the grid provided.
    """
    diff = mean_loss_fcst - mean_loss_priv
    if np.all(diff < 0) or np.all(diff > 0):
        return None
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        return None
    i = sign_change[0]
    x0, x1 = sigma_grid[i], sigma_grid[i + 1]
    y0, y1 = diff[i], diff[i + 1]
    return float(x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0))


def summarize(values: np.ndarray) -> dict[str, float]:
    """Mean, 95% CI of the mean, median, p95 -- experiments.md 4.7's "Summaries per
    setting"."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return dict(
        mean=mean,
        ci95_lo=mean - 1.96 * sem,
        ci95_hi=mean + 1.96 * sem,
        median=float(np.median(values)),
        p95=float(np.percentile(values, 95)),
        n=n,
    )


def paired_difference(a: pd.DataFrame, b: pd.DataFrame, value_col: str) -> dict[str, float]:
    """Compare two settings run with common random numbers (config.RunSettings.seed):
    match their rows draw by draw and summarise the per-draw difference b - a. The CI is
    the one of that mean difference -- narrower than comparing the two settings' own CIs,
    because the shared error shape cancels. `share_b_ge_a`: fraction of draws where b's
    value is at least a's. Asserts both settings really used the same seed per draw."""
    m = a[["draw", "seed", value_col]].merge(b[["draw", "seed", value_col]], on="draw", suffixes=("_a", "_b"))
    assert len(m) == len(a) == len(b), "settings must have the same draws"
    assert (m["seed_a"] == m["seed_b"]).all(), "draws differ in seed: not common random numbers"
    d = (m[f"{value_col}_b"] - m[f"{value_col}_a"]).to_numpy()
    out = summarize(d)
    out["share_b_ge_a"] = float(np.mean(d >= -1e-9))
    return out


def summarize_by_setting(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols):
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row.update(summarize(g[value_col].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)
