"""GUIDE.md section 7: AR(1) forecast-error model and scenario generation for wind and
PV, with the two knobs GUIDE.md specifies -- error-magnitude scale `k` and mean bias
`delta`. Deliberately NOT a price forecaster: only wind/PV get an error model. Load is
used as the deterministic `load_RT` series throughout (see `data.py`'s docstring and
GUIDE.md's "Data caveats" -- `load_DA`/`load_RT` correlate at 1.000, so treating the gap
between them as uncertainty would be modelling a fixed scaling factor, not a forecast
error).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AR1Params:
    rho: float  # autocorrelation, e_t = rho * e_{t-1} + eta_t
    sigma: float  # std dev of the innovation eta_t


def fit_ar1(error: np.ndarray) -> AR1Params:
    """OLS fit of `e_t = rho * e_{t-1} + eta_t` to an hourly error series (typically
    `RT - DA` computed over a long window, e.g. the full year). GUIDE.md quotes
    rho ~ 0.87 for wind, ~0.73 for PV as sanity references.
    """
    e_prev = error[:-1]
    e_next = error[1:]
    rho = float(np.dot(e_prev, e_next) / np.dot(e_prev, e_prev))
    residual = e_next - rho * e_prev
    sigma = float(np.std(residual, ddof=1))
    return AR1Params(rho=rho, sigma=sigma)


def generate_scenarios(
    forecast: np.ndarray,
    ar1: AR1Params,
    k: float,
    delta: float,
    n_scenarios: int,
    capacity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """`n_scenarios` realised paths around `forecast` (the DA series for one day),
    built by simulating the AR(1) error process and adding it in:
    `realised_t = clip(forecast_t + k * e_t + delta, 0, capacity)`.

    `k = 0` returns `forecast` exactly, repeated for every scenario -- GUIDE.md
    section 7.5's perfect-information case, and the regression test in section 9.2.
    Autocorrelation (not independent per-hour noise) is preserved deliberately: GUIDE.md
    section 7.2 warns that independent hourly errors let the battery average them away
    across the day and understate the welfare gap this whole thesis is trying to measure.
    """
    T = len(forecast)
    if k == 0.0:
        return np.tile(forecast, (n_scenarios, 1))

    eta = rng.normal(loc=0.0, scale=ar1.sigma, size=(n_scenarios, T))
    e = np.zeros((n_scenarios, T))
    e[:, 0] = eta[:, 0]
    for t in range(1, T):
        e[:, t] = ar1.rho * e[:, t - 1] + eta[:, t]

    realised = forecast[None, :] + k * e + delta
    return np.clip(realised, 0.0, capacity)


def error_series(da: np.ndarray, rt: np.ndarray) -> np.ndarray:
    """`RT - DA`, the raw hourly forecast error to fit `fit_ar1` on."""
    return rt - da
