"""Forecast-error model -- experiments.md section 4.2.

    lam_hat[e,t] = max(price_floor, lam_star[t] + bias_pct/100*lam_bar + sigma*z[e,t])
    z[e,t]       = sqrt(omega)*chi[t] + sqrt(1-omega)*eta[e,t]
    x[t]         = rho*x[t-1] + sqrt(1-rho^2)*w[t],  w ~ N(0,1) iid,  x[1] ~ N(0,1)
    sigma        = sqrt(pi/2)*MAE    when the error size is given as an MAE (config.ErrorModel)

The floor (default 0, see config.ErrorModel) censors the error from below: in an interval with
lam_star = 0 the forecast error is max(0, sigma*z), never negative.

`chi` (common across ESS) and each `eta[e]` (ESS-specific) are independent AR(1)
processes with unit variance. Randomness only ever flows through an `rng` argument
(numpy.random.default_rng(seed)) -- no module-level RNG state.
"""

from __future__ import annotations

import numpy as np

from .config import ErrorModel


def ar1_unit_variance(T: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """One length-`T` AR(1) draw with unit marginal variance: x[0]~N(0,1),
    x[t] = rho*x[t-1] + sqrt(1-rho^2)*w[t]. `rho=0` reduces to iid N(0,1)."""
    x = np.empty(T)
    x[0] = rng.normal()
    for t in range(1, T):
        x[t] = rho * x[t - 1] + np.sqrt(1.0 - rho**2) * rng.normal()
    return x


def forecast_prices(
    lam_star: np.ndarray, error: ErrorModel, n_ess: int, T: int, rng: np.random.Generator
) -> np.ndarray:
    """`lam_hat`, shape (n_ess, T) $/MWh -- experiments.md section 4.2. `sigma=0` (the
    default `ErrorModel()`) returns `lam_star` broadcast to every ESS exactly, the
    deterministic/perfect-information case."""
    lam_bar = float(lam_star.mean())
    sigma = error.sigma_value(lam_bar)
    bias = error.bias_pct / 100.0 * lam_bar

    out = np.empty((n_ess, T))
    if sigma == 0.0:
        out[:] = lam_star + bias
    else:
        chi = ar1_unit_variance(T, error.rho, rng)
        for e in range(n_ess):
            eta = ar1_unit_variance(T, error.rho, rng)
            z = np.sqrt(error.omega) * chi + np.sqrt(1.0 - error.omega) * eta
            out[e] = lam_star + bias + sigma * z
    if error.price_floor is not None:
        out = np.maximum(out, error.price_floor)
    return out


def terminal_values(w: np.ndarray, error: ErrorModel, rng: np.random.Generator) -> np.ndarray:
    """`w_hat`, shape (n_ess,) $/MWh: each owner's estimate of its own terminal value,
    w_hat[e] = max(price_floor, w[e] + sigma_w*z_w[e]), z_w ~ N(0,1) iid across batteries.
    z_w is drawn even when sigma_w = 0, so the stream is the same for every setting."""
    w = np.asarray(w, dtype=float)
    z_w = rng.normal(size=len(w))
    out = w + error.sigma_w * z_w
    if error.price_floor is not None:
        out = np.maximum(out, error.price_floor)
    return out


def bootstrap_days(residuals: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Stage-3 empirical error model: resample whole historical days from `residuals`.
    Left unimplemented until real day-ahead price data is supplied (experiments.md
    section 4.2 point 5 / section "Calibration") -- do not invent a data source."""
    raise NotImplementedError(
        "bootstrap_days needs a CSV of historical day-ahead prices (DK1/DK2, hourly) to "
        "resample from. None is present in this repository -- see experiments/calibration.py."
    )
