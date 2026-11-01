"""Tests for experiments.forecasts -- experiments.md section 5's forecast-model check:
sample variance of z ~ 1, lag-1 autocorrelation ~ rho, cross-ESS correlation ~ omega.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from experiments.config import BidRule, ErrorModel, RealTime, RunSettings
from experiments.metrics import paired_difference
from experiments.forecasts import ar1_unit_variance, forecast_prices


def test_ar1_unit_variance_and_autocorrelation():
    rng = np.random.default_rng(0)
    rho, N, T = 0.7, 20000, 30
    draws = np.array([ar1_unit_variance(T, rho, rng) for _ in range(N)])
    assert draws.var(axis=0).mean() == pytest.approx(1.0, abs=0.05)
    lag1 = np.mean([np.corrcoef(draws[:, t], draws[:, t + 1])[0, 1] for t in range(T - 1)])
    assert lag1 == pytest.approx(rho, abs=0.03)


def test_cross_ess_correlation_matches_omega():
    rng = np.random.default_rng(1)
    T, omega, n = 30, 0.6, 20000
    lam_star = np.full(T, 50.0)
    err = ErrorModel(sigma_abs=10.0, rho=0.0, omega=omega)
    e0s, e1s = [], []
    for _ in range(n):
        lh = forecast_prices(lam_star, err, n_ess=2, T=T, rng=rng)
        e0s.append(lh[0] - lam_star)
        e1s.append(lh[1] - lam_star)
    e0s, e1s = np.array(e0s), np.array(e1s)
    corr = np.mean([np.corrcoef(e0s[:, t], e1s[:, t])[0, 1] for t in range(T)])
    assert corr == pytest.approx(omega, abs=0.03)


def test_sigma_zero_is_exactly_deterministic():
    """experiments.md 4.2 point 5: k=0 must give the deterministic perfect-information
    case exactly."""
    lam_star = np.array([10.0, 20.0, 30.0])
    out = forecast_prices(lam_star, ErrorModel(), n_ess=3, T=3, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(out, np.tile(lam_star, (3, 1)))


def test_forecast_never_below_price_floor():
    """lam_hat >= price_floor (default 0) even where lam_star = 0 and the error is large;
    price_floor=None restores the unclipped Gaussian forecast."""
    lam_star = np.array([0.0, 0.0, 50.0, 50.0])
    clipped = forecast_prices(lam_star, ErrorModel(sigma_abs=20.0), n_ess=1, T=4, rng=np.random.default_rng(3))
    raw = forecast_prices(lam_star, ErrorModel(sigma_abs=20.0, price_floor=None), n_ess=1, T=4,
                          rng=np.random.default_rng(3))
    assert clipped.min() >= 0.0
    np.testing.assert_array_equal(clipped, np.maximum(raw, 0.0))
    biased = forecast_prices(lam_star, ErrorModel(bias_pct=-50.0), n_ess=1, T=4, rng=np.random.default_rng(3))
    assert biased.min() >= 0.0  # the sigma = 0 path is clipped too


def test_sigma_abs_overrides_sigma_pct():
    err = ErrorModel(sigma_pct=50.0, sigma_abs=7.0)
    assert err.sigma_value(lam_bar=100.0) == 7.0
    err2 = ErrorModel(sigma_pct=10.0)
    assert err2.sigma_value(lam_bar=100.0) == pytest.approx(10.0)


def test_seed_is_stable_and_common_across_settings():
    """RunSettings.seed depends on (base_seed, draw) only: the same in every session (no
    hash() randomisation; the literal values below would fail otherwise) and the same for
    every setting -- common random numbers."""
    a = RunSettings(error=ErrorModel(sigma_abs=1.0), bid=BidRule(), realtime=RealTime(), base_seed=42, setting_id="a")
    b = RunSettings(error=ErrorModel(sigma_abs=5.0, rho=0.5), bid=BidRule(family="B_star_p"), realtime=RealTime(psi=50.0),
                    base_seed=42, setting_id="b")
    assert [a.seed(d) for d in range(3)] == [3444837047, 3329053876, 955475868]
    assert [a.seed(d) for d in range(50)] == [b.seed(d) for d in range(50)]
    assert len({a.seed(d) for d in range(1000)}) == 1000


def test_same_draw_same_error_shape_at_every_sigma():
    """With common random numbers draw d's lam_hat differs between sigma values only by the
    scale of the same z (no floor here, so nothing is clipped)."""
    lam_star = np.array([43.1, 50.0, 0.0, 0.0, 50.0, 50.0])
    seed = RunSettings(error=ErrorModel(), bid=BidRule(), realtime=RealTime()).seed(7)
    z = {}
    for sigma in (1.0, 5.0, 20.0):
        lam_hat = forecast_prices(lam_star, ErrorModel(sigma_abs=sigma, price_floor=None), 1, 6, np.random.default_rng(seed))[0]
        z[sigma] = (lam_hat - lam_star) / sigma
    np.testing.assert_allclose(z[1.0], z[5.0], atol=1e-12)
    np.testing.assert_allclose(z[1.0], z[20.0], atol=1e-12)


def test_paired_difference():
    a = pd.DataFrame(dict(draw=[0, 1, 2], seed=[5, 6, 7], loss=[1.0, 2.0, 3.0]))
    b = pd.DataFrame(dict(draw=[0, 1, 2], seed=[5, 6, 7], loss=[1.5, 2.0, 4.0]))
    out = paired_difference(a, b, "loss")
    assert out["mean"] == pytest.approx(0.5) and out["share_b_ge_a"] == 1.0
    with pytest.raises(AssertionError):
        paired_difference(a, b.assign(seed=[5, 6, 8]), "loss")


def test_terminal_value_error():
    """sigma_w = 0 gives the true w exactly; the w-error stream does not depend on the price-error
    settings (common random numbers for both errors); w_hat respects the floor."""
    from experiments.forecasts import terminal_values

    w = np.array([40.0, 55.0])
    rs = RunSettings(error=ErrorModel(), bid=BidRule(), realtime=RealTime(), n_ess=2)
    assert np.array_equal(terminal_values(w, ErrorModel(sigma_abs=5.0), np.random.default_rng(rs.seed_w(3))), w)
    a = terminal_values(w, ErrorModel(sigma_abs=0.0, sigma_w=5.0), np.random.default_rng(rs.seed_w(3)))
    b = terminal_values(w, ErrorModel(sigma_abs=20.0, omega=1.0, sigma_w=5.0), np.random.default_rng(rs.seed_w(3)))
    assert np.array_equal(a, b) and not np.array_equal(a, w)
    assert rs.seed_w(3) != rs.seed(3)
    big = terminal_values(np.full(5000, 1.0), ErrorModel(sigma_w=10.0), np.random.default_rng(0))
    assert big.min() >= 0.0
