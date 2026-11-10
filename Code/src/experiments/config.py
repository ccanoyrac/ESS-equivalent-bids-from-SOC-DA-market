"""Settings dataclasses and the stage grids of experiments.md section 5.

Every grid value used anywhere in the experiments package is defined here once; no
number is repeated in cases.py, bids.py, runner.py or the notebooks (experiments.md
section 6). Settings are frozen (immutable) dataclasses -- no hidden defaults inside
functions, no global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

# sigma of a zero-mean normal error with a given mean absolute error: E|sigma*z| = sigma*sqrt(2/pi)
MAE_TO_SIGMA = float(np.sqrt(np.pi / 2.0))


@dataclass(frozen=True)
class ErrorModel:
    """Forecast-error model, experiments.md section 4.2:

        lam_hat[e,t] = lam_star[t] + bias_pct/100*lam_bar + sigma*z[e,t]
        z[e,t]       = sqrt(omega)*chi[t] + sqrt(1-omega)*eta[e,t]

    `chi` (common across ESS) and each `eta[e]` (ESS-specific) are independent AR(1)
    processes with unit variance and autocorrelation `rho`. `sigma_pct` is a fraction
    of `lam_bar` (the mean benchmark price); `sigma_abs` overrides it with an absolute
    $/MWh value -- needed because the stage-0 gate's own pilot values are quoted in
    $/MWh (section 5: "sigma in {0,1,2,5,10,20} $/MWh"), not as a percentage.

    `price_floor` ($/MWh): the forecast is clipped from below, lam_hat = max(lam_hat, floor).
    A forecaster never predicts a price the market cannot clear at; in the test systems the
    cheapest supply is free, curtailable renewables, so lam_star >= 0 and the floor is 0.
    `None` disables the clipping (pure Gaussian errors). For a market with negative prices,
    set it to that market's minimum clearing price.

    `mae_pct` / `mae_abs`: the error size given as a mean absolute error instead of a std --
    `mae_pct` as a % of `lam_bar`, `mae_abs` in $/MWh. For a normal error e = sigma*z,
    E|e| = sigma*sqrt(2/pi), so

        sigma = sqrt(pi/2) * MAE  (~1.2533 * MAE).

    MAE is the size forecasting studies report, so it is the natural input when the grid is
    anchored on real forecasts. It is the MAE of the random part sigma*z only: the bias is
    a separate knob, and the price floor is applied afterwards (where it binds, lambda*
    near the floor, the realised error is smaller than this MAE). Give either an MAE or a
    sigma, not both.

    `sigma_w` ($/MWh): error of each owner's estimate of its own terminal value,

        w_hat[e] = max(price_floor, w[e] + sigma_w*z_w[e]),   z_w[e] ~ N(0,1) iid,

    one draw per battery and scenario, independent of the price error (forecasts.
    terminal_values). It enters the owner-built bids only (B_fcst, B_priv); the benchmark
    and the welfare evaluation keep the true w. 0 = the owner knows its w.
    """

    sigma_pct: float = 0.0
    sigma_abs: Optional[float] = None
    rho: float = 0.0
    bias_pct: float = 0.0
    omega: float = 0.0
    price_floor: Optional[float] = 0.0
    sigma_w: float = 0.0
    mae_pct: Optional[float] = None
    mae_abs: Optional[float] = None

    def __post_init__(self):
        if self.mae_pct is not None and self.mae_abs is not None:
            raise ValueError("give the MAE once: mae_pct or mae_abs, not both")
        if self.has_mae and (self.sigma_abs is not None or self.sigma_pct != 0.0):
            raise ValueError("give the error size once: an MAE (mae_pct/mae_abs) or a sigma (sigma_pct/sigma_abs)")
        if (self.mae_pct or 0.0) < 0 or (self.mae_abs or 0.0) < 0:
            raise ValueError("the MAE must be >= 0")

    @property
    def has_mae(self) -> bool:
        return self.mae_pct is not None or self.mae_abs is not None

    def mae_value(self, lam_bar: float) -> float:
        """Mean absolute error of the random part sigma*z, $/MWh. When the size is given as a
        sigma, this is the MAE that sigma implies, sigma*sqrt(2/pi)."""
        if self.mae_abs is not None:
            return self.mae_abs
        if self.mae_pct is not None:
            return self.mae_pct / 100.0 * lam_bar
        return self.sigma_value(lam_bar) / MAE_TO_SIGMA

    def sigma_value(self, lam_bar: float) -> float:
        """Absolute error std in $/MWh: sqrt(pi/2)*MAE when the size is an MAE, otherwise
        the sigma_pct/sigma_abs choice."""
        if self.has_mae:
            return MAE_TO_SIGMA * self.mae_value(lam_bar)
        if self.sigma_abs is not None:
            return self.sigma_abs
        return self.sigma_pct / 100.0 * lam_bar

    def label(self) -> str:
        if self.mae_abs is not None:
            s = f"MAE={self.mae_abs}$/MWh"
        elif self.mae_pct is not None:
            s = f"MAE={self.mae_pct}%"
        else:
            s = f"sigma={self.sigma_abs}$/MWh" if self.sigma_abs is not None else f"sigma={self.sigma_pct}%"
        return (f"{s}, rho={self.rho}, bias={self.bias_pct}%, omega={self.omega}, "
                f"floor={self.price_floor}, sigma_w={self.sigma_w}$/MWh")


BidFamily = Literal["B_star", "B_star_p", "B_fcst", "B_priv"]
PiRule = Literal["prelimit", "solver"]


@dataclass(frozen=True)
class BidRule:
    """Which bid family to build and, for the families whose price depends on a choice
    of dual multiplier (B_star_p, B_fcst), which selection rule -- experiments.md
    section 4.3. `delta` ($/MWh) is the quantity-rule's block spread.

    `tie_break_reward` (kappa, $/MWh): when the self-schedule P^SS(lam) has several
    optimal plans, pick the one with the most throughput sum(p+g) -- the self-schedule
    objective gets an extra -kappa*dT*sum(p+g) term (bids.self_schedule). It is a pure
    tie-breaker: kappa is reduced until the plan is still optimal for the true profit, so
    it never trades profit for activity, and it is not part of any welfare calculation.
    0 disables it (the LP solver's own vertex choice, as before).
    """

    family: BidFamily = "B_fcst"
    rule: PiRule = "prelimit"
    delta: float = 0.01
    tie_break_reward: float = 1e-3


@dataclass(frozen=True)
class RealTime:
    """Real-time redispatch settings, experiments.md section 4.5. `psi` ($/MWh) is the
    premium generators charge per MWh moved up OR down from their day-ahead schedule;
    `voll` ($/MWh) prices any shortage that redispatch cannot cover."""

    psi: float = 0.0
    voll: float = 1000.0


@dataclass(frozen=True)
class RunSettings:
    """One point in the experiment grid: one (error model, bid rule, real-time setting)
    combination, run for `draws` independent scenarios starting at `base_seed`.

    Common random numbers: the seed of draw `d` depends on (`base_seed`, `d`) only, not on
    the setting, so draw `d` has the same error shape z in every setting that shares
    `base_seed` (lam_hat = lam_star + bias + sigma*z: sigma, bias, rho, omega and the bid
    design change, the underlying random numbers do not). Comparisons between settings
    are then paired draw by draw (metrics.paired_difference)."""

    error: ErrorModel
    bid: BidRule
    realtime: RealTime
    n_ess: int = 1
    draws: int = 1
    base_seed: int = 42
    setting_id: str = ""

    def seed(self, draw: int) -> int:
        """Deterministic per-draw seed from (base_seed, draw), so any single draw can be
        reproduced alone, in any session (experiments.md section 4.2), and is shared by
        every setting (common random numbers, see the class docstring). numpy's
        SeedSequence, unlike Python's hash(), is not randomised per process."""
        return int(np.random.SeedSequence([self.base_seed, draw]).generate_state(1)[0])

    def seed_w(self, draw: int) -> int:
        """Seed of the terminal-value error of draw `draw`: its own stream, so z_w is the
        same whatever the price-error settings (sigma = 0 draws no price noise at all)."""
        return int(np.random.SeedSequence([self.base_seed, draw, 1]).generate_state(1)[0])


# ---------------------------------------------------------------------------
# Stage 0 -- the gate (experiments.md section 5 "Stage 0 gate")
# ---------------------------------------------------------------------------
# NOTE: pilot/pilot2.py (the reference implementation whose run produced the published
# gate numbers -- 11.1% of V = $875, V = $7,866.6/day, etc.) is not present in this
# repository (confirmed absent; see Code/README.md "Known gaps"). Stage 0 here therefore
# checks the gate's *structural* properties (B_star reproduces dSW=0, the forecast model's
# sampling statistics, no simultaneous charge/discharge, dSW monotone in the stochastic
# settings it can control) rather than matching those specific published numbers -- see
# the stage-0 notebook for the full discussion.

STAGE0_SEED = 42
STAGE0_DRAWS = 150  # matches the pilot's own N for the stochastic settings it quotes
STAGE0_PSI_FOR_PRIV = (0.0, 10.0, 50.0)
# Error size as an MAE, % of mean(lambda*) (sigma = sqrt(pi/2)*MAE, ErrorModel): 0 = perfect prices,
# 5-16% = state-of-the-art day-ahead forecasts (Lago et al. 2021), 20-40% = naive / no skill.
STAGE0_MAE_PCT = (0.0, 2.5, 5.0, 10.0, 20.0, 40.0)
STAGE0_FAMILIES_FOR_STOCHASTIC: tuple[BidFamily, ...] = ("B_star_p", "B_fcst", "B_priv")
STAGE0_PSI_FOR_STOCHASTIC = 10.0
STAGE0_RHO_FOR_STOCHASTIC = 0.0
STAGE0_RULE_FOR_STOCHASTIC: PiRule = "prelimit"


def stage0_deterministic_settings() -> list[RunSettings]:
    """B_star (must give dSW~0) and B_priv at three psi values -- deterministic, draws=1."""
    settings = [
        RunSettings(
            error=ErrorModel(),
            bid=BidRule(family="B_star"),
            realtime=RealTime(psi=0.0),
            draws=1,
            base_seed=STAGE0_SEED,
            setting_id="stage0_B_star",
        )
    ]
    for psi in STAGE0_PSI_FOR_PRIV:
        settings.append(
            RunSettings(
                error=ErrorModel(),
                bid=BidRule(family="B_priv"),
                realtime=RealTime(psi=psi),
                draws=1,
                base_seed=STAGE0_SEED,
                setting_id=f"stage0_B_priv_psi{psi:g}",
            )
        )
    return settings


def stage0_stochastic_settings() -> list[RunSettings]:
    """The MAE sweep STAGE0_MAE_PCT at psi=10, rho=0, for the three families that face the
    forecast question: B_star_p (exact prices), B_fcst (forecast prices) and B_priv (no
    forecast at all). MAE = 0 is the deterministic check. B_priv uses no forecast and
    sigma_w = 0 here, so it has no randomness: one draw per MAE is its exact value, and it
    is repeated over the grid only so the three families line up setting by setting."""
    settings = []
    for family in STAGE0_FAMILIES_FOR_STOCHASTIC:
        for mae_pct in STAGE0_MAE_PCT:
            draws = 1 if mae_pct == 0.0 or family == "B_priv" else STAGE0_DRAWS
            settings.append(
                RunSettings(
                    error=ErrorModel(mae_pct=mae_pct, rho=STAGE0_RHO_FOR_STOCHASTIC),
                    bid=BidRule(family=family, rule=STAGE0_RULE_FOR_STOCHASTIC),
                    realtime=RealTime(psi=STAGE0_PSI_FOR_STOCHASTIC),
                    draws=draws,
                    base_seed=STAGE0_SEED,
                    setting_id=f"stage0_{family}_mae{mae_pct:g}",
                )
            )
    return settings


def stage0_settings() -> list[RunSettings]:
    return stage0_deterministic_settings() + stage0_stochastic_settings()


# ---------------------------------------------------------------------------
# Stage 1 (grid defined now; running it is a separate, later invocation --
# experiments.md section 7: "only if the gate passes... run the full grid only after
# Carlos confirms")
# ---------------------------------------------------------------------------
STAGE1_SIGMA_PCT = (0.0, 2.5, 5.0, 10.0, 20.0, 40.0)
STAGE1_RHO = (0.0, 0.5, 0.9)
STAGE1_BIAS_PCT = (-10.0, 0.0, 10.0)
STAGE1_RULE: tuple[PiRule, ...] = ("prelimit", "solver")
STAGE1_PSI = (0.0, 10.0, 50.0)
STAGE1_DRAWS = 500
STAGE1_SEED = 42


def stage1_settings() -> list[RunSettings]:
    """The 6x3x3x2x3 = 324-setting grid of experiments.md section 5, family B_fcst."""
    settings = []
    for sigma_pct in STAGE1_SIGMA_PCT:
        for rho in STAGE1_RHO:
            for bias_pct in STAGE1_BIAS_PCT:
                for rule in STAGE1_RULE:
                    for psi in STAGE1_PSI:
                        sid = f"stage1_sigma{sigma_pct:g}_rho{rho:g}_bias{bias_pct:g}_{rule}_psi{psi:g}"
                        settings.append(
                            RunSettings(
                                error=ErrorModel(sigma_pct=sigma_pct, rho=rho, bias_pct=bias_pct),
                                bid=BidRule(family="B_fcst", rule=rule),
                                realtime=RealTime(psi=psi),
                                draws=STAGE1_DRAWS,
                                base_seed=STAGE1_SEED,
                                setting_id=sid,
                            )
                        )
    return settings


# ---------------------------------------------------------------------------
# Stage 2 (grid defined now; `cases.fleet_case` raises NotImplementedError until
# pilot/multi_ess.py's heterogeneous-ESS data is available -- see cases.py)
# ---------------------------------------------------------------------------
STAGE2_RHO = (0.0, 0.9)
STAGE2_BIAS_PCT = (0.0,)
STAGE2_OMEGA = (0.0, 0.5, 1.0)
STAGE2_FLEET_SIZES = (5, 10, 20)
STAGE2_SHARE_OF_PEAK = (0.10, 0.25, 0.40)
STAGE2_DRAWS = 500
STAGE2_SEED = 42
