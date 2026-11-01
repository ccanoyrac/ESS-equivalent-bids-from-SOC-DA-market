"""Model 2 (GUIDE.md section 5): the storage's own value function, by backward dynamic
programming on a discretised SOC grid, against a *given* price path (the forecast
`hat_lambda_t`, not a re-solved market at every grid point -- section 5 is explicit that
this needs no LP solver at all).

At each (t, grid point s_i) the reachable next-SOC values are a contiguous interval
(pure charge moves s up, pure discharge moves it down); GUIDE.md section 4's own
arbitrage argument (K2)-(K3) is why only pure charge or pure discharge is ever optimal
in one step, never both -- so checking "which single grid point is best to move to" is
exactly the right search, and it's evaluated for every candidate pair of grid points at
once via NumPy broadcasting.

Convergence note (see `tests/test_equivalence.py`'s
`test_conventional_clearing_converges_to_baldick`): against a continuous reference LP,
the discretised value function's error shrinks towards 0 as `delta` shrinks but *not*
monotonically -- whichever grid points happen to bracket the true continuous optimum
most tightly at a given `delta` matters more than `delta` alone. Don't assume a finer
grid is strictly better run to run; check the achieved gap, not just the setting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.baldick import StorageSpec


@dataclass(frozen=True)
class ValueFunction:
    soc_grid: np.ndarray  # (N+1,) MWh
    V: np.ndarray  # (T+1, N+1) EUR, V[T] is the terminal value b(soc_grid)
    v: np.ndarray  # (T, N+1) EUR/MWh, v[t, i] = (V[t, i+1]-V[t,i])/delta for i<N, NaN at i=N
    policy: np.ndarray  # (T, N+1) int, argmax next-grid-index for each (t, i)
    delta: float


def solve_value_function(
    storage: StorageSpec,
    price_forecast: np.ndarray,
    delta: float,
    terminal_slope: float | None = None,
) -> ValueFunction:
    """`price_forecast` is `hat_lambda_t`, one value per period (see `forecast_price`
    below for the recommended source). `terminal_slope` overrides `storage.b_slope` for
    sensitivity sweeps (GUIDE.md section 10) without rebuilding the whole StorageSpec.
    """
    T = len(price_forecast)
    slope = storage.b_slope if terminal_slope is None else terminal_slope

    n_steps = int(round((storage.s_max - storage.s_min) / delta))
    if n_steps < 1:
        raise ValueError("delta too large for the storage's SOC range")
    grid = storage.s_min + delta * np.arange(n_steps + 1)
    grid[-1] = storage.s_max  # guard against float round-off from arange

    n = len(grid)
    V = np.zeros((T + 1, n))
    V[T] = slope * grid
    v = np.full((T, n), np.nan)
    policy = np.full((T, n), -1, dtype=int)

    s_i = grid[:, None]  # (n, 1)
    s_j = grid[None, :]  # (1, n)
    d_soc = s_j - s_i  # (n, n): column j = target grid point, row i = starting point

    charge_mask = d_soc >= -1e-9
    c_needed = np.where(charge_mask, np.clip(d_soc, 0.0, None) / storage.alpha, np.nan)
    feasible_charge = charge_mask & (c_needed <= storage.c_max + 1e-9)

    discharge_mask = d_soc < -1e-9
    d_needed = np.where(discharge_mask, np.clip(-d_soc, 0.0, None) * storage.beta, np.nan)
    feasible_discharge = discharge_mask & (d_needed <= storage.d_max + 1e-9)

    feasible = feasible_charge | feasible_discharge
    if not feasible.any(axis=1).all():
        raise RuntimeError("some grid points have no feasible next state -- check c_max/d_max vs. delta")

    for t in range(T - 1, -1, -1):
        lam = price_forecast[t]
        payoff = np.full((n, n), -np.inf)
        payoff = np.where(feasible_charge, -(lam + storage.epsilon) * c_needed, payoff)
        payoff = np.where(feasible_discharge, (lam - storage.zeta) * d_needed, payoff)

        total = payoff + V[t + 1][None, :]
        total = np.where(feasible, total, -np.inf)
        best_j = np.argmax(total, axis=1)
        V[t] = total[np.arange(n), best_j]
        policy[t] = best_j

    v[:, :-1] = (V[:T, 1:] - V[:T, :-1]) / delta
    # Forward difference is undefined at the top grid point (no s+delta to compare
    # against); fall back to the last valid (backward) difference there so `curves.py`
    # always has a usable price for a discharge block that starts at s_max.
    v[:, -1] = v[:, -2]

    return ValueFunction(soc_grid=grid, V=V, v=v, policy=policy, delta=delta)


def is_concave(values: np.ndarray, tol: float = 1e-6) -> bool:
    """True iff the forward differences of `values` are non-increasing -- GUIDE.md
    section 5's required check before `v_t` can be turned into a valid monotone curve.
    Report this, don't silently coerce it (section 5: "that is a finding, not a bug").
    """
    diffs = np.diff(values)
    return bool(np.all(np.diff(diffs) <= tol))


def simulate_policy(vf: ValueFunction, storage: StorageSpec, s0: float | None = None) -> np.ndarray:
    """The SOC trajectory (length T+1, index 0 = s0) obtained by greedily following
    `vf.policy` forward from `s0` (default `storage.s0`). Used to centre each period's
    curve (`curves.build_curve`) on the DP's own expected SOC rather than an arbitrary
    choice.
    """
    start = storage.s0 if s0 is None else s0
    T = vf.policy.shape[0]
    traj = np.empty(T + 1)
    traj[0] = start
    idx = int(np.argmin(np.abs(vf.soc_grid - start)))
    for t in range(T):
        idx = int(vf.policy[t, idx])
        traj[t + 1] = vf.soc_grid[idx]
    return traj
