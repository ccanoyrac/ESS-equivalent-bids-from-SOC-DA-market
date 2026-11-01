# Implementation brief — ESS bidding: Baldick SOC model vs. conventional curves

Paste this into Claude Code as the opening message, or save it as `CLAUDE.md` in the
repo root so it is picked up automatically.

---

## Context

Thesis project. We compare two electricity market designs for how an energy storage
system (ESS) participates:

- **Baldick design**: the owner submits SOC parameters and an end-of-horizon value
  function; the ISO optimises with the state of charge inside the clearing problem.
- **Conventional design**: the owner submits only a price–quantity curve per period,
  built from its own forecast; the ISO clears without (or with limited) SOC knowledge.

Under perfect foresight these are **equivalent**. Under forecast uncertainty a welfare
gap opens. The deliverable is that gap, as a function of forecast quality and renewable
penetration.

**This is research code for a thesis.** Correctness and traceability matter far more than
performance or generality. No premature abstraction. No frameworks. Plain functions.

---

## Working agreement — read this before writing code

1. **Build in the milestone order in §6. Do not skip ahead.** Each milestone has an
   acceptance test. Stop and report when a milestone's test passes; do not start the next
   one in the same run.
2. **Milestone 3 (perfect-foresight equivalence) is a hard gate.** Nothing downstream is
   meaningful until it passes to numerical tolerance. If it fails, debug it — do not
   loosen the tolerance, do not add a fudge factor, and do not proceed.
3. When a modelling choice is ambiguous, **stop and ask** rather than picking one
   silently. Ambiguities in the physics or the market design are thesis decisions, not
   implementation details.
4. Every module gets a docstring stating the equations it implements, using the notation
   in §3.
5. Assert aggressively: units, signs, bounds, energy balance. Failing loudly beats a
   plausible wrong number.

---

## 1. Data (already in `data/`)

### `units.csv`
Thermal fleet, RTS-GMLC Area 1, aggregated to one bus.
Columns: `Block, PMax_MW, MC_eur_MWh, n_units`.

| Block | PMax (MW) | MC (€/MWh) |
|---|---|---|
| NUCLEAR_Nuclear | 400 | 8.10 |
| STEAM_Coal | 1119 | 25.58 |
| CC_NG | 710 | 28.14 |
| CT_NG | 385 | 50.35 |
| CT_Oil | 80 | 143.61 |
| STEAM_Oil | 24 | 179.46 |

### `timeseries_hourly_2020.csv`
8784 rows (hourly, 2020). Columns:
`Year, Month, Day, Period, load_DA, load_RT, wind_DA, wind_RT, pv_DA, pv_RT`
`Period` is hour of day, 1–24. Capacities: wind 713.5 MW, PV 404 MW. Load peak 2850 MW.

### Data facts that constrain the implementation

- **`load_DA` and `load_RT` correlate at 1.000**, with `load_RT ≈ 0.92 × load_DA`. This is
  a scaling factor, not a forecast error. **Treat load as deterministic.** Use `load_RT`
  throughout and ignore `load_DA`.
- **Wind is the uncertainty source**: error std ≈184 MW (≈26% of capacity), AR(1) ρ ≈ 0.87.
- **PV secondary**: error std ≈7.6% of capacity, AR(1) ρ ≈ 0.73.
- Baseline VRE share is 26.9% of load energy (wind 18.8%, PV 8.1%).

---

## 2. Stack

Python 3.11+, `numpy`, `pandas`, `cvxpy` (or `pyomo`) with **HiGHS**, `matplotlib`,
`pytest`. Do not add Gurobi. Do not add a config framework — a plain dataclass is enough.

---

## 3. Notation (use these exact names in code)

| Symbol | Code name | Meaning |
|---|---|---|
| $t$ | `t` | period index, 24 per day |
| $\lambda_t$ | `lam[t]` | energy price, dual of balance |
| $\mu_t$ | `mu[t]` | dual of SOC dynamics = marginal value of stored energy |
| $s_t$ | `s[t]` | state of charge, MWh |
| $c_t, d_t$ | `c[t]`, `d[t]` | charge / discharge power, MW |
| $\alpha, \beta$ | `alpha`, `beta` | charge / discharge efficiency |
| $\varepsilon, \zeta$ | `eps`, `zeta` | charge / discharge marginal cost, €/MWh |
| $\bar S, \bar C, \bar D$ | `S_max`, `C_max`, `D_max` | energy and power limits |
| $b(s_T)$ | `terminal_value` | end-of-horizon value function |
| $v_t(s)$ | `v[t, i]` | marginal value of stored energy, on SOC grid |
| $\omega$ | `omega` | scenario index |
| $m$ | `m` | VRE capacity multiplier |
| $k$ | `k` | forecast error scale |

Battery defaults: `C_max = D_max = 100` MW, `S_max = 400` MWh, `alpha = beta = 0.95`,
`eps = zeta = 0.5` €/MWh, `s_0 = 0.5 * S_max`.

---

## 4. The models

### 4.1 Baldick benchmark (`baldick.py`)

$$\max \;\; \sum_t \Big[ U_t(q_t) - C_t(g_t) - \zeta d_t - \varepsilon c_t \Big] + b(s_T)$$

subject to

$$\textstyle\sum_j g_{j,t} + r_t - u_t + d_t - c_t = q_t \qquad (\lambda_t)$$
$$s_t = s_{t-1} + \alpha c_t - d_t/\beta \qquad (\mu_t)$$
$$0 \le s_t \le \bar S,\quad 0 \le c_t \le \bar C,\quad 0 \le d_t \le \bar D$$
$$0 \le g_{j,t} \le \bar G_j,\quad 0 \le u_t \le r_t \ \ (\text{curtailment})$$

where $r_t$ is available VRE and $u_t$ is curtailment. **Curtailment must be an explicit
variable** or the balance constraint goes infeasible at high penetration.

Solve once per realised scenario. Returns `SW`, `lam`, `mu`, `s`, `c`, `d`, dispatch.

### 4.2 Owner's dynamic program (`dp.py`)

$$V_T(s) = b(s), \qquad
V_t(s) = \max_{c,d} \Big\{ \hat\lambda_t (d-c) - \zeta d - \varepsilon c
+ V_{t+1}\big(s + \alpha c - d/\beta\big) \Big\}$$

$$v_t(s) = \frac{V_t(s+\Delta) - V_t(s)}{\Delta}$$

SOC grid step `delta = 10` MWh (40 points). Pure NumPy, no solver. Interpolate
$V_{t+1}$ linearly off-grid.

$\hat\lambda_t$ is the **forecast** price — obtained by running `clearing.py` with no
battery against the forecast scenario.

**Assert that $V_t$ is concave** (i.e. `v[t]` non-increasing in `s`) at every `t`. If it
is not, raise with a clear message — that is a research finding, not a bug to silence.

### 4.3 Curve construction (`curves.py`)

$$\pi^{\text{offer}}_t(s) = \frac{v_t(s)}{\beta} + \zeta, \qquad
\pi^{\text{bid}}_t(s) = \alpha\, v_t(s) - \varepsilon$$

Stack blocks from the expected SOC: downward for discharge, upward for charge. Each block
is `delta` MWh of storage; delivered energy per discharge block is `delta * beta`, energy
bought per charge block is `delta / alpha`. Return, per period, two monotone step
functions as `(price, quantity)` arrays.

Assert offer prices non-decreasing in quantity and bid prices non-increasing.

### 4.4 Conventional clearing (`clearing.py`)

Same LP as 4.1 with the battery removed as a state-constrained resource and inserted as
ordinary supply blocks (the offer curve) and demand blocks (the bid curve).

Two variants behind a flag:
- `soc_enforced=False` — European / OMIE style, no SOC in the clearing.
- `soc_enforced=True` — CAISO style, SOC dynamics kept, but the willingness to trade
  comes from the submitted curve rather than from an internal value function.

Default `False` until told otherwise. **Do not guess which one the thesis needs — build
both, make the flag explicit.**

### 4.5 Scenarios (`errors.py`)

1. Fit AR(1) to $e_t = \text{wind\_RT}_t - \text{wind\_DA}_t$ and the PV equivalent.
2. Generate error paths $e_t = \rho e_{t-1} + \eta_t$, preserving $\rho$.
   **Autocorrelation is essential** — independent hourly errors let the battery average
   them away and will understate the result.
3. Clip to physical bounds: `0 <= wind <= 713.5 * m`, PV zero where `pv_DA == 0`.
4. Knobs: error scale `k`, bias `delta_bias`, VRE multiplier `m` (separate multipliers for
   wind and PV).
5. `k = 0` must give the deterministic perfect-information case exactly.

### 4.6 VRE scaling (`scenarios.py`)

Multiply wind and PV by `m_wind`, `m_pv`. Absolute forecast error scales with capacity.
Provide a `smoothing_gamma` parameter so the error scales as $m^\gamma$ with
$\gamma \le 1$, defaulting to `1.0` (conservative, no geographic smoothing).

When `m > 1.5`, retire thermal: drop `STEAM_Coal` and `STEAM_Oil` via a
`retire_blocks` list argument. Do not hardcode the retirement schedule.

---

## 5. Repo layout

```
ess-thesis/
├── data/
│   ├── units.csv
│   └── timeseries_hourly_2020.csv
├── src/
│   ├── config.py        # dataclass: battery params, grid step, tolerances
│   ├── data.py          # load csvs, select days
│   ├── scenarios.py     # VRE scaling, thermal retirement
│   ├── errors.py        # AR(1) fit + scenario generation
│   ├── baldick.py       # benchmark LP
│   ├── dp.py            # value function
│   ├── curves.py        # v_t -> price/quantity blocks
│   ├── clearing.py      # conventional clearing, both variants
│   └── compare.py       # the experiment loop
├── tests/
│   ├── test_merit_order.py
│   ├── test_equivalence.py     # THE GATE
│   └── test_curves.py
└── results/
```

---

## 6. Milestones — build in this order, stop after each

### M1 — Data loading
`data.py`: load both CSVs, return a tidy per-day frame. Helper `select_days()` returning
three representative days: highest wind, lowest wind, highest load.
**Accept when**: the three days print with sensible summary stats.

### M2 — Merit-order clearing, no battery
`clearing.py` with no storage, deterministic, using `load_RT` and realised VRE.
**Accept when**: every cleared price equals one of the six marginal costs in `units.csv`
(or 0 under surplus). Any other price means a bug.

### M3 — Perfect-foresight equivalence — **THE GATE**
Implement `baldick.py`, `dp.py`, `curves.py`. Run with `k = 0`, `m = 1`, forecast equal to
realisation. Assert all three:

1. `abs(v[t, i_star[t]] - mu[t]) < 1e-4` along the optimal SOC trajectory
2. conventional dispatch equals Baldick dispatch within `1e-6`
3. `abs(SW_conv - SW_baldick) < 1e-6`

Also assert no simultaneous charge and discharge in any period.
**Do not proceed until all pass.** Report the numbers, not just pass/fail.

### M4 — Uncertainty
`errors.py` + the loop in `compare.py`. Sweep `k ∈ {0, 0.5, 1, 2}` with 20 scenarios.
**Accept when**: `k = 0` reproduces M3 exactly, and the gap is monotone non-decreasing
in `k`. If it is not monotone, investigate before moving on.

### M5 — Penetration sweep
`scenarios.py`. Sweep `m ∈ {1.0, 1.5, 2.0, 2.5, 3.0}` crossed with `k`.
**Accept when**: a heatmap of `ΔSW(m, k)` is produced in `results/`.

### M6 — Grid and terminal-value sensitivity
Re-run M3 and M5 with `delta ∈ {5, 10, 20}` MWh and three choices of `b(s_T)`.
**Accept when**: sensitivity tables are written to `results/`.

---

## 7. Known traps — handle these explicitly

- **Degenerate duals.** When the battery is marginal, $\mu_t$ is not unique. Perturb block
  marginal costs by a few cents (deterministically, seeded) to break ties.
- **Terminal condition drives everything.** $b(s_T)$ propagates backwards through the whole
  $\mu_t$ path. Default to linear `b(s) = mu_bar * s` with `mu_bar` near the average
  off-peak price. Never default to `b ≡ 0` — it empties the battery at `T` and contaminates
  every result.
- **Efficiency bookkeeping.** $\alpha$ multiplies energy *into* storage; $1/\beta$ divides
  energy *out*. Add a round-trip test: charging then discharging must lose
  $1 - \alpha\beta \approx 10\%$.
- **Units.** MW and MWh are numerically equal at hourly resolution, which hides errors.
  Suffix every array name with `_mw` or `_mwh`.
- **Welfare with inelastic demand** reduces to minus cost. Include one elastic demand block
  (price cap 300 €/MWh) so "social welfare" is meaningful, and document it.
- **Curtailment** must be an explicit variable, not a slack. At `m >= 2` there will be
  surplus hours.

---

## 8. Open question — do not resolve alone

Whether the conventional design being compared against is the European (no SOC in the
clearing) or CAISO (SOC enforced, but no submitted $b(s_T)$) one is **unresolved and is a
thesis decision**. Build both behind the `soc_enforced` flag, default `False`, and flag it
in the output. Do not pick one and bury the choice.

---

## Start here

Begin with M1 and M2 only. Report the merit-order prices for the three representative days
and stop.
