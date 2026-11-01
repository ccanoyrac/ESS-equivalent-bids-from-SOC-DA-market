---
description: Build and run the forecast-error experiments of the ESS thesis (stages 0-3), following the existing repository layout
argument-hint: [plan | status | stage0 | stage1 | stage2 | calibration | figures | <free instruction>]
---

# /experiments — forecast-error experiments for the ESS thesis

You are helping Carlos (MSc thesis, DTU) implement the experiments of his thesis on
energy storage (ESS) bids in day-ahead electricity markets. The theory and a pilot
already exist and are verified. Your job is to turn the pilot into **clear, modular,
reproducible experiment code** that fits the repository as it is today.

Requested task for this invocation: **$ARGUMENTS**
(If empty, behave as `plan`.)

---

## 0. Always do this first (every invocation)

1. **Inspect the repository before writing anything.** List the folder tree (depth 3,
   ignoring `.git`, `__pycache__`, `results/`), read `README.md`, and open every
   module you will import from. The layout described in this file is the one Carlos
   received in `ess_equivalence_code/`; **if the actual layout on disk differs, the
   actual layout wins**: put new code where it fits the existing structure and
   naming, and say in one line what you adapted.
2. Run the existing verification (`python numerical_check.py`) once and confirm it
   still passes. If it fails, stop and report; do not build on a broken baseline.
3. For anything other than `status`, write a short plan (files to create or change,
   functions, tests) and **wait for Carlos's OK before writing code**, unless he
   wrote "go" in the arguments.

---

## 1. Context in five lines

- **Current design (CD):** each ESS submits a charge bid `u_t` and a discharge offer `c_t`
  per hour; the clearing has no state of charge (SOC).
- **Baldick's SOC-based design:** the ESS submits technical data `theta` and an end-of-day
  willingness to pay `w`; the operator (ISO) optimises the SOC. It is the welfare benchmark.
- **Proven:** bids `u = alpha*(pi - nu_up) - eps`, `c = zeta + beta*(pi + nu_lo)` built from
  any optimal dual of the ESS's own self-schedule at the true prices `lam*` reproduce
  Baldick's prices and dispatch (Theorem 1, thesis report ch. 6), also with several ESS.
- **Open (these experiments):** how much welfare is lost when the bids are built from a
  price **forecast** `lam_hat` instead of `lam*`, and which error properties drive it.
- The full specification is Chapter 9 "Experimental design" and Chapter 10 "Pilot study"
  of `thesis_report/` (if present in the repo, read `chapters/ch8_design.tex` and
  `chapters/ch9_pilot.tex`). This command file restates everything needed.

---

## 2. What already exists (reuse, do not rewrite)

| File | What to reuse | Rule |
|---|---|---|
| `verify_equivalence.py` | `baldick_case`, `build(case, "SOC" or "CD", bids)`, `duals(...)`, KKT checks | **Read-only.** Verified; never edit. |
| `verify_price_based.py` | `self_schedule(S, lam, T, dT)`, `prop5_polyhedron(...)`, `pi_range(...)`, `capture_build(...)` | **Read-only.** |
| `numerical_check.py`, `computational_burden.py` | verification and benchmark scripts | **Read-only.** |
| `pilot/pilot2.py` | `day_case`, `clear_cd`, `feasible_schedule`, `realised_welfare`, `blocks_from`, `bids_forecast`, `bids_private`, `errors` | **Reference implementation.** Keep the file as it is; move its logic into the new package and use it as a regression target. |
| `pilot/multi_ess.py` | five heterogeneous ESS (data for stage 2) | Reference. |

If a reused function has a bug or a limitation (for example single ESS only), write the
new, general version in the experiments package and leave the old one untouched.

---

## 3. Target layout (adapt to the real tree if it differs)

```
<repo root>/
├── verify_equivalence.py          (existing, read-only)
├── verify_price_based.py          (existing, read-only)
├── numerical_check.py             (existing)
├── computational_burden.py        (existing)
├── pilot/                         (existing, reference only)
├── experiments/                   NEW package: all experiment logic
│   ├── __init__.py
│   ├── config.py        dataclasses: ErrorModel, BidRule, RealTime, RunSettings; stage grids
│   ├── cases.py         test systems: day_case(), baldick_cases(), fleet_case(n_ess, share)
│   ├── benchmark.py     solve Baldick once: lam*, schedules, SW*, SW_noESS, storage value V
│   ├── forecasts.py     error model (AR(1), common/specific, bias); empirical bootstrap stub
│   ├── bids.py          bid families B_star, B_star_p, B_fcst, B_priv; quantity rule
│   ├── market.py        day-ahead clearing of the current design (one or many ESS)
│   ├── realtime.py      SOC-feasible repair, redispatch with premium psi, shedding at VOLL
│   ├── welfare.py       realised welfare, ESS profit
│   ├── metrics.py       dSW, % of V, decomposition, undeliverable MWh, price distortion, sigma*
│   ├── runner.py        loop settings x draws, save one row per draw, resume
│   └── plots.py         thesis figures (same style as thesis_report/make_figures.py)
├── scripts/
│   ├── run_stage0_gate.py
│   ├── run_stage1.py
│   ├── run_stage2.py
│   └── make_experiment_figures.py
├── tests/
│   ├── test_forecasts.py
│   ├── test_bids.py
│   ├── test_market_realtime.py
│   └── test_regression_pilot.py
├── results/        (git-ignored) results/<stage>/<setting_id>.csv + results/<stage>/meta.json
└── figures/        (existing) experiment figures go in figures/experiments/
```

Scripts are thin: parse arguments, build settings from `config.py`, call `runner.py`, print a
summary. All logic lives in `experiments/`.

---

## 4. Model specification (implement exactly this)

Units: power MW, energy MWh, prices $/MWh, interval length `dT` in hours. Multipliers are in
$/MWh (power rows scaled by `dT`), as in the existing code. Keep the existing sign conventions
of `verify_equivalence.py`.

### 4.1 Benchmark (once per instance)
Solve Baldick's design → `lam_star[t]`, ESS schedules `p_star, g_star, s_star`, generator
schedule. Evaluate `SW_star` with the same welfare function used for the realised outcome
(4.6), with no redispatch and no shedding. Evaluate `SW_noESS` (clearing without storage,
shedding allowed). Storage value `V = SW_star - SW_noESS`.

### 4.2 Forecast error model (`forecasts.py`)
For ESS `e`, hour `t`, with `lam_bar = mean(lam_star)`:
```
lam_hat[e,t] = lam_star[t] + bias*lam_bar + sigma*lam_bar*z[e,t]
z[e,t]       = sqrt(omega)*chi[t] + sqrt(1-omega)*eta[e,t]
x[t]         = rho*x[t-1] + sqrt(1-rho^2)*w[t],  w ~ N(0,1) iid,  x[1] ~ N(0,1)
```
`chi` (common) and each `eta[e]` (specific) are independent AR(1) processes with unit
variance. `sigma` and `bias` are fractions of `lam_bar` (the config holds them in %).
Use `numpy.random.default_rng(seed)` with a seed derived deterministically from
`(base_seed, setting_id, draw)` so any single draw can be reproduced alone.
Add a stub `bootstrap_days(residuals, n, rng)` for the empirical errors of stage 3
(resample whole historical days); leave it raising `NotImplementedError` with a clear
message until data are provided.

### 4.3 Bid families (`bids.py`)
| Family | Prices used | Quantity plan `(p°, g°)` | Role |
|---|---|---|---|
| `B_star`   | `lam_star` | benchmark schedule | must give dSW = 0 (gate) |
| `B_star_p` | `lam_star` | ESS's own self-schedule at `lam_star` | quantity-information loss |
| `B_fcst`   | `lam_hat[e]` | own self-schedule at `lam_hat[e]` | main experiment |
| `B_priv`   | none | full capacity | lower bound, technical data only |

- Bid prices from multipliers: `u = alpha*(pi - nu_up) - eps`, `c = zeta + beta*(pi + nu_lo)`.
- Multipliers come from the polyhedron of `verify_price_based.prop5_polyhedron`, selected by:
  - `"prelimit"`: sequentially for t = 1..T, `pi[t] = clip(pi[t-1]/gamma, lo_t, hi_t)` where
    `[lo_t, hi_t]` is the admissible range given the values already fixed (first hour: `hi_1`);
  - `"solver"`: the vertex minimising `sum(pi)`.
  Reproduce `pilot2.bids_forecast` exactly.
- `B_priv`: `pi[t] = gamma^(T-t) * w`, no rents: `u = alpha*pi - eps`, `c = zeta + beta*pi`,
  one block of full capacity each.
- **Quantity rule** (all families except `B_priv`), with `delta = 0.01` $/MWh:
  ```
  charge blocks:    (u + delta, p°),  (u - delta, p_max - p°)
  discharge blocks: (c - delta, g°),  (c + delta, g_max - g°)
  ```

### 4.4 Day-ahead clearing (`market.py`)
Clear the current design with the **true** system data and the submitted blocks of all ESS.
Load shedding at VOLL = 1000 $/MWh keeps it feasible. Single bus: hour-by-hour LP (as
`pilot2.clear_cd`) when there are no ramps; with ramps or a network, use one multi-hour LP
(reuse `verify_equivalence.build(case, "CD", ...)` when it supports the blocks; otherwise write
the general version here). Return per-ESS `p_da, g_da`, generator schedule `q_da`, prices `lam_cd`.

### 4.5 Real time (`realtime.py`)
1. Each ESS delivers the SOC-feasible schedule closest to its day-ahead schedule: minimise
   `sum |p - p_da| + |g - g_da|` subject to its SOC dynamics, SOC limits and power limits
   (as `pilot2.feasible_schedule`). Undeliverable energy = `dT * sum(|p - p_da| + |g - g_da|)`.
2. Generators move from `q_da` to cover the difference, paying their offer price plus a
   premium `psi` per MWh moved up **or** down; any shortage is shed at VOLL.

### 4.6 Welfare (`welfare.py`)
```
SW_real = -( generation cost + psi * redispatched MWh + VOLL * shed MWh
             + ESS degradation (eps*p + zeta*g)*dT  -  sum_e w_e * s_e[T] )
```
Always with the true ESS parameters, never the bids. Also compute each ESS's profit
(day-ahead settlement at `lam_cd` minus real-time adjustment costs it causes, if defined;
otherwise report day-ahead profit and flag the assumption).

### 4.7 Metrics (`metrics.py`)
- `dSW = SW_star - SW_real` (must be >= -1e-6; assert it), `dSW_pct = 100*dSW/V`.
- Decomposition (accounting split, state it as such):
  `quantity part = dSW(B_star_p)`, `price part = dSW(B_fcst) - dSW(B_star_p)`.
- Undeliverable MWh, price distortion `mean |lam_cd - lam_star|`, ESS profit vs benchmark.
- Loss by hour type: split the cost difference by hour, then group hours by whether an SOC
  limit binds in the benchmark.
- `sigma_star(psi, rho)`: error size where mean loss of `B_fcst` equals that of `B_priv`
  (linear interpolation on the grid; `None` if no crossing).
- Summaries per setting: mean, 95% CI of the mean, median, p95.

### 4.8 Results format (`runner.py`)
One CSV row per draw with: `stage, setting_id, family, rule, sigma_pct, rho, bias_pct, omega,
psi, n_ess, draw, seed, dSW, dSW_pct, undeliverable_MWh, price_dist, ess_profit, SW_real`.
Write `meta.json` per stage with the config, git commit hash (if any), date, package versions
and the benchmark values (`SW_star`, `SW_noESS`, `V`, `lam_star`). Skip settings whose CSV is
already complete (resume); never overwrite existing results without asking.

---

## 5. Stages and grids (`config.py`)

| Stage | System | Grid | Draws |
|---|---|---|---|
| 0 gate | pilot day case, 1 ESS | `B_star`, `B_star_p`, `B_priv` (psi 0, 10, 50), `B_fcst` with sigma = 0 | 1 |
| 1 | pilot day case, 1 ESS | sigma % {0, 2.5, 5, 10, 20, 40} × rho {0, 0.5, 0.9} × bias % {-10, 0, +10} × rule {prelimit, solver} × psi {0, 10, 50} = 324 settings | 500 |
| 2 | day case with a fleet of 5–20 ESS (10–40 % of peak load; start from `pilot/multi_ess.py` data) | stage-1 grid reduced to rho {0, 0.9}, bias 0, plus omega {0, 0.5, 1} and fleet size | 500 |
| 3 | realistic system (to be decided with the supervisors) + calibrated / empirical errors | defined later | — |

Keep every grid value in `config.py`; no number in the code should repeat a config value.

### Stage 0 gate (must pass before any other stage runs)
- `B_star` gives `dSW` within 1e-6 of 0.
- With seed 42 and N = 150, the pilot is reproduced: `B_star_p` and `B_fcst` at sigma = 0 →
  11.1 % of V (875 $), undeliverable 46.5 MWh; `B_priv` → 74.5 % (psi 0), 175.4 % (psi 10),
  579.0 % (psi 50), undeliverable 793.6 MWh; storage value V = 7 866.6 $/day.
  Stochastic settings must match the pilot means (e.g. prelimit, rho 0, sigma 1 $/MWh →
  36.5 % ± 5.5) within their confidence interval; exact equality only if the RNG call order
  is reproduced on purpose. The pilot used absolute error sizes sigma ∈ {0, 1, 2, 5, 10, 20}
  $/MWh (= {0, 2.2, 4.4, 10.9, 21.8, 43.6} % of lam_bar = 45.9) and psi = 10, so the config
  must also accept sigma in $/MWh for this regression test.
- Forecast model: sample variance of `z` ≈ 1, lag-1 autocorrelation ≈ rho,
  cross-ESS correlation ≈ omega (tests with large samples).

### Calibration (`$ARGUMENTS = calibration`)
Prepare `experiments/calibration.py` for stage 3: given a CSV of historical day-ahead prices
(DK1/DK2, hourly), backtest a naive forecaster (same hour one week earlier) with a rolling
window, and estimate bias, sigma (% of mean price), within-day lag-1 autocorrelation rho, and
omega as the correlation between the residuals of two different forecasters. Do not download
data; ask Carlos for the file.

---

## 6. Code style: clarity first

- One responsibility per module; functions short (aim for under ~40 lines), named after
  what they compute (`closest_feasible_schedule`, not `repair2`).
- Type hints and a docstring on every public function stating inputs with **units**, output,
  and the equation or section of the thesis report it implements.
- Variable names follow the thesis symbols: `lam`, `lam_hat`, `lam_star`, `pi`, `nu_up`,
  `nu_lo`, `p`, `g`, `s`, `u`, `c`, `alpha`, `beta`, `gamma`, `eps`, `zeta`, `w`, `sigma`, `rho`,
  `bias`, `omega`, `psi`, `delta`, `dT`, `T`.
- Settings are frozen dataclasses; no global state, no hidden defaults inside functions.
- Pure functions where possible; randomness only through an `rng` argument.
- Dependencies: NumPy, SciPy (`linprog`, method `highs`, as the existing code), Matplotlib;
  pandas is allowed for results. Add anything new to `requirements.txt` and say so.
- Comments explain *why*, not *what*. No commented-out code, no prints inside library
  functions (use return values; scripts print).
- Optional parallelism (`multiprocessing`) only in `runner.py`, switchable off, with
  per-draw seeds so results do not depend on the number of workers.

---

## 7. Behaviour for each argument

- `plan`: inspect (step 0), then propose the module-by-module plan and the order of work.
- `status`: report which modules, tests and results exist, what passes, and what is next.
- `stage0`: implement what the gate needs (`config`, `cases`, `benchmark`, `forecasts`, `bids`,
  `market`, `realtime`, `welfare`, `metrics`, minimal `runner`) and the tests; run the gate.
- `stage1` / `stage2`: only if the gate passes. First run a smoke test (2 settings × 5 draws)
  and report timing per draw and the estimated total time; run the full grid only after Carlos
  confirms.
- `figures`: produce the thesis figures from saved results (loss vs sigma per rho and rule,
  undeliverable energy, decomposition bars, sigma* vs psi), in PDF, same style as
  `thesis_report/make_figures.py`.
- Anything else: treat it as a free instruction within the rules of this file.

---

## 8. Never

- Never edit `verify_equivalence.py`, `verify_price_based.py`, `numerical_check.py`,
  `computational_burden.py` or `pilot/`.
- Never invent data or results; never change a definition in section 4 silently. If the
  specification is ambiguous, ask.
- Never overwrite files in `results/` or `figures/` without asking.
- Open decisions still awaiting the supervisors' approval (error model, quantity rule,
  real-time evaluation and psi values, main indicator, stage-3 system) must stay
  **configurable**, not hard-coded.

## 9. Finish every invocation with

A short report: files created or changed, tests run and their result, numbers that matter
(for example the gate values next to the pilot values), and the next step.
