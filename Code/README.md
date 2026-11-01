# ESS bidding: Baldick's SOC design vs. conventional curves, under forecast error

Working framework for `../experiments.md`'s stage-0/1 forecast-error experiments (read
that first). Builds on the already-validated equivalence proof
(`src/verify_equivalence.py`, `src/verify_price_based.py`) to measure how much welfare
is lost when a storage owner's bid curve is built from a price **forecast** instead of
the true clearing price -- and what drives the gap.

The earlier GUIDE.md-based framework (real RTS-GMLC data, a different and now-superseded
research direction) has been moved to `../Archive/guide_md_framework/`.

## Layout

```
Code/
├── src/
│   ├── verify_equivalence.py     # the SOC/CD equivalence proof's LP machinery -- READ-ONLY
│   ├── verify_price_based.py     # self-schedule, Prop. 5 polyhedron, pi_range -- READ-ONLY
│   ├── numerical_check.py        # KKT/equivalence regression check -- READ-ONLY
│   ├── computational_burden.py   # solve-time benchmarking -- READ-ONLY
│   └── experiments/              # NEW: all forecast-error experiment logic
│       ├── config.py        settings dataclasses + stage 0/1/2 grids
│       ├── cases.py         day_case() -- see "Known gaps" below
│       ├── benchmark.py     solve Baldick once: lam_star, schedules, SW_star, SW_noESS, V
│       ├── forecasts.py     AR(1) common/specific error model
│       ├── bids.py          B_star, B_star_p, B_fcst, B_priv + the quantity rule
│       ├── market.py        day-ahead clearing of the current design (multi-block)
│       ├── realtime.py      SOC-feasible repair + generator redispatch
│       ├── welfare.py       realised welfare, ESS profit
│       ├── metrics.py       dSW, decomposition, sigma*, summary stats
│       └── runner.py        the experiment loop: one CSV row per draw, resume-safe
├── scripts/
│   └── run_stage0_gate.py
├── tests/
│   ├── test_forecasts.py
│   ├── test_bids.py
│   └── test_market_realtime.py
├── notebooks/
│   ├── verify_equivalence.ipynb     # the equivalence proof, numerically checked
│   └── stage0_experiments.ipynb     # stage 0: procedure + results, start here
└── results/stage0/                  # one CSV per setting + meta.json (git-ignored)
```

## Solver

`experiments.md` asks for SciPy/HiGHS, no Gurobi. `Code/.venv` can't import `scipy`
(Windows Application Control blocks its compiled extensions -- confirmed, not just
slow), but the `DTU_SYS` conda environment already has a working `scipy`/HiGHS plus
pandas/matplotlib/jupyter, so every script, test and notebook here runs under that
interpreter:

```
"C:\Users\<you>\anaconda3\envs\DTU_SYS\python.exe" -m pytest tests/
"C:\Users\<you>\anaconda3\envs\DTU_SYS\python.exe" scripts/run_stage0_gate.py
```

## Known gaps -- read before trusting a published-looking number

`pilot/pilot2.py` and `pilot/multi_ess.py` -- the reference implementation
`experiments.md` treats as existing (it defines the actual pilot `day_case()` and
produced the published stage-0 target numbers: 11.1% of V = \$875, V = \$7,866.6/day,
the `B_priv` figures at 74.5/175.4/579.0%, 793.6 MWh undeliverable) -- **are not present
in this repository**, confirmed by search. Resolved with the user directly: `day_case()`
reuses `verify_equivalence.baldick_case` (Baldick 2026 Sec. 4.7's own validated
single-bus system) instead, and stage 0 checks the gate's *structural* properties
(`B_star` gives dSW≈0, no simultaneous charge/discharge, dSW never negative, the
forecast model's own sampling statistics) rather than matching those specific numbers.
`fleet_case()` (stage 2) raises `NotImplementedError` for the same reason -- it needs
`pilot/multi_ess.py`'s heterogeneous-ESS data, which isn't present either.

## Status / what's validated

Run `pytest tests/` under `DTU_SYS` (19 tests, all passing):

- `B_star` reproduces Baldick's own dispatch exactly through the full pipeline (bid
  construction -> day-ahead clearing -> real-time repair -> welfare) and gives
  `dSW = 0` to floating-point precision -- the stage-0 gate's hardest requirement.
- `dSW >= -1e-6` for every family/setting tested (no design beats its own benchmark).
- No simultaneous charge and discharge, any family.
- `B_fcst` at zero forecast error reproduces `B_star_p` exactly (the `k=0`
  perfect-information regression test).
- The forecast model's sampling statistics (unit variance, lag-1 autocorrelation
  `~rho`, cross-ESS correlation `~omega`) match specification under large samples, and
  the forecast never falls below `ErrorModel.price_floor` (default 0 $/MWh).
- The self-schedule's throughput tie-breaker (`BidRule.tie_break_reward`) only chooses among
  plans that are optimal for the true profit, and at `lambda*` it picks a more active plan
  than the LP solver's own vertex.
- Per-draw seeds are stable across sessions and shared by every setting (common random numbers),
  and `metrics.paired_difference` refuses to compare settings whose draws used different seeds.

Run `python scripts/run_stage0_gate.py` for the full stage-0 run (16 settings, 1,506
draws, a few seconds); see `notebooks/stage0_experiments.ipynb` for the same thing with
every step shown and discussed.

## Known findings (not bugs -- see the notebook for the full derivation of each)

1. **`day_case`'s storage value `V` is reliability-dominated.** Without the battery,
   hour 5 has a genuine 75 MW supply shortfall (thermal 100 MW cap + 25 MW renewable vs.
   200 MW demand), shed at VOLL. `V = SW_star - SW_noESS` is therefore large and mostly
   reflects avoided shedding, not pure price arbitrage -- worth a second, non-reliability-
   constrained case before generalising any `dSW_pct` figure.
2. **`B_priv` at `psi=0` gives exactly `dSW=0`**, despite 1,051 MWh of real-time
   correction. Both the repaired and the benchmark schedule are forced through the same
   binding SOC bound (`s=800=s_max` at hour 4); with no self-discharge (`gamma=1`),
   terminal SOC and total degradation cost depend only on aggregate charge/discharge,
   not timing -- so once both are pinned to the same bound they land on the same totals.
   A system with real capacity scarcity at the hours a bad bid shifts dispatch to would
   show a genuine `psi=0` loss; this result is specific to this tightly-bound toy case.
3. **Negative price forecasts caused the heavy-tailed, flat stage-0 result; the price floor
   removes it.** Without a floor, about 1 draw in 5 shed 75 MW at VOLL at every `sigma` (mean
   loss 25-32% of `V`, flat in `sigma`): in the zero-price intervals (lambda* = 0, curtailed
   renewables) a negative forecast made the battery's charge bid negative, so it bought nothing
   and reached the 16-20h shortfall empty. With `lambda_hat >= 0` (`ErrorModel.price_floor = 0`,
   now the default) load shedding becomes rare (none in the current stage-0 run); the mean loss is
   5.8, 5.8, 6.2, 7.1 and 7.9% of `V` at `sigma` = 1, 2, 5, 10, 20 \$/MWh, and the draw-by-draw
   (paired) comparison resolves every step from `sigma` = 2 upwards. Losses still take a few discrete values because of near-ties in this small case:
   perturbing block marginal costs (`experiments.md` section 7) or a larger stage-1 system is still
   advised. Why the error values are what they are: `docs/model_structure.pdf`, Section 6.
4. **At `lambda*` the battery is indifferent between many plans; the throughput tie-breaker picks
   the most active one.** The self-schedule objective carries `-kappa*dT*sum(p+g)` with
   `kappa = BidRule.tie_break_reward = 1e-3` $/MWh, reduced until the plan is still optimal for the
   true profit; it never enters welfare (`bids.self_schedule`, `docs/model_structure.pdf` M3). The
   zero-error loss (`B_star_p`) drops from 6.32% to 1.30% of `V`. It changed the plan in 0 of the
   750 forecast draws, so `sigma >= 1` results are unaffected: a forecast error breaks the ties by
   itself, and the jump from 1.3% (`sigma` = 0) to about 6% (`sigma` = 1) is the cost of
   near-indifference. The remaining 1.30% is a timing tie (which free interval to charge in) that
   throughput cannot break.

5. **Common random numbers.** `RunSettings.seed` is `SeedSequence(base_seed, draw)` (it was
   `hash((base_seed, setting_id, draw))`, which changed every session and gave every setting its
   own error shapes). Draw `d` now has the same error shape in every setting, so settings are
   compared draw by draw (`metrics.paired_difference`): paired confidence intervals are 2.5-3x
   narrower, and in 99 of 150 stage-0 draws the loss is the same at every `sigma` -- the error's
   shape, not its size, decides the battery's plan.

`results/stage0/` is the current run (floor 0, tie-breaker on, common random numbers; the
notebook reproduces it in any session); the earlier run (no floor, no tie-breaker) is kept in
`results/stage0_pre_tiebreak/`.

## Next step

Per `experiments.md` section 7: stage 1 (the 324-setting grid) only runs after this
gate is reviewed and a smoke test (2 settings x 5 draws, timed) is confirmed -- not
automatically. `config.py` already defines the full stage-1/2 grids.
