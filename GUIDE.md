# ESS bidding under Baldick's SOC model vs. conventional curves — working guide

Companion to `units.csv` and `timeseries_hourly_2020.csv`.

---

## 1. What this is for

**Research question.** Baldick's SOC-based market model lets the ESS submit its
state-of-charge parameters and an end-of-horizon value, and the ISO optimises over them.
The conventional design accepts only a price–quantity curve per period. How must the
conventional bid/offer curves be shaped to reach the same social welfare, and how large
is the residual gap?

**The result to aim for.**

$$\Delta SW \;=\; SW_{\text{Baldick}} \;-\; \max_{B}\; \mathbb{E}_\omega\big[SW(B,\omega)\big]$$

This is the value of the SOC-based design measured against the *best possible*
conventional bid, not against a naive one. Nobody can dismiss it by saying the owner
would have bid better.

**Two facts that shape the whole thesis:**

1. Under perfect foresight the two designs are **equivalent** — the conventional curve
   that reproduces Baldick's dispatch always exists. This is a proposition to prove, and
   also your first numerical test.
2. The gap only opens under uncertainty, because the curve must be committed before
   prices are known, and because a single curve per period cannot be contingent on SOC.

---

## 2. The data

### `units.csv`

Thermal fleet of RTS-GMLC **Area 1**, aggregated to one bus, grouped by unit type and
fuel. Marginal cost is `HR_avg_0/1000 * fuel_price + VOM`, capacity-weighted within each
block.

| Block | PMax (MW) | MC (€/MWh) | units |
|---|---|---|---|
| NUCLEAR_Nuclear | 400 | 8.10 | 1 |
| STEAM_Coal | 1119 | 25.58 | 8 |
| CC_NG | 710 | 28.14 | 2 |
| CT_NG | 385 | 50.35 | 7 |
| CT_Oil | 80 | 143.61 | 4 |
| STEAM_Oil | 24 | 179.46 | 2 |

Total thermal 2718 MW. The step from ~28 to ~50 €/MWh between CC and CT gas is where
storage arbitrage will bite — check your results against it.

### `timeseries_hourly_2020.csv`

8784 rows, one per hour of 2020. Columns:

| Column | Meaning |
|---|---|
| `Year, Month, Day, Period` | `Period` = hour of day, 1–24 |
| `load_DA`, `load_RT` | Area 1 regional load, MW |
| `wind_DA`, `wind_RT` | plant `122_WIND_1`, 713.5 MW capacity |
| `pv_DA`, `pv_RT` | 10 Area-1 utility PV plants summed, 404 MW capacity |

`_DA` = day-ahead forecast (hourly, native). `_RT` = actual, originally 5-minute,
averaged to hourly here.

Load peaks at 2850 MW, mean 1385 MW.

### Data caveats — read before using

- **Load DA/RT is not a forecast error.** The two series correlate at 1.000 and
  `load_RT ≈ 0.92 × load_DA`. It is a fixed scaling factor, not uncertainty. Use load as
  **deterministic**; take one of the two series and stick to it.
- **Wind is the real uncertainty source.** Correlation 0.749, error std 184 MW
  (≈26% of capacity in RMSE), AR(1) coefficient 0.87. Not a time-alignment artifact —
  shifting ±1, ±2 hours does not improve correlation.
- **Wind error is large by real-world standards.** Operational day-ahead wind forecasts
  are typically better. Cross-check against ENTSO-E Transparency / Open Power System Data
  for Spain, and if RTS-GMLC is pessimistic, scale it down and document the scaling.
- **PV**: error std ≈7.6% of capacity, AR(1) 0.73. Plausible, usable.
- **Provenance**: RTS-GMLC renewable profiles come from WWSIS-2 projected onto a
  southwestern US footprint. The DA series are constructed, not real control-room
  forecasts. State this in the thesis rather than letting an examiner find it.

---

## 3. Notation

| Symbol | Meaning |
|---|---|
| $t = 1,\dots,T$ | periods (24) |
| $\lambda_t$ | energy price (dual of balance) |
| $\mu_t$ | dual of SOC dynamics = marginal value of stored energy |
| $s_t$ | state of charge, MWh |
| $c_t, d_t$ | charge / discharge power, MW |
| $\alpha, \beta$ | charge / discharge efficiency |
| $\varepsilon, \zeta$ | charge / discharge marginal cost |
| $\bar S, \bar C, \bar D$ | energy and power limits |
| $b(s_T)$ | end-of-horizon value function |
| $v_t(s)$ | marginal value of stored energy at time $t$, state $s$ |
| $\omega$ | scenario |

Suggested battery: 100 MW / 400 MWh, $\alpha = \beta = 0.95$. Roughly 4% of peak load in
power and enough energy for a 4-hour shift — big enough to move prices, small enough not
to dominate.

---

## 4. Model 1 — Baldick benchmark (upper bound)

$$\max \;\; \sum_t \Big[ U_t(q_t) - C_t(g_t) - \zeta d_t - \varepsilon c_t \Big] + b(s_T)$$

subject to

$$g_t + d_t - c_t = q_t \qquad (\lambda_t)$$
$$s_t = s_{t-1} + \alpha c_t - \tfrac{d_t}{\beta} \qquad (\mu_t)$$
$$0 \le s_t \le \bar S, \quad 0 \le c_t \le \bar C, \quad 0 \le d_t \le \bar D, \quad s_0 \text{ given}$$

Solve once **per scenario** with that scenario's realised wind and PV. This is
$SW_{\text{Baldick}}(\omega)$, the benchmark. Store $\mu_t$ — you need it for validation.

Note on simultaneous charge/discharge: with $\alpha\beta < 1$ and positive prices it
should not occur. **Check it rather than assume it**; if it does, your relaxation is
leaking and the LP is not equivalent to the physical model.

---

## 5. Model 2 — Value function by dynamic programming

Backward recursion on a discretised SOC grid $s^0,\dots,s^N$ with step $\Delta$
(start with $\Delta = 10$ MWh, i.e. 40 points for a 400 MWh battery).

$$V_T(s) = b(s)$$

$$V_t(s) = \max_{c,d} \Big\{ \hat\lambda_t (d - c) - \zeta d - \varepsilon c
+ V_{t+1}\!\big(s + \alpha c - \tfrac{d}{\beta}\big) \Big\}$$

with $\hat\lambda_t$ the **forecast** price (see §7). Under uncertainty, replace with
$\mathbb{E}_\omega$ over scenarios.

$$v_t(s) = \frac{V_t(s+\Delta) - V_t(s)}{\Delta}$$

Runs in well under a second in plain NumPy at this size. No solver needed.

**Concavity of $V_t$ is required** for the next step to produce a valid curve. Check it
explicitly — if $v_t$ is not monotone there is no submittable curve, and that is a
finding, not a bug to paper over.

---

## 6. Model 3 — Curve construction

The curve is a stack of blocks built from $v_t$, starting at the expected SOC and
working outward.

**Discharge block** taking SOC from $s$ to $s - \Delta/\beta$, delivering $\Delta\beta$ MWh:

$$\pi^{\text{offer}}_t(s) = \frac{v_t(s)}{\beta} + \zeta$$

**Charge block** taking SOC from $s$ to $s + \Delta$, buying $\Delta/\alpha$ MWh:

$$\pi^{\text{bid}}_t(s) = \alpha\, v_t(s) - \varepsilon$$

Since $v_t$ falls as $s$ rises, offer prices increase with quantity (upward-sloping
supply) and bid prices decrease with quantity (downward-sloping demand). The gap between
the two at the same $s$ is the round-trip efficiency wedge plus $\varepsilon + \zeta$.

---

## 7. Forecast and scenarios

Do **not** build a price forecaster. Make forecast quality a parameter.

1. **Error model.** Fit AR(1) to the hourly wind error
   $e_t = \text{wind\_RT}_t - \text{wind\_DA}_t$:
   $e_t = \rho\, e_{t-1} + \eta_t$, $\rho \approx 0.87$. Same for PV, $\rho \approx 0.73$.
2. **Preserve autocorrelation.** This matters more than anything else. Independent hourly
   errors let the battery average them away across the day and will systematically
   **understate** $\Delta SW$ — the quantity you are trying to measure.
3. **Clip to physical bounds**: $0 \le \text{wind} \le 713.5$, PV zero at night.
4. **Two knobs:**
   - scale $k$ multiplying error magnitude: $k \in \{0, 0.5, 1, 2\}$
   - bias $\delta$ shifting the mean
5. **$k = 0$ is the perfect-information case** and must reproduce the equivalence result
   exactly. This is your regression test — see §9.
6. Generate $N$ scenarios (start with 20–50). The forecast $\hat\lambda_t$ used in the DP
   comes from clearing the market against the **mean** scenario.

---

## 8. The comparison loop

```
for each forecast quality (k, delta):
    build forecast scenario
    solve DP  ->  v_t(s)
    build curve B_t from v_t
    for each realised scenario omega:
        clear conventional market with B_t, realised wind/PV   -> SW_conv(omega)
        solve Baldick LP with realised wind/PV                 -> SW_bald(omega)
    report mean gap, distribution of gap
```

The conventional clearing is the same LP as Baldick with the battery replaced by ordinary
supply/demand blocks. Whether you keep the SOC constraints depends on which market you
are comparing against — see §11.

---

## 9. Validation — do this before anything else

Build these in order and make each pass exactly before moving on.

1. **Merit order sanity.** No battery, no uncertainty. Prices should sit at the block
   marginal costs from `units.csv`. If a price appears that is not in that list, something
   is wrong.
2. **Perfect-foresight equivalence.** With $k = 0$:
   - $v_t(s^*_t) = \mu_t$ along the optimal trajectory, to numerical tolerance
   - conventional dispatch = Baldick dispatch
   - $SW_{\text{conv}} = SW_{\text{Baldick}}$

   **Until this matches, any welfare gap you measure later is as likely to be a bug as a
   result.** This is the single most important test in the project.
3. **No simultaneous charge and discharge** in any period.
4. **Grid sensitivity.** Re-run with $\Delta = 5, 10, 20$ MWh. Report how $\Delta SW$
   moves. Too coarse distorts the curve; too fine gives hundreds of blocks.

---

## 10. Known traps

- **Degenerate duals.** When the battery is marginal, $\mu_t$ is not unique and neither is
  the equivalent curve. Perturb block marginal costs by a few cents to break ties.
- **Terminal condition drives everything.** $b(s_T)$ determines the whole $\mu_t$ path by
  backward propagation. A careless choice (e.g. $b \equiv 0$) will empty the battery at
  $T$ and contaminate every result. Use a linear $b(s) = \bar\mu\, s$ with $\bar\mu$ near
  the average off-peak price, and run a sensitivity on it.
- **Efficiency bookkeeping.** $\alpha$ multiplies energy *into* storage, $1/\beta$ divides
  energy *out*. Getting this backwards produces plausible-looking but wrong arbitrage
  spreads. Verify: the round trip should lose $1 - \alpha\beta \approx 10\%$.
- **Units.** MW vs MWh with hourly periods are numerically equal and it is easy to hide an
  error. Label every array.
- **Welfare needs elastic demand.** With purely inelastic load, welfare reduces to minus
  cost, which is fine but say so. Adding one elastic block makes "social welfare" a
  meaningful phrase.

---

## 11. Open question to settle with your supervisor

Which "current design" is the comparison?

- **European (OMIE / EUPHEMIA)** — no resource model at all. No SOC, no efficiencies in
  the clearing. The battery submits independent hourly bids and offers; the only
  intertemporal expression is block orders and exclusive groups. Maximum contrast with
  Baldick.
- **CAISO (NGR model)** — the ISO *does* enforce SOC dynamics and limits, and the owner
  submits an initial SOC and optionally an end-of-hour SOC parameter. The difference with
  Baldick narrows to two things: the bid is a price–quantity curve rather than a function
  of SOC, and there is **no $b(s_T)$** — the owner must embed end-of-horizon value in the
  curve prices.

The second framing is the more interesting thesis, and probably where the result lives.
Verify the CAISO details against the Business Practice Manuals directly; secondary
sources are not enough for a citation.

---

## 12. Suggested layout

```
ess-thesis/
├── data/
│   ├── units.csv
│   └── timeseries_hourly_2020.csv
├── src/
│   ├── data.py          # load csvs, select days, build scenarios
│   ├── errors.py        # AR(1) fit + scenario generation, k and delta knobs
│   ├── baldick.py       # benchmark LP, returns SW and mu_t
│   ├── dp.py            # value function, returns V_t and v_t
│   ├── curves.py        # v_t -> price/quantity blocks
│   ├── clearing.py      # conventional market clearing
│   └── compare.py       # the loop in §8
├── tests/
│   └── test_equivalence.py   # §9.2 — the critical one
└── notebooks/
```

Python + Pyomo or CVXPY + HiGHS is plenty at this size. Gurobi is unnecessary.

---

## 13. First session

1. Load both CSVs, plot `wind_DA` against `wind_RT` for one week. Confirm the error
   structure looks like something worth modelling.
2. Pick three representative days: high wind, low wind, high load. Debug on these, not on
   the full year.
3. Build the merit-order clearing with no battery and check test 9.1.

Do not touch uncertainty until test 9.2 passes.

---

## 14. Citations to line up

- RTS-GMLC: C. Barrows et al., "The IEEE Reliability Test System: A Proposed 2019 Update",
  *IEEE Transactions on Power Systems*. Repository: `github.com/GridMod/RTS-GMLC`.
- WWSIS-2 for the underlying renewable profiles.
- Baldick's SOC formulation (your primary reference).
- CAISO Business Practice Manuals and DMM annual/special reports on battery storage, if
  the CAISO framing is chosen.
- ENTSO-E Transparency Platform / Open Power System Data for the European error
  cross-check.
