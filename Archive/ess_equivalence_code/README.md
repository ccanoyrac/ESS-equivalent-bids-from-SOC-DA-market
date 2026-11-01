# Equivalent ESS bids and Baldick's SOC-based design: code

Code behind the numerical results of the note *KKT Equivalence of Interval-by-Interval ESS Bids and
Baldick's SOC-Based Day-Ahead Market Design* (C. Canoyra Calero, DTU, October 2026).
All models are linear programs solved with HiGHS through SciPy. Test cases: the three single-bus
cases of Baldick [1, §4.7].

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.13, NumPy 2.5, SciPy 1.18, Matplotlib 3.11, pandas 3.0.

## Quick start

```bash
jupyter notebook check_results.ipynb      # step-by-step check of every result, ~1 min
```

or run the scripts directly from this folder:

| Script | What it checks | Where in the note | Run time |
|---|---|---|---|
| `verify_equivalence.py` | Solves the SOC design, KKT residuals (S1)–(S8), unrolled costate, bids (E), current-design clearing, Props 1–2 | Sections 2–3 | seconds |
| `verify_price_based.py` | Lemma 4 (self-schedule), Prop. 5 polyhedron vs dual optimal face, Theorem 1 at every extreme point | Section 4 | seconds |
| `numerical_check.py` | All checks above for the three cases, plus Prop. 4 (π from prices) and the pre-limit rule; writes `figures/` | Section 5: Table 3, Figures 1–2 | ~20 s |
| `computational_burden.py` | Problem size and solve time of both designs for T = 24, 96, 288 intervals/day and up to E = 200 batteries | Project narrative, Section 2 | a few minutes |

`numerical_check.py` imports the first two; `check_results.ipynb` imports all four.

## Conventions

* **Multipliers.** The Lagrangian is written as f + multiplier × (lhs − rhs) for every constraint, and
  multipliers of rows in MW are divided by ΔT, so all multipliers are in $/MWh (as in the note).
  SciPy returns marginals = ∂f*/∂rhs, hence `multiplier = −marginal` (÷ ΔT for MW rows); see
  `duals()` in `verify_equivalence.py`.
* **Network.** Lines are node pairs (n, m) with flow f_{n,m} = B_{n,m}(θ_n − θ_m), as in the note.
  Baldick's cases have a single bus, so the network rows are empty.
* **Charge/discharge exclusivity** is relaxed (pure LP). With round-trip losses (β > α) the LP never
  charges and discharges at once as long as π_t > −(ε + ζ)/(β − α), which holds in all cases here.
* **Terminal bid** is linear, b(Δ) = w·Δ.

## Notes

* A 3-bus test system is still defined in `verify_equivalence.py` (`three_bus_case`) but is not run;
  the current work uses Baldick's example only.
* When the current design is cleared with the equivalent bids, its solver may return a different
  dispatch with the same cost and prices (ties at the bid edges, Remark 1 of the note). The checks
  therefore verify that Baldick's dispatch attains the current-design optimum, not that the solver
  returns it.
* Timings in `computational_burden.py` depend on the machine; the size formulas
  (Δrows = 3ET, Δcols = ET, Δnnz = 8ET − 3E) are exact.

[1] R. Baldick, "A state-of-charge based formulation for storage participation in electricity
markets: Technical reference," arXiv:2606.14993, 2026.
