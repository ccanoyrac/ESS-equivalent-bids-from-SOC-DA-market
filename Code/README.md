# SOC Market Framework

A reproduction and research framework for Ross Baldick's *"A state-of-charge based
formulation for storage participation in electricity markets"* (arXiv:2606.14993,
June 2026) -- see [`../Literature/`](../Literature/) for the paper and
[`../Archive/baldick_thesis_conversation.md`](../Archive/baldick_thesis_conversation.md)
for the discussion that shaped this framework's direction. [`ROADMAP.md`](ROADMAP.md)
has the research plan and the ranked list of thesis directions.

## The model, in one paragraph

Instead of the ESS submitting interval-by-interval charge/discharge bids that require it
to forecast prices within the market horizon, the ISO directly optimizes the ESS's SOC
trajectory using its physical characteristics (round-trip efficiency, SOC limits,
degradation costs). The ESS's only private input is a concave **terminal value function**
`b(s_T - sbar_T)`: its willingness-to-pay for ending the horizon with more or less energy
than it would have had absent market participation. In a dynamic-programming reading, `b`
is the value-to-go beyond the end of the horizon -- see `terminal_value/base.py`.

## Layout

```
core/               StorageParams, ThermalGenerator, RenewableGenerator, System
terminal_value/     the b(.) bid function -- constant and piecewise-linear constructors
markets/            day_ahead.py: builds and solves the ISO welfare-maximization LP with Gurobi
experiments/        reproduce_baldick.py: the paper's three Section 4.7 examples
notebooks/          reproduce_baldick.ipynb: the above, run and plotted
tests/              regression tests locking the solver to Tables 2-4
```

## Setup

```
python -m venv .venv
.venv/Scripts/activate           # .venv/bin/activate on macOS/Linux
pip install -e ".[dev,viz,notebook]"
```

`viz` and `notebook` are optional -- skip them if you only need `pytest` and the plain
script below. `gurobipy` is a core dependency and needs no license setup: its pip package
ships a free "size-limited" license good for up to 2000 variables/constraints, which the
examples here (~50 variables) are nowhere near. That cap will eventually matter for
ROADMAP.md stage 2 ("well over 100 ESS" per paper Section 4.8) -- at that point a real
license (e.g. a free academic one, if you're at a university) will be needed.

## Run

```
python -m experiments.reproduce_baldick   # prints Tables 2-4 reproduced, plus objective/surplus
pytest                                    # regression tests against the paper
jupyter lab notebooks/reproduce_baldick.ipynb   # the same run, with tables and plots
```

If `jupyter`/`jupyter lab` itself won't launch (Windows "Application Control policy
blocked this file" on the console-script shim), run nbconvert as a module instead,
which sidesteps the shim entirely:
`python -m nbconvert --to notebook --execute --inplace notebooks/reproduce_baldick.ipynb`.

## A note on non-uniqueness

The paper flags that clearing prices in interval 1 (and interval 6 of the base case) are
not unique -- any price in a stated range is dual-feasible. Running this solver surfaces a
related fact the paper doesn't spell out: **parts of the primal dispatch aren't unique
either.** Intervals 3-4 share identical demand (300 MW) and RAC (500 MW); in the base case,
thermal (interior, not at a bound) and storage discharge tie exactly in marginal cost
across intervals 5-6. Both are genuine continuous degeneracies -- confirmed algebraically,
not solver quirks -- so the specific interval-by-interval split you get from Gurobi's
simplex can differ from the paper's chosen vertex while the *sums* across each tied pair, the
SOC at every bracketing checkpoint, and the total objective are always identical. See
`tests/test_reproduce_baldick.py` for exactly what's asserted exact vs. as a pair-sum/range,
and ROADMAP.md's pricing section for why this matters for the thesis.
