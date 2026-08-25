# Research Roadmap

Full discussion: [`../Archive/baldick_thesis_conversation.md`](../Archive/baldick_thesis_conversation.md).
This file is the condensed, working version -- update it as directions are chosen,
dropped, or found to overlap with new literature.

## Working thesis question

> How should the end-of-horizon SOC value function `b(.)` be constructed under
> uncertainty, and how does its accuracy affect market efficiency, storage profitability,
> dispatch, and price formation relative to conventional interval-by-interval storage
> bidding?

## Five-stage plan

1. **Exact reproduction** -- done for the stylized 6-interval example (`experiments/reproduce_baldick.py`,
   `tests/test_reproduce_baldick.py`). Dispatch, SOC checkpoints, and objective match
   Tables 2-4; degenerate quantities are documented rather than pinned to one arbitrary
   solver vertex (see "Price formation & dispatch degeneracy" below).
2. **Generalize the code** -- multiple thermal/renewable/storage units done (`System`,
   `solve_day_ahead`, and `DispatchResult` all support any number of each; every storage
   unit carries its own independent terminal value function -- see
   `experiments/Multi_Participant_Baldick.py`). Arbitrary `T` already worked (it was never
   hardcoded). Still open: a DC network (`network/copperplate.py` → `network/dcopf.py`,
   not yet built).
3. **Conventional interval-bidding benchmark** -- implement the paired generator/load
   bid-and-offer model the paper critiques, on the *same* physical system, so efficiency
   loss from price-forecast error can be measured directly (`participants/price_taker.py`,
   not yet built).
4. **Uncertainty and terminal-value estimation** -- construct `b(.)` via perfect
   foresight, stochastic DP, and/or a learned value function
   (`terminal_value/stochastic_dp.py`, `terminal_value/learned_value.py`, not yet built).
5. **Controlled experiments** -- vary renewable penetration, storage penetration,
   terminal-value forecast error, and storage duration; measure welfare loss, storage
   profit, curtailment, and price volatility against the interval-bidding benchmark.

## Ranked research directions

| Direction | Potential | MSc feasibility | Notes |
|---|---:|---:|---|
| **Learning/optimizing `b(.)`** | ★★★★★ | ★★★★★ | Best candidate. `b(delta) ≈ V_{T+1}(sbar_T+delta) - V_{T+1}(sbar_T)`; compare `b^perfect, b^forecast, b^learned, b^robust, b^wrong` against conventional bidding. |
| DA-RT operation under uncertainty | ★★★★★ | ★★★★☆ | Paper only sketches RT (Appendix A.4); re-defines `sbar_T` as the DA SOC at/interpolated to the RT horizon end. |
| Strategic terminal bidding / market power | ★★★★★ | ★★★☆☆ | Is truthful `b(.)` revelation incentive-compatible? Bilevel: ESS maximizes surplus over `b_hat`, subject to the ISO's LP. Extends naturally to multiple co-located storage owners and to network congestion (a price-making ESS behind a binding line has more leverage than one on an uncongested copperplate). |
| Price formation & dispatch degeneracy | ★★★★☆ | ★★★★☆ | See below -- this framework already surfaces a concrete, previously undocumented instance. |
| Energy + ancillary-service co-optimization | ★★★★★ | ★★★☆☆ | Paper explicitly reserves this for a future paper by Baldick -- risk of being scooped. |
| Realistic degradation & SOC-dependent efficiency | ★★★☆☆ | ★★★★★ | Appendix A.1 sketches SOC-dependent power limits and degradation; good extension, weaker novelty. |
| Large-scale computation | ★★★☆☆ | ★★★☆☆ | Paper already reports >100 ESS feasible on MISO-scale systems. |
| Flexible demand via the same SOC model | ★★★☆☆ | ★★★★☆ | Section 5: zero discharge capacity + zero degradation + `gamma=1` turns the ESS model into a daily-energy-budget load. |

## Price formation & dispatch degeneracy (concrete finding)

The archive discussion flagged that interval-1 (and, in the base case, interval-6)
clearing prices are non-unique -- taking the paper's own text at face value. Building an
actual tool to check this (below) shows that claim doesn't hold in this exact
reproduction: interval 1 is dual-*unique* at \$43.1/MWh in all three cases, and so is the
base case's interval 6 (\$50/MWh) -- only the high-willingness-to-pay case's interval 6
turns out to have a genuine range. Rather than silently fix the earlier claim, it's worth
keeping the correction visible: trusting a paper's stated range instead of independently
verifying it is exactly the kind of mistake this framework exists to catch. Running the
solver also surfaces a related, sharper fact: **the primal dispatch is not fully unique
either** -- and, importantly, primal degeneracy and price (dual) degeneracy turn out to
be *different* phenomena that don't necessarily co-occur at the same intervals (see
below).

- Intervals 3-4 have identical demand (300 MW) and RAC (500 MW) in every example case, so
  any split of renewable dispatch/charging between them that respects the running SOC
  bound is equally optimal -- the paper's own text half-acknowledges this
  ("renewable production could be somewhat lower in interval 3 and somewhat higher in
  interval 4...").
- In the base case specifically, thermal sits strictly interior (not at 0 or its 100 MW
  cap) in the 5-6 window, and thermal's $50/MWh offer times `dt` ties exactly with
  storage discharge's $1/MWh degradation cost times `dt` scaled by the SOC constraint
  that pins `discharge[4]+discharge[5] = 200` -- so *any* split with
  `thermal[4]+thermal[5] = 75` and `discharge[4]+discharge[5] = 200`, subject to each
  interval's own balance, is optimal. This does **not** happen in the high-willingness-to-pay
  or increased-max-SOC cases, because there thermal is saturated (100 MW cap) or absent
  (0 MW) rather than interior -- confirming the tie is a genuine LP-structural artifact of
  identical marginal costs across adjacent intervals, not a bug.
- What *is* always forced: dispatch in intervals 1-2 (no tie), SOC at every checkpoint
  bracketing a tied pair (it's pinned by the constraints either side), and the total
  objective value (alternate optima share it by definition).

This is worth developing into its own angle on "Price formation & dual degeneracy":
which pricing/dispatch-selection rule should the ISO commit to when ties like this occur
at scale (hundreds of ESS, many tied interval pairs), and does the choice have real
distributional consequences for ESS surplus even though total welfare is invariant?
(Footnote 17 in the paper already flags that different tie-breaks redistribute ESS
surplus without changing total cost.)

**Independent corroboration**: an earlier, independently-written reproduction of the same
three cases (flat `MarketData`/`StorageData` dataclasses, same `linprog(method="highs")`
LP, different variable ordering) hard-asserted the paper's exact interval-3/4 split in its
own regression tests. Running that code's own test suite in this environment (`pytest` on
its `tests/test_paper_cases.py`) fails on precisely the interval-3/4 swap described above,
for both the base case and the increased-max-SOC case -- a different implementation hitting
the identical degeneracy is good evidence this is a structural property of the LP, not an
artifact of one particular formulation.

**Second corroboration, same codebase**: switching `markets/day_ahead.py` itself from
`scipy`'s HiGHS to Gurobi (same equations, same `System`/`TerminalValueFunction` objects,
only the solver call changed) flipped which vertex gets picked for the interval-3/4 tie --
Gurobi's simplex now lands exactly on the paper's own split in all three cases, where HiGHS
picked the opposite one. The interval-5/6 tie in the base case, however, stayed genuinely
free under both solvers (different splits, same $75 thermal / 200 MWh discharge totals).
That two different solvers *disagree* on one tied pair while *agreeing* the other pair is
still free is about as clean a demonstration as this framework can produce that these are
properties of the optimum, not of any one solver's internals -- and a caution against ever
hard-coding a specific degenerate split into a test, no matter how consistently one solver
version currently reproduces it.

**Rigorous verification via the explicit LP dual**: rather than trust a solver's
arbitrary tie-break or a paper's stated range, a one-off analysis built the actual LP
dual of the solved Gurobi model generically (from `model.getVars()`, `model.getConstrs()`,
`model.getCol()`), fixed it to the primal's own optimal objective value (any dual-feasible
point matching that value is, by strong duality, itself an optimal dual solution), and
solved for the min/max of each balance constraint's dual variable. Validated two ways
before trusting it: (1) solving the dual with a neutral objective reproduced Gurobi's own
reported prices to 10+ decimal places, and (2) `computeIIS()` confirmed $0/MWh is
*infeasible* for interval 1 given the fixed-objective constraint, not just unreached by an
optimizer -- i.e. the uniqueness is a real property of the feasible region, not a search
artifact. This produced the table below (the code itself was a throwaway exploration, not
kept in the package -- worth rebuilding as a permanent tool if this direction gets pursued
further, e.g. as the basis for a controlled "does the ISO's tie-break rule affect ESS
surplus distribution" experiment per stage 5).

Sweeping every interval of all three cases:

| Interval | base_case | high_willingness_to_pay | increased_max_soc |
|---|---|---|---|
| 1 | 43.1 (unique) | 43.1 (unique) | 43.1 (unique) |
| 2 | 50 (unique) | 50 (unique) | 50 (unique) |
| 3 | 0 (unique) | 0 (unique) | 0 (unique) |
| 4 | 0 (unique) | 0 (unique) | 0 (unique) |
| 5 | 50 (unique) | 56 (unique) | 41 (unique) |
| 6 | 50 (unique) | **[50, 56]** | 41 (unique) |

Only one genuine price range exists across all eighteen interval/case combinations. That
it shows up in a *different* case and interval than the ranges the paper's text
describes for the base case is itself informative: primal-dispatch degeneracy (intervals
3-4 always, interval 5-6 in the base case only) and price degeneracy (interval 6, high-w
case only) don't track each other. A tied pair of intervals with multiple valid dispatch
splits can still have a fully unique clearing price throughout (as in the base case's
5-6), and conversely a single interval can be price-degenerate without any adjacent
primal tie explaining it (high-w's interval 6). Worth treating as its own open question
rather than assuming one implies the other.

**Third instance, multi-participant**: `experiments/Multi_Participant_Baldick.py` gives
several renewable owners the same \$0/MWh offer price. The LP is indifferent about which
specific owner supplies a given MW as long as each stays within its own capacity and the
total is right -- the same tied-optimum structure, now between *N* symmetric-cost
participants instead of two adjacent intervals. At the scale this framework eventually
targets (paper Section 4.8's "well over 100 ESS"), an ISO would see this kind of tie
constantly; which allocation rule it commits to is a live instance of the pricing/dispatch
question above, not a corner case.

## Deliberately deprioritized

- **Ancillary services co-optimization**: Baldick explicitly reserves this for a
  follow-up paper (Section 1, Section 6) -- real risk of overlap.
