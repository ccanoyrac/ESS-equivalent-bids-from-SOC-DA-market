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
2. **Generalize the code** -- arbitrary `T`, multiple thermal/renewable/storage units,
   eventually a DC network (`network/copperplate.py` → `network/dcopf.py`, not yet built).
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
clearing prices are non-unique. Running the actual solver surfaces a related, sharper
fact: **the primal dispatch is not fully unique either.**

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

## Deliberately deprioritized

- **Ancillary services co-optimization**: Baldick explicitly reserves this for a
  follow-up paper (Section 1, Section 6) -- real risk of overlap.
