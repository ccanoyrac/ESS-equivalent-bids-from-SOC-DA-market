# Stage-0 results with the error size given as sigma (archived 2026-11-10)

- `results_stage0/`: `Code/results/stage0` as it was before the switch to the MAE grid:
  B_star_p and B_fcst at sigma = 0, 1, 2, 5, 10, 20 $/MWh (psi = 10, rho = 0, price floor 0,
  prelimit, tie-breaker kappa = 1e-3, 150 draws), B_star, and B_priv at psi = 0 / 10 / 50.
- `stage0_experiments_sigma.ipynb`: the stage-0 notebook executed on those settings
  (Section 9 plots the loss against sigma).

In this case (mean lambda* = 32.18 $/MWh) the sigma values correspond to an MAE of
2.5, 5.0, 12.4, 24.8 and 49.6 % of the mean price (MAE = sigma * sqrt(2/pi)).
The current runs use the MAE grid 0, 2.5, 5, 10, 20, 40 % as input (sigma = sqrt(pi/2) * MAE).
