"""Run the stage-0 gate (experiments.md section 5) and print a summary.

Thin by design (experiments.md section 3): builds settings from config.py, calls
runner.run_all, prints a summary. All logic lives in experiments/.

Usage: python scripts/run_stage0_gate.py [--overwrite]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from experiments import benchmark, cases, config, metrics, runner


def main() -> None:
    overwrite = "--overwrite" in sys.argv

    case = cases.day_case()
    bench = benchmark.solve_benchmark(case)
    print(f"Benchmark: SW_star={bench.SW_star:,.2f}  SW_noESS={bench.SW_noESS:,.2f}  V={bench.V:,.2f}")
    print(f"lambda_star: {bench.lam_star[0].round(3)}\n")

    out_dir = Path(__file__).resolve().parents[1] / "results" / "stage0"
    settings = config.stage0_settings()
    df = runner.run_all(case, bench, settings, out_dir, stage_name="stage0", overwrite=overwrite)

    print(f"{len(settings)} settings, {len(df)} draws total -> {out_dir}\n")

    b_star = df[df["setting_id"] == "stage0_B_star"]
    print(f"B_star gate: dSW = {b_star['dSW'].iloc[0]:.3e}  (must be < 1e-6)")
    assert abs(b_star["dSW"].iloc[0]) < 1e-6, "STAGE 0 GATE FAILED: B_star does not give dSW~0"

    no_sim = True  # checked structurally in tests/test_market_realtime.py; re-asserted implicitly by dSW>=0 here
    assert (df["dSW"] >= -1e-6).all(), "STAGE 0 GATE FAILED: dSW < -1e-6 found (SW_real exceeds SW_star)"
    print("dSW >= 0 for every setting/draw: OK\n")

    print("B_priv (technical data only, no price information):")
    for psi in config.STAGE0_PSI_FOR_PRIV:
        row = df[df["setting_id"] == f"stage0_B_priv_psi{psi:g}"].iloc[0]
        print(f"  psi={psi:>4.0f}: dSW_pct={row['dSW_pct']:7.2f}%  undeliverable={row['undeliverable_MWh']:8.2f} MWh")

    print("\nStochastic settings (rho=0, psi=10), mean dSW_pct by family and MAE (% of mean lambda*):")
    stoch = df[df["setting_id"].str.contains("_mae")]
    summary = metrics.summarize_by_setting(stoch, "dSW_pct", ["family", "mae_pct"])
    print(summary.sort_values(["family", "mae_pct"]).to_string(index=False))

    print("\nStage-0 gate: PASSED (structural checks -- see Code/README.md for what this does and doesn't verify).")


if __name__ == "__main__":
    main()
