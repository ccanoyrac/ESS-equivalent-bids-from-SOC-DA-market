"""Loading and slicing for `Data/units.csv` and `Data/timeseries_hourly_2020.csv`
(see `GUIDE.md` sections 1-2 at the repo root for the full data description).

Both files live outside `Code/`, at the repo root's `Data/` folder, and are located
relative to this file so callers don't need to worry about the working directory.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "Data"
UNITS_CSV = DATA_DIR / "units.csv"
TIMESERIES_CSV = DATA_DIR / "timeseries_hourly_2020.csv"

WIND_CAPACITY_MW = 713.5
PV_CAPACITY_MW = 404.0


def load_units(path: Path = UNITS_CSV) -> pd.DataFrame:
    """RTS-GMLC Area 1 thermal fleet, aggregated to one block per unit type/fuel.

    `PMax_MW` is already the *block* total (the six values sum to the 2718 MW quoted in
    GUIDE.md, not per-unit), so each row is directly usable as one aggregate generator:
    capacity `PMax_MW`, marginal cost `MC_eur_MWh`. `n_units` is informational only (how
    many physical units were aggregated into the block).
    """
    df = pd.read_csv(path)
    return df.sort_values("MC_eur_MWh").reset_index(drop=True)


def load_timeseries(path: Path = TIMESERIES_CSV) -> pd.DataFrame:
    """8784 hourly rows for 2020. Adds a `date` (Month, Day) key and a 0-based `hour`
    column (`Period` is 1-24) for convenient slicing; every other column is passed
    through unchanged. See GUIDE.md's "Data caveats" before treating `load_DA`/`load_RT`
    or `wind_DA`/`wind_RT` as independent series -- load is a near-fixed scaling of DA,
    not a forecast error, and only wind/PV carry genuine day-ahead uncertainty.
    """
    df = pd.read_csv(path)
    df["hour"] = df["Period"] - 1
    return df


def day_slice(ts: pd.DataFrame, month: int, day: int) -> pd.DataFrame:
    """The 24 rows (Period 1..24, in order) for one calendar day."""
    out = ts[(ts["Month"] == month) & (ts["Day"] == day)].sort_values("Period")
    if len(out) != 24:
        raise ValueError(f"expected 24 hours for {month}/{day}, found {len(out)}")
    return out.reset_index(drop=True)


def pick_representative_days(ts: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Three (month, day) pairs for early debugging, per GUIDE.md section 13: the
    highest-mean-wind day, the lowest-mean-wind day, and the highest-mean-load day
    (using the RT/actual series, since that's what "high wind" or "high load" means
    physically; the DA forecast is used separately as the forecast input).
    """
    daily = ts.groupby(["Month", "Day"]).agg(
        wind_RT=("wind_RT", "mean"), load_RT=("load_RT", "mean")
    )
    high_wind = daily["wind_RT"].idxmax()
    low_wind = daily["wind_RT"].idxmin()
    high_load = daily["load_RT"].idxmax()
    return {"high_wind": high_wind, "low_wind": low_wind, "high_load": high_load}
