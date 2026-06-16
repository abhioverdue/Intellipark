from pathlib import Path
import json
import logging

import pandas as pd


# ==========================================================
# CONFIG
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "module1_pipeline" / "output" / "cleaned.parquet"

OUTPUT_DIR = ROOT / "module1_pipeline" / "output"

FEATURED_PATH = OUTPUT_DIR / "featured.parquet"
QUALITY_PATH = OUTPUT_DIR / "feature_quality.json"

PEAK_HOURS = [8, 9, 10, 11, 16, 17, 18, 19, 20]


# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ==========================================================
# MAIN
# ==========================================================

def main():

    logger.info("Loading cleaned dataset")

    df = pd.read_parquet(INPUT_PATH)

    logger.info(f"Loaded {len(df):,} rows")

    missing_dates = int(
        df["created_datetime"].isna().sum()
    )

    if missing_dates > 0:
        logger.warning(
            f"Removing {missing_dates} rows with invalid timestamps"
        )

        df = df.loc[
            df["created_datetime"].notna()
        ].copy()

    dt = df["created_datetime"]

    df["hour"] = dt.dt.hour.astype("int8")

    df["dow"] = dt.dt.dayofweek.astype("int8")

    df["day_name"] = dt.dt.day_name().astype("category")

    df["month"] = dt.dt.month.astype("int8")

    df["week_of_year"] = (
        dt.dt.isocalendar()
        .week
        .astype("Int64")
    )

    df["is_weekend"] = (
        df["dow"] >= 5
    ).astype("int8")

    df["is_peak"] = (
        df["hour"]
        .isin(PEAK_HOURS)
    ).astype("int8")

    df["is_junction"] = (
        df["junction_name"]
        .astype(str)
        .ne("UNKNOWN")
    ).astype("int8")

    vehicle_counts = (
        df.groupby("vehicle_number")
        .size()
        .rename("historical_violation_count")
        .reset_index()
    )

    df = df.merge(
        vehicle_counts,
        on="vehicle_number",
        how="left",
    )

    df["historical_violation_count"] = (
        df["historical_violation_count"]
        .astype("int16")
    )

    df["is_repeat_offender"] = (
        df["historical_violation_count"] >= 2
    ).astype("int8")

    grid_density = (
        df.groupby("grid_cell_id")
        .size()
        .rename("grid_violation_count")
        .reset_index()
    )

    df = df.merge(
        grid_density,
        on="grid_cell_id",
        how="left",
    )

    df["grid_violation_count"] = (
        df["grid_violation_count"]
        .astype("int32")
    )

    hourly_density = (
        df.groupby(["grid_cell_id", "hour"])
        .size()
        .rename("grid_hour_violation_count")
        .reset_index()
    )

    df = df.merge(
        hourly_density,
        on=["grid_cell_id", "hour"],
        how="left",
    )

    df["grid_hour_violation_count"] = (
        df["grid_hour_violation_count"]
        .astype("int32")
    )

    df["approved_flag"] = (
        df["validation_status"] == "approved"
    ).astype("int8")

    approval_rates = (
        df.groupby("police_station")["approved_flag"]
        .mean()
        .rename("station_approval_rate")
        .reset_index()
    )

    df = df.merge(
        approval_rates,
        on="police_station",
        how="left",
    )

    df["station_approval_rate"] = (
        df["station_approval_rate"]
        .round(4)
    )

    df = df.sort_values(
        "created_datetime"
    ).reset_index(drop=True)

    quality_report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_created_datetime_removed": missing_dates,
        "repeat_offenders": int(
            df["is_repeat_offender"].sum()
        ),
        "unique_grids": int(
            df["grid_cell_id"].nunique()
        ),
    }

    with open(QUALITY_PATH, "w") as f:
        json.dump(quality_report, f, indent=2)

    df.to_parquet(
        FEATURED_PATH,
        index=False,
    )

    logger.info(f"Saved {FEATURED_PATH}")
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Feature report: {QUALITY_PATH}")


if __name__ == "__main__":
    main()