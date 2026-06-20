from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "module3_repeat_offender"
    / "output"
    / "risk_tagged.parquet"
)

CONFIG_PATH = (
    ROOT
    / "module4_hotspot_forecast"
    / "config.json"
)

OUTPUT_DIR = (
    ROOT
    / "module4_hotspot_forecast"
    / "output"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "training_data.parquet"
)

# Fallback if config.json is missing entirely -- matches the lags/windows
# this script has always computed, so a missing config doesn't change
# behavior.
DEFAULT_LAGS = [1, 3, 6, 24, 168]
DEFAULT_ROLLING_WINDOWS = [6, 12, 24]


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {
            "grid_features": {
                "lags": DEFAULT_LAGS,
                "rolling_windows": DEFAULT_ROLLING_WINDOWS,
            }
        }
    with open(CONFIG_PATH) as f:
        return json.load(f)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def add_lag_features(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    lags: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:

    grouped = df.groupby(
        group_col,
        observed=True
    )

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = (
            grouped[target_col]
            .shift(lag)
        )

    for window in rolling_windows:
        df[f"{target_col}_rolling_{window}"] = (
            grouped[target_col]
            .transform(
                lambda x: (
                    x.shift(1)
                     .rolling(window, min_periods=1)
                     .mean()
                )
            )
        )

    return df


def validate_schema(df: pd.DataFrame) -> None:

    required_cols = [
        "created_datetime",
        "grid_cell_id",
        "impact_score",
        "impact_priority",
        "risk_level",
        "vehicle_number"
    ]

    missing = [
        c for c in required_cols
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:

    int_cols = [
        "hour",
        "dow",
        "month",
        "is_weekend"
    ]

    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype("int8")

    float_cols = df.select_dtypes(
        include=["float64"]
    ).columns

    for col in float_cols:
        df[col] = df[col].astype("float32")

    return df


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    config = load_config()
    gf_config = config.get("grid_features", {})
    lags = gf_config.get("lags", DEFAULT_LAGS)
    rolling_windows = gf_config.get("rolling_windows", DEFAULT_ROLLING_WINDOWS)

    logger.info(f"Lag features: {lags}")
    logger.info(f"Rolling-window features: {rolling_windows}")

    logger.info("Loading risk-tagged dataset")

    df = pd.read_parquet(INPUT_PATH)

    logger.info(f"Loaded {len(df):,} rows")

    validate_schema(df)

    logger.info("Schema validation passed")

    df["created_datetime"] = pd.to_datetime(
        df["created_datetime"],
        errors="coerce"
    )

    invalid_dates = df["created_datetime"].isna().sum()

    if invalid_dates > 0:
        logger.warning(
            f"Removing {invalid_dates:,} rows with invalid timestamps"
        )

        df = df.loc[
            df["created_datetime"].notna()
        ].copy()

    df["date_hour"] = (
        df["created_datetime"]
        .dt.floor("h")
    )

    grid_df = (
        df.groupby(
            [
                "grid_cell_id",
                "date_hour"
            ],
            observed=True
        )
        .agg(
            violation_count=(
                "vehicle_number",
                "size"
            ),

            avg_impact_score=(
                "impact_score",
                "mean"
            ),

            critical_ratio=(
                "impact_priority",
                lambda x: (
                    x.eq("CRITICAL")
                    .mean()
                )
            ),

            repeat_offender_ratio=(
                "risk_level",
                lambda x: (
                    x.isin(
                        [
                            "HIGH",
                            "CRITICAL"
                        ]
                    ).mean()
                )
            )
        )
        .reset_index()
    )

    # Grid-level priors: long-term spatial context for each grid cell
    grid_stats = (
        grid_df.groupby(
            "grid_cell_id",
            observed=True
        )
        .agg(
            grid_mean_count=(
                "violation_count",
                "mean"
            ),
            grid_mean_impact=(
                "avg_impact_score",
                "mean"
            ),
            grid_peak_count=(
                "violation_count",
                "max"
            )
        )
        .reset_index()
    )

    grid_df = grid_df.merge(
        grid_stats,
        on="grid_cell_id",
        how="left"
    )

    logger.info(
        f"Created {len(grid_df):,} grid-hour rows"
    )

    grid_df["hour"] = (
        grid_df["date_hour"]
        .dt.hour
    )

    grid_df["dow"] = (
        grid_df["date_hour"]
        .dt.dayofweek
    )

    grid_df["month"] = (
        grid_df["date_hour"]
        .dt.month
    )

    grid_df["is_weekend"] = (
        grid_df["dow"]
        .isin([5, 6])
        .astype(int)
    )

    # Cyclical encoding

    grid_df["hour_sin"] = np.sin(
        2 * np.pi * grid_df["hour"] / 24
    )

    grid_df["hour_cos"] = np.cos(
        2 * np.pi * grid_df["hour"] / 24
    )

    grid_df["dow_sin"] = np.sin(
        2 * np.pi * grid_df["dow"] / 7
    )

    grid_df["dow_cos"] = np.cos(
        2 * np.pi * grid_df["dow"] / 7
    )

    grid_df = grid_df.sort_values(
        [
            "grid_cell_id",
            "date_hour"
        ]
    )

    logger.info(
        "Generating lag and rolling features"
    )

    grid_df = add_lag_features(
        grid_df,
        "grid_cell_id",
        "violation_count",
        lags,
        rolling_windows,
    )

    grid_df = add_lag_features(
        grid_df,
        "grid_cell_id",
        "avg_impact_score",
        lags,
        rolling_windows,
    )

    logger.info(
        "Imputing missing lag and rolling features"
    )

    # Instead of dropping rows, fill missing lags intelligently
    lag_cols = [
        c for c in grid_df.columns
        if "_lag_" in c or "_rolling_" in c
    ]

    for col in lag_cols:
        if "violation_count" in col:
            # No history = 0 prior violations
            grid_df[col] = grid_df[col].fillna(0)
        else:
            # No impact history = median impact
            grid_df[col] = grid_df[col].fillna(
                grid_df["avg_impact_score"].median()
            )

    logger.info(
        f"Retained all {len(grid_df):,} rows with feature imputation"
    )

    grid_df = optimize_dtypes(grid_df)

    grid_df = grid_df.reset_index(drop=True)

    grid_df.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    logger.info(f"Saved {OUTPUT_PATH}")

    logger.info(
        f"Final shape: {grid_df.shape}"
    )

    logger.info(
        "Feature columns:"
    )

    logger.info(
        ", ".join(grid_df.columns)
    )


if __name__ == "__main__":
    main()