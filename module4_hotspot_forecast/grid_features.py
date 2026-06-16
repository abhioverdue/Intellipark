from pathlib import Path
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

OUTPUT_DIR = (
    ROOT
    / "module4_hotspot_forecast"
    / "output"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "training_data.parquet"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def add_lag_features(
    df: pd.DataFrame,
    group_col: str,
    target_col: str
) -> pd.DataFrame:

    grouped = df.groupby(
        group_col,
        observed=True
    )
    # Add lag features: skip lag_2 (rarely helps) for selective signal
    for lag in [1, 3, 6, 24, 168]:
        df[f"{target_col}_lag_{lag}"] = (
            grouped[target_col]
            .shift(lag)
        )

    # Add rolling-window mean features (skip rolling_3 as too noisy)
    for window in [6, 12, 24]:
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
        "violation_count"
    )

    grid_df = add_lag_features(
        grid_df,
        "grid_cell_id",
        "avg_impact_score"
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