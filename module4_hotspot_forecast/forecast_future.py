from pathlib import Path
import json
import logging

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TRAINING_DATA_PATH = (
    ROOT
    / "module4_hotspot_forecast"
    / "output"
    / "training_data.parquet"
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

COUNT_MODEL_PATH = OUTPUT_DIR / "count_model.pkl"
IMPACT_MODEL_PATH = OUTPUT_DIR / "impact_model.pkl"

FUTURE_FORECAST_PATH = OUTPUT_DIR / "future_grid_forecasts.parquet"

# Default if config.json is missing entirely -- one week, matching what
# this script has always hardcoded.
DEFAULT_FORECAST_HORIZON_HOURS = 24 * 7


def load_forecast_horizon_hours() -> int:
    if not CONFIG_PATH.exists():
        return DEFAULT_FORECAST_HORIZON_HOURS
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    hours = config.get("forecast_horizon_hours", DEFAULT_FORECAST_HORIZON_HOURS)
    if not isinstance(hours, int) or hours <= 0:
        raise ValueError(
            f"forecast_horizon_hours must be a positive integer "
            f"(got {hours!r} in {CONFIG_PATH})"
        )
    return hours


# How many hours forward to forecast, starting the hour right after
# the last real timestamp in training_data.parquet. Read from
# module4_hotspot_forecast/config.json -- previously hardcoded here
# while config.json silently said something different (1 hour vs this
# script's old 24*7), so this is now the single source of truth.
FORECAST_HORIZON_HOURS = load_forecast_horizon_hours()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


BASE_FEATURES = [
    "hour", "dow", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "critical_ratio", "repeat_offender_ratio",
    "violation_count_lag_1", "violation_count_lag_24",
    "violation_count_rolling_6", "violation_count_rolling_24",
    "avg_impact_score_lag_1", "avg_impact_score_lag_24",
    "avg_impact_score_rolling_6", "avg_impact_score_rolling_24",
]

OPTIONAL_FEATURES = [
    "violation_count_lag_3", "violation_count_lag_6", "violation_count_lag_168",
    "avg_impact_score_lag_3", "avg_impact_score_lag_6", "avg_impact_score_lag_168",
    "violation_count_rolling_12", "avg_impact_score_rolling_12",
    "grid_mean_count", "grid_mean_impact",
]


def build_future_skeleton(
    history: pd.DataFrame,
    horizon_hours: int,
) -> pd.DataFrame:
    """
    Construct one row per (grid_cell_id, future hour) for the
    requested horizon, starting right after the last real timestamp
    in the historical data.

    Lag and rolling features are pulled from each grid cell's actual
    recent history (the real tail of training_data.parquet) wherever
    the lag still points backward into known data. For lags that
    would reach past the start of the forecast horizon (lag_1 on
    hour 2 of the forecast, for example, needs hour 1's *predicted*
    count, which isn't known yet at skeleton-build time) we fall back
    to the grid's own historical mean — the same imputation strategy
    already used in grid_features.py for cold-start cells, applied
    here for cold-start future hours instead.
    """

    last_timestamp = history["date_hour"].max()

    future_hours = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=horizon_hours,
        freq="h",
    )

    grid_ids = history["grid_cell_id"].unique()

    skeleton = pd.MultiIndex.from_product(
        [grid_ids, future_hours],
        names=["grid_cell_id", "date_hour"],
    ).to_frame(index=False)

    skeleton["hour"] = skeleton["date_hour"].dt.hour.astype("int8")
    skeleton["dow"] = skeleton["date_hour"].dt.dayofweek.astype("int8")
    skeleton["month"] = skeleton["date_hour"].dt.month.astype("int8")
    skeleton["is_weekend"] = skeleton["dow"].isin([5, 6]).astype("int8")

    skeleton["hour_sin"] = np.sin(2 * np.pi * skeleton["hour"] / 24)
    skeleton["hour_cos"] = np.cos(2 * np.pi * skeleton["hour"] / 24)
    skeleton["dow_sin"] = np.sin(2 * np.pi * skeleton["dow"] / 7)
    skeleton["dow_cos"] = np.cos(2 * np.pi * skeleton["dow"] / 7)

    # Grid-level priors carry forward unchanged — they're long-run
    # averages, not time-dependent.
    grid_priors = (
        history.groupby("grid_cell_id", observed=True)
        .agg(
            grid_mean_count=("grid_mean_count", "first"),
            grid_mean_impact=("grid_mean_impact", "first"),
            grid_peak_count=("grid_peak_count", "first"),
            critical_ratio=("critical_ratio", "mean"),
            repeat_offender_ratio=("repeat_offender_ratio", "mean"),
        )
        .reset_index()
    )

    skeleton = skeleton.merge(grid_priors, on="grid_cell_id", how="left")

    # For lag/rolling features: use each grid cell's last known value
    # as a same-position-in-cycle proxy (e.g. lag_168 = the actual
    # value from exactly one week before forecast start, which is
    # real historical data, not a guess).
    history_sorted = history.sort_values(["grid_cell_id", "date_hour"])
    last_known = (
        history_sorted.groupby("grid_cell_id", observed=True)
        .tail(168)  # last full week per cell, for lag_168 lookups
        .copy()
    )

    for feature_root, fallback_col in [
        ("violation_count", "grid_mean_count"),
        ("avg_impact_score", "grid_mean_impact"),
    ]:
        for lag in [1, 3, 6, 24, 168]:
            col = f"{feature_root}_lag_{lag}"
            if col not in BASE_FEATURES + OPTIONAL_FEATURES:
                continue

            lookup = (
                last_known.groupby("grid_cell_id", observed=True)[feature_root]
                .last()
                .reindex(grid_ids)
            )

            skeleton[col] = skeleton["grid_cell_id"].map(lookup)
            skeleton[col] = skeleton[col].fillna(skeleton[fallback_col])

        for window in [6, 12, 24]:
            col = f"{feature_root}_rolling_{window}"
            if col not in BASE_FEATURES + OPTIONAL_FEATURES:
                continue

            lookup = (
                last_known.groupby("grid_cell_id", observed=True)[feature_root]
                .apply(lambda s: s.tail(window).mean())
            )

            skeleton[col] = skeleton["grid_cell_id"].map(lookup)
            skeleton[col] = skeleton[col].fillna(skeleton[fallback_col])

    return skeleton


def get_features(df: pd.DataFrame) -> list:
    features = BASE_FEATURES.copy()
    for col in OPTIONAL_FEATURES:
        if col in df.columns:
            features.append(col)
    return features


def main():

    logger.info("Loading historical training data")

    history = pd.read_parquet(TRAINING_DATA_PATH)

    logger.info(f"Loaded {len(history):,} historical grid-hour rows")
    logger.info(
        f"History ends at: {history['date_hour'].max()} "
        f"({history['date_hour'].max().day_name()})"
    )

    logger.info("Loading trained models")

    count_model = joblib.load(COUNT_MODEL_PATH)
    impact_model = joblib.load(IMPACT_MODEL_PATH)

    logger.info(
        f"Building future skeleton for the next "
        f"{FORECAST_HORIZON_HOURS} hours "
        f"({FORECAST_HORIZON_HOURS // 24} days)"
    )

    future_df = build_future_skeleton(history, FORECAST_HORIZON_HOURS)

    logger.info(
        f"Future skeleton: {len(future_df):,} grid-hour rows "
        f"({future_df['grid_cell_id'].nunique()} grids x "
        f"{FORECAST_HORIZON_HOURS} hours)"
    )

    features = get_features(future_df)

    missing = [c for c in features if c not in future_df.columns]
    if missing:
        raise ValueError(f"Future skeleton missing columns: {missing}")

    future_df["predicted_violations"] = np.clip(
        count_model.predict(future_df[features]), 0, None
    )

    future_df["predicted_impact"] = np.clip(
        impact_model.predict(future_df[features]), 0, 100
    )

    future_df["forecast_priority"] = pd.qcut(
        future_df["predicted_impact"],
        q=[0, 0.50, 0.80, 0.95, 1.0],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        duplicates="drop",
    )

    output_columns = [
        "grid_cell_id", "date_hour", "hour", "dow",
        "predicted_violations", "predicted_impact",
        "forecast_priority", "critical_ratio", "repeat_offender_ratio",
    ]

    future_df[output_columns].to_parquet(FUTURE_FORECAST_PATH, index=False)

    logger.info(
        f"Mean predicted_violations: {future_df['predicted_violations'].mean():.2f}"
    )
    logger.info(
        f"Mean predicted_impact: {future_df['predicted_impact'].mean():.2f}"
    )

    logger.info(f"Saved {FUTURE_FORECAST_PATH}")


if __name__ == "__main__":
    main()