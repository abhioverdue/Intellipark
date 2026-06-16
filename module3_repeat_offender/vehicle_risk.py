from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "module2_impact_score"
    / "output"
    / "scored.parquet"
)

CONFIG_PATH = (
    ROOT
    / "module3_repeat_offender"
    / "config.json"
)

OUTPUT_DIR = (
    ROOT
    / "module3_repeat_offender"
    / "output"
)

VEHICLE_OUTPUT = (
    OUTPUT_DIR
    / "vehicle_risk.parquet"
)

MERGED_OUTPUT = (
    OUTPUT_DIR
    / "risk_tagged.parquet"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def min_max(series):

    minimum = series.min()
    maximum = series.max()

    if minimum == maximum:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (series - minimum) / (
        maximum - minimum
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    logger.info("Loading scored dataset")

    df = pd.read_parquet(INPUT_PATH)

    logger.info(f"Loaded {len(df):,} rows")

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    weights = config["risk_weights"]

    risk_quantiles = config["risk_quantiles"]

    minimum_violations = config["minimum_violations"]

    critical_escalation = config["critical_escalation"]

    weight_sum = sum(weights.values())

    if not np.isclose(weight_sum, 1.0, atol=0.01):
        raise ValueError(
            f"Risk weights must sum to 1.0 "
            f"(current: {weight_sum:.2f})"
        )

    logger.info("Weight validation passed")

    df["created_datetime"] = pd.to_datetime(
        df["created_datetime"]
    )

    df["violation_date"] = (
        df["created_datetime"]
        .dt.date
    )

    required_columns = [
        "vehicle_number",
        "created_datetime",
        "impact_score",
        "impact_priority",
        "grid_cell_id",
        "is_peak"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    logger.info("Schema validation passed")

    vehicle_col = "vehicle_number"

    grouped = df.groupby(
        vehicle_col,
        observed=True
    )

    vehicle_df = grouped.agg(
        total_violations=(
            vehicle_col,
            "size"
        ),

        active_days=(
            "violation_date",
            "nunique"
        ),

        mean_impact_score=(
            "impact_score",
            "mean"
        ),

        max_impact_score=(
            "impact_score",
            "max"
        ),

        critical_ratio=(
            "impact_priority",
            lambda x: (
                (x == "CRITICAL").mean()
            )
        ),

        unique_grid_cells=(
            "grid_cell_id",
            "nunique"
        ),

        peak_hour_ratio=(
            "is_peak",
            "mean"
        )
    ).reset_index()

    top_grid = (
        grouped["grid_cell_id"]
        .agg(
            lambda x:
            x.value_counts(normalize=True)
            .iloc[0]
        )
        .reset_index(
            name="top_grid_concentration"
        )
    )

    vehicle_df = vehicle_df.merge(
        top_grid,
        on=vehicle_col,
        how="left"
    )

    vehicle_df = vehicle_df[
        vehicle_df["total_violations"]
        >= minimum_violations
    ].copy()

    logger.info(
        f"After filtering (min {minimum_violations} violations): "
        f"{len(vehicle_df):,} vehicles"
    )

    vehicle_df["violations_per_day"] = (
        vehicle_df["total_violations"]
        / vehicle_df["active_days"]
    )

    vehicle_df["future_risk_score"] = (
        weights["total_violations"]
        * min_max(
            vehicle_df["total_violations"]
        )

        + weights["critical_ratio"]
        * min_max(
            vehicle_df["critical_ratio"]
        )

        + weights["top_grid_concentration"]
        * min_max(
            vehicle_df[
                "top_grid_concentration"
            ]
        )

        + weights["mean_impact_score"]
        * min_max(
            vehicle_df[
                "mean_impact_score"
            ]
        )
    )

    vehicle_df["risk_level"] = pd.qcut(
        vehicle_df["future_risk_score"],
        q=[
            0,
            risk_quantiles["low"],
            risk_quantiles["medium"],
            risk_quantiles["high"],
            1.0
        ],
        labels=[
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
        ],
        duplicates="drop"
    )

    escalation_mask = (
        (vehicle_df["total_violations"]
            >= critical_escalation["min_violations"])
        &
        (vehicle_df["critical_ratio"]
            >= critical_escalation["critical_ratio"])
    )

    vehicle_df.loc[
        escalation_mask,
        "risk_level"
    ] = "CRITICAL"

    risk_dist = (
        vehicle_df["risk_level"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    logger.info("Risk distribution (%)")

    for level, pct in risk_dist.items():
        logger.info(f"{level}: {pct}%")

    vehicle_df["risk_explanation"] = (
        "Violations="
        + vehicle_df["total_violations"].astype(str)
        + "; CriticalRatio="
        + vehicle_df["critical_ratio"].round(2).astype(str)
        + "; Recurrence="
        + vehicle_df["top_grid_concentration"].round(2).astype(str)
        + "; MeanImpact="
        + vehicle_df["mean_impact_score"].round(1).astype(str)
    )

    action_map = {
        "LOW": "IGNORE",
        "MEDIUM": "MONITOR",
        "HIGH": "PRIORITY_TICKET",
        "CRITICAL": "IMMEDIATE_TOW"
    }

    vehicle_df["recommended_action"] = (
        vehicle_df["risk_level"]
        .astype(str)
        .map(action_map)
    )

    vehicle_df.to_parquet(
        VEHICLE_OUTPUT,
        index=False
    )

    merged = df.merge(
        vehicle_df[
            [
                vehicle_col,
                "future_risk_score",
                "risk_level"
            ]
        ],
        on=vehicle_col,
        how="left"
    )

    merged.to_parquet(
        MERGED_OUTPUT,
        index=False
    )

    logger.info(
        f"Vehicles: {len(vehicle_df):,}"
    )

    logger.info(
        f"Critical vehicles: "
        f"{(vehicle_df['risk_level'] == 'CRITICAL').sum():,}"
    )

    logger.info(
        f"Saved {MERGED_OUTPUT}"
    )


if __name__ == "__main__":
    main()