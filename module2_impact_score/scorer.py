from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd


# CONFIG

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "module1_pipeline" / "output" / "featured.parquet"

CONFIG_PATH = ROOT / "module2_impact_score" / "config.json"

OUTPUT_DIR = ROOT / "module2_impact_score" / "output"

OUTPUT_PATH = OUTPUT_DIR / "scored.parquet"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


EXPECTED_WEIGHTS = [
    "junction_flag",
    "violation_severity",
    "vehicle_size",
    "peak_hour",
    "hotspot_density",
    "repeat_flag"
]


# HELPERS

def min_max_scale(series: pd.Series) -> pd.Series:

    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    if minimum == maximum:
        return pd.Series(
            np.ones(len(series)),
            index=series.index
        )

    return (series - minimum) / (maximum - minimum)


def normalize_violation(series: pd.Series) -> pd.Series:

    if pd.api.types.is_categorical_dtype(series.dtype):
        series = series.astype(object)

    return (
        series.fillna("DEFAULT")
        .astype(str)
        .str.upper()
        .str.strip()
    )


def validate_weights(weights: dict):

    missing = set(EXPECTED_WEIGHTS) - set(weights.keys())

    if missing:
        raise ValueError(
            f"Missing weights: {missing}"
        )

    total = sum(weights.values())

    if not np.isclose(total, 1.0, atol=0.01):
        raise ValueError(
            f"Weights must sum to 1.0. Current sum: {total:.3f}"
        )

    logger.info(f"Weight sum validated: {total:.2f}")


# MAIN

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    logger.info("Loading featured dataset")

    df = pd.read_parquet(INPUT_PATH)

    logger.info(f"Loaded {len(df):,} rows")

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    weights = config["weights"]

    validate_weights(weights)

    vehicle_scores = config["vehicle_size_scores"]

    severity_scores = config["violation_severity_scores"]

    # Violation severity
    violation_col = None

    candidate_columns = [
        "primary_violation",
        "violation_type",
        "violation",
        "offence_type",
        "description"
    ]

    for col in candidate_columns:
        if col in df.columns:
            violation_col = col
            break

    df["violation_severity_rank"] = severity_scores["DEFAULT"]

    if violation_col is not None:

        logger.info(
            f"Using {violation_col} for severity scoring"
        )

        normalized = normalize_violation(
            df[violation_col]
        )

        mapped_scores = normalized.map(severity_scores)

        matched_pct = (
            mapped_scores.notna().mean() * 100
        )

        logger.info(
            f"Matched severity labels: {matched_pct:.2f}%"
        )

        df["violation_severity_rank"] = (
            mapped_scores.fillna(
                severity_scores["DEFAULT"]
            )
        )

    # Fallback heuristics for rows that remain DEFAULT

    default_mask = (
        df["violation_severity_rank"]
        == severity_scores["DEFAULT"]
    )

    df.loc[
        default_mask
        & df["vehicle_size_class"].isin(
            ["LARGE", "XLARGE"]
        ),
        "violation_severity_rank"
    ] += 1

    df.loc[
        default_mask
        & (df["is_peak"] == 1),
        "violation_severity_rank"
    ] += 1

    hotspot_threshold = (
        df["grid_violation_count"]
        .quantile(0.90)
    )

    df.loc[
        default_mask
        & (
            df["grid_violation_count"]
            >= hotspot_threshold
        ),
        "violation_severity_rank"
    ] += 1

    df["violation_severity_rank"] = (
        df["violation_severity_rank"]
        .clip(1, 5)
    )

    # Vehicle size
    df["vehicle_size_score"] = (
        df["vehicle_size_class"]
        .astype(str)
        .map(vehicle_scores)
        .fillna(2)
    )

    # Normalize features
    df["severity_norm"] = min_max_scale(
        df["violation_severity_rank"]
    )

    df["vehicle_size_norm"] = min_max_scale(
        df["vehicle_size_score"]
    )

    df["hotspot_density_score"] = min_max_scale(
        df["grid_violation_count"]
    )

    # Binary features
    junction_counts = (
        df.groupby(
            "junction_name",
            observed=True
        )
        .size()
    )

    busy_threshold = junction_counts.quantile(0.95)

    busy_junctions = set(
        junction_counts[
            junction_counts >= busy_threshold
        ].index
    )

    df["junction_score"] = (
        df["junction_name"]
        .isin(busy_junctions)
        .astype(int)
    )

    df["peak_score"] = (
        df["is_peak"]
        .fillna(0)
        .astype(int)
    )

    df["repeat_score"] = (
        df["is_repeat_offender"]
        .fillna(0)
        .astype(int)
    )

    # Explainability components
    df["impact_junction"] = (
        weights["junction_flag"]
        * df["junction_score"]
    )

    df["impact_severity"] = (
        weights["violation_severity"]
        * df["severity_norm"]
    )

    df["impact_vehicle"] = (
        weights["vehicle_size"]
        * df["vehicle_size_norm"]
    )

    df["impact_peak"] = (
        weights["peak_hour"]
        * df["peak_score"]
    )

    df["impact_hotspot"] = (
        weights["hotspot_density"]
        * df["hotspot_density_score"]
    )

    df["impact_repeat"] = (
        weights["repeat_flag"]
        * df["repeat_score"]
    )

    # Final score
    component_cols = [
        "impact_junction",
        "impact_severity",
        "impact_vehicle",
        "impact_peak",
        "impact_hotspot",
        "impact_repeat"
    ]

    df["impact_score"] = (
        df[component_cols]
        .sum(axis=1)
        * 100
    ).round(2)

    # Priority bands
    df["impact_priority"] = pd.qcut(
        df["impact_score"],
        q=[0, 0.50, 0.80, 0.95, 1.0],
        labels=[
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
        ],
        duplicates="drop"
    )

    # Ranking
    df["citywide_rank"] = (
        df["impact_score"]
        .rank(
            method="dense",
            ascending=False
        )
        .astype("int32")
    )

    df = df.sort_values(
        "impact_score",
        ascending=False
    ).reset_index(drop=True)

    # Diagnostics
    logger.info(
        f"Mean impact score: "
        f"{df['impact_score'].mean():.2f}"
    )

    logger.info(
        f"Median impact score: "
        f"{df['impact_score'].median():.2f}"
    )

    logger.info(
        f"Max impact score: "
        f"{df['impact_score'].max():.2f}"
    )

    logger.info(
        f"Min impact score: "
        f"{df['impact_score'].min():.2f}"
    )

    priority_dist = (
        df["impact_priority"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .sort_index()
    )

    logger.info("Priority distribution (%)")

    for level, pct in priority_dist.items():
        logger.info(f"{level}: {pct}%")

    logger.info(
        f"Critical violations: "
        f"{(df['impact_priority'] == 'CRITICAL').sum():,}"
    )

    logger.info("Feature contribution means")

    contribution_summary = (
        df[component_cols]
        .mean()
        .mul(100)
        .round(2)
    )

    for feature, value in contribution_summary.items():
        logger.info(f"{feature}: {value}")

    logger.info(
        f"Junction score distribution:\n"
        f"{df['junction_score'].value_counts(normalize=True)}"
    )

    logger.info(
        f"Severity distribution:\n"
        f"{df['violation_severity_rank'].value_counts(normalize=True)}"
    )

    # Save
    df.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    logger.info(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()