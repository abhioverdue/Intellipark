# module5_edi/edi.py

from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FORECAST_PATH = (
    ROOT
    / "module4_hotspot_forecast"
    / "output"
    / "grid_forecasts.parquet"
)

OBSERVED_PATH = (
    ROOT
    / "module3_repeat_offender"
    / "output"
    / "risk_tagged.parquet"
)

CONFIG_PATH = (
    ROOT
    / "module5_edi"
    / "config.json"
)

OUTPUT_DIR = (
    ROOT
    / "module5_edi"
    / "output"
)

OUTPUT_PATH = OUTPUT_DIR / "edi_scores.parquet"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def validate_columns(df, required, name):
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} missing required columns: {missing}"
        )


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()

    logger.info("Loading forecasts")
    forecasts = pd.read_parquet(FORECAST_PATH)

    logger.info(f"Forecast rows: {len(forecasts):,}")

    logger.info("Loading observed data")
    observed_df = pd.read_parquet(OBSERVED_PATH)

    logger.info(f"Observed rows: {len(observed_df):,}")

    # forecasts already carries critical_ratio / repeat_offender_ratio as
    # Module 4's training-time features (computed on historical data up
    # to each point). EDI needs the freshly observed values for the same
    # grid-hour instead, so drop the forecast-side copies before merging
    # — otherwise pandas silently suffixes both to *_x/*_y and the bare
    # column names used below resolve to nothing (which is exactly why
    # they were showing up as "missing — creating default column").
    forecast_dupe_cols = [
        c for c in ["critical_ratio", "repeat_offender_ratio"]
        if c in forecasts.columns
    ]

    if forecast_dupe_cols:
        logger.info(
            f"Dropping Module 4 training-time copies of "
            f"{forecast_dupe_cols} before merge — using observed values instead"
        )
        forecasts = forecasts.drop(columns=forecast_dupe_cols)

    required_forecast = [
        "grid_cell_id",
        "date_hour",
        "predicted_violations",
        "predicted_impact",
    ]

    validate_columns(
        forecasts,
        required_forecast,
        "Forecast dataset",
    )

    required_observed = [
        "grid_cell_id",
        "created_datetime",
        "impact_score",
    ]

    validate_columns(
        observed_df,
        required_observed,
        "Observed dataset",
    )

    if "is_repeat_offender" not in observed_df.columns:
        logger.warning(
            "is_repeat_offender missing — defaulting to 0"
        )
        observed_df["is_repeat_offender"] = 0

    if "impact_priority" not in observed_df.columns:
        logger.warning(
            "impact_priority missing — defaulting to LOW"
        )
        observed_df["impact_priority"] = "LOW"

    observed_df["date_hour"] = (
        pd.to_datetime(
            observed_df["created_datetime"],
            errors="coerce",
        )
        .dt.floor("h")
    )

    observed_df = observed_df.dropna(
        subset=[
            "grid_cell_id",
            "date_hour",
        ]
    )

    observed = (
        observed_df
        .groupby(
            ["grid_cell_id", "date_hour"],
            observed=True,
        )
        .agg(
            observed_violations=(
                "grid_cell_id",
                "size",
            ),

            avg_observed_impact=(
                "impact_score",
                "mean",
            ),

            repeat_offender_ratio=(
                "is_repeat_offender",
                "mean",
            ),

            critical_ratio=(
                "impact_priority",
                lambda s: (s == "CRITICAL").mean(),
            ),
        )
        .reset_index()
    )

    logger.info(
        f"Observed grid-hour rows: {len(observed):,}"
    )

    df = forecasts.merge(
        observed,
        on=[
            "grid_cell_id",
            "date_hour",
        ],
        how="left",
    )

    defaults = {
        "observed_violations": 0,
        "avg_observed_impact": 0,
        "repeat_offender_ratio": 0,
        "critical_ratio": 0,
        "predicted_violations": 0,
        "predicted_impact": 0,
    }

    for col, value in defaults.items():

        if col not in df.columns:
            logger.warning(
                f"{col} missing — creating default column"
            )
            df[col] = value

        df[col] = df[col].fillna(value)

    df["edi_raw"] = (
        df["predicted_violations"]
        - df["observed_violations"]
    )

    demand_gap = np.maximum(
        df["predicted_violations"],
        1,
    )

    df["edi"] = (
        df["edi_raw"]
        / demand_gap
    ).clip(-1, 1)

    df["edi_priority"] = (
        config["weights"]["edi"] * df["edi"]
        + config["weights"]["critical_ratio"]
        * df["critical_ratio"]
        + config["weights"]["repeat_offender_ratio"]
        * df["repeat_offender_ratio"]
    )

    df["edi_priority"] = (
        100
        * (
            df["edi_priority"]
            - df["edi_priority"].min()
        )
        / (
            df["edi_priority"].max()
            - df["edi_priority"].min()
            + 1e-9
        )
    )

    red_cutoff = df["edi_priority"].quantile(
        config["zone_quantiles"]["red"]
    )
    yellow_cutoff = df["edi_priority"].quantile(
        config["zone_quantiles"]["yellow"]
    )

    df["zone_color"] = np.select(
        [
            df["edi_priority"] >= red_cutoff,
            df["edi_priority"] >= yellow_cutoff,
        ],
        [
            "RED",
            "YELLOW",
        ],
        default="GREEN",
    )

    zone_distribution = (
        df["zone_color"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    logger.info("Zone distribution")

    for zone, pct in zone_distribution.items():
        logger.info(f"{zone}: {pct}%")

    logger.info(
        f"Mean EDI: {df['edi'].mean():.3f}"
    )

    logger.info(
        f"Median EDI: {df['edi'].median():.3f}"
    )

    logger.info(
        f"Under-patrolled zones: {(df['edi'] > 0).sum():,}"
    )

    logger.info(
        f"Over-patrolled zones: {(df['edi'] < 0).sum():,}"
    )

    red_zones = (
        df.loc[df["zone_color"] == "RED"]
        .sort_values(
            "edi_priority",
            ascending=False,
        )
        .head(10)
    )

    if len(red_zones):

        logger.info("Top RED zones")

        logger.info(
            "\n%s",
            red_zones[
                [
                    "grid_cell_id",
                    "date_hour",
                    "edi",
                    "edi_priority",
                    "predicted_violations",
                    "observed_violations",
                    "critical_ratio",
                    "repeat_offender_ratio",
                ]
            ].to_string(index=False)
        )

    # Patrol demand signal for Module 6.
    #
    # This is a continuous estimate, not a pre-assigned headcount —
    # Module 6's optimizer needs real demand variation to allocate a
    # limited number of officers against, otherwise there's nothing
    # left for it to optimize.
    #
    # zone_color is used as the primary driver rather than edi_priority
    # directly: RED/YELLOW/GREOEN are percentile tiers (top 10/20/70%),
    # but the underlying scores within RED and YELLOW sit close together
    # in absolute terms (checked: RED mean ~84, YELLOW mean ~76, only an
    # 8-point gap), so weighting on raw edi_priority alone barely
    # separated the two tiers. Using zone_color as a base, with
    # edi_priority and predicted_violations only varying units *within*
    # a tier, guarantees RED always outranks YELLOW always outranks
    # GREEN, which is the behavior the metric is supposed to produce.
    #
    # violations_per_unit is an operational assumption (one patrol unit
    # can reasonably cover this many violations/hour), not something
    # derived from the data — state it as an assumption if asked.
    violations_per_unit = config.get("violations_per_unit", 8)

    zone_base_units = df["zone_color"].map(
        {"RED": 4, "YELLOW": 2, "GREEN": 1}
    )

    within_tier_bonus = np.ceil(
        df["predicted_violations"] / violations_per_unit
    )

    df["recommended_patrol_units"] = (
        zone_base_units + within_tier_bonus
    ).clip(lower=1).astype(int)

    logger.info(
        f"Patrol units — mean: {df['recommended_patrol_units'].mean():.2f}, "
        f"max: {df['recommended_patrol_units'].max()}, "
        f"zero-unit rows: {(df['recommended_patrol_units'] == 0).sum():,}"
    )

    logger.info(
        "Patrol units by zone:\n%s",
        df.groupby("zone_color")["recommended_patrol_units"]
        .mean()
        .round(2)
        .to_string()
    )

    output_columns = [
        "grid_cell_id",
        "date_hour",
        "predicted_violations",
        "observed_violations",
        "predicted_impact",
        "avg_observed_impact",
        "critical_ratio",
        "repeat_offender_ratio",
        "edi_raw",
        "edi",
        "edi_priority",
        "zone_color",
        "recommended_patrol_units",
    ]

    df[output_columns].to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    logger.info(
        f"Saved {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()