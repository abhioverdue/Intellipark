from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FUTURE_FORECAST_PATH = (
    ROOT
    / "module4_hotspot_forecast"
    / "output"
    / "future_grid_forecasts.parquet"
)

CONFIG_PATH = ROOT / "module5_edi" / "config.json"

OUTPUT_DIR = ROOT / "module5_edi" / "output"
OUTPUT_PATH = OUTPUT_DIR / "future_priority_scores.parquet"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def min_max_scale(series: pd.Series) -> pd.Series:
    minimum = series.min()
    maximum = series.max()

    if minimum == maximum:
        return pd.Series(0.0, index=series.index)

    return (series - minimum) / (maximum - minimum)


def main():
    """
    EDI proper (module5_edi/edi.py) measures predicted minus observed
    violations — a real enforcement GAP, which only makes sense for
    hours that have already happened and could have been patrolled.

    For genuinely future hours there is no "observed" yet, so a gap
    can't be computed. This script instead produces a forward-looking
    deployment PRIORITY score driven by predicted demand and the same
    critical/repeat-offender risk signals EDI uses — framed honestly
    as a forecast-based risk projection, not a backward enforcement
    gap. Module 6 can consume either this or edi_scores.parquet
    depending on whether it's planning future patrols or analyzing
    historical enforcement gaps.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()

    logger.info("Loading future forecasts")

    df = pd.read_parquet(FUTURE_FORECAST_PATH)

    logger.info(f"Future forecast rows: {len(df):,}")
    logger.info(
        f"Forecast window: {df['date_hour'].min()} to {df['date_hour'].max()}"
    )

    demand_norm = min_max_scale(df["predicted_violations"])
    impact_norm = min_max_scale(df["predicted_impact"])

    df["priority_score"] = (
        config["weights"]["edi"] * demand_norm
        + config["weights"]["critical_ratio"] * df["critical_ratio"]
        + config["weights"]["repeat_offender_ratio"] * df["repeat_offender_ratio"]
        + (1 - sum(config["weights"].values())) * impact_norm
    )

    df["priority_score"] = (
        100
        * (df["priority_score"] - df["priority_score"].min())
        / (df["priority_score"].max() - df["priority_score"].min() + 1e-9)
    )

    red_cutoff = df["priority_score"].quantile(config["zone_quantiles"]["red"])
    yellow_cutoff = df["priority_score"].quantile(config["zone_quantiles"]["yellow"])

    df["zone_color"] = np.select(
        [
            df["priority_score"] >= red_cutoff,
            df["priority_score"] >= yellow_cutoff,
        ],
        ["RED", "YELLOW"],
        default="GREEN",
    )

    zone_dist = df["zone_color"].value_counts(normalize=True).mul(100).round(2)

    logger.info("Zone distribution")
    for zone, pct in zone_dist.items():
        logger.info(f"{zone}: {pct}%")

    violations_per_unit = config.get("violations_per_unit", 8)

    zone_base_units = df["zone_color"].map({"RED": 4, "YELLOW": 2, "GREEN": 1})
    within_tier_bonus = np.ceil(df["predicted_violations"] / violations_per_unit)

    df["recommended_patrol_units"] = (
        zone_base_units + within_tier_bonus
    ).clip(lower=1).astype(int)

    output_columns = [
        "grid_cell_id", "date_hour", "predicted_violations",
        "predicted_impact", "critical_ratio", "repeat_offender_ratio",
        "priority_score", "zone_color", "recommended_patrol_units",
    ]

    df[output_columns].to_parquet(OUTPUT_PATH, index=False)

    logger.info(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()