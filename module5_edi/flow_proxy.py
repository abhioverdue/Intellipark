from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

HISTORICAL_INPUT_PATH = (
    ROOT / "module5_edi" / "output" / "edi_scores_with_context.parquet"
)

FUTURE_INPUT_PATH = (
    ROOT / "module5_edi" / "output" / "future_priority_with_context.parquet"
)

SCORED_PATH = (
    ROOT / "module2_impact_score" / "output" / "scored.parquet"
)

OPTIMIZER_CONFIG_PATH = (
    ROOT / "module6_optimizer" / "config.json"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def get_deployment_mode() -> str:
    if OPTIMIZER_CONFIG_PATH.exists():
        with open(OPTIMIZER_CONFIG_PATH) as f:
            config = json.load(f)
        return config.get("deployment_mode", "historical_gap")
    return "historical_gap"


# This dataset has no speed, occupancy, or sensor-based traffic flow
# measurement of any kind (checked: columns are violation records
# only — id, lat/lon, vehicle info, timestamps, validation status,
# nothing resembling a flow/speed/volume sensor feed). So this is
# explicitly a PROXY built from violation characteristics that
# plausibly correlate with carriageway obstruction, not a measurement
# of real traffic flow. State this distinction if asked — claiming
# this IS traffic flow data would be a real overstatement.
#
# Not every violation type physically blocks a lane. Behavioral
# violations (seatbelt, fare disputes, refusing hire) happen to share
# this dataset but have zero spatial/obstruction effect — they were
# checked individually against the 17 actual violation labels and
# excluded from the obstruction score entirely (set to 0 lane-width
# impact) rather than defaulting them to some nonzero value, which
# would have silently inflated the proxy with irrelevant violations.
LANE_OBSTRUCTION_WEIGHTS = {
    "DOUBLE PARKING": 1.0,            # blocks an entire second lane
    "PARKING NEAR ROAD CROSSING": 0.9,
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 0.9,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 0.9,
    "PARKING IN A MAIN ROAD": 0.7,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 0.6,
    "PARKING ON FOOTPATH": 0.3,        # obstructs pedestrians, not vehicle lanes
    "PARKING OTHER THAN BUS STOP": 0.5,
    "WRONG PARKING": 0.5,
    "NO PARKING": 0.5,
    "OBSTRUCTING DRIVER": 0.6,
    "H T V PROHIBITED": 0.4,
    # Genuinely non-spatial violations — zero obstruction effect.
    "DEFECTIVE NUMBER PLATE": 0.0,
    "WITHOUT SIDE MIRROR": 0.0,
    "REFUSE TO GO FOR HIRE": 0.0,
    "DEMANDING EXCESS FARE": 0.0,
    "FAIL TO USE SAFETY BELTS": 0.0,
}

# Vehicle size acts as a multiplier on lane-width impact: a truck
# double-parked blocks far more carriageway than a scooter doing the
# same thing. Reuses the same SMALL/MEDIUM/LARGE/XLARGE -> 1-4 scale
# already established in module2_impact_score/config.json.
VEHICLE_SIZE_FLOW_MULTIPLIER = {
    1: 1.0,   # SMALL
    2: 1.3,   # MEDIUM
    3: 1.7,   # LARGE
    4: 2.2,   # XLARGE
}


def main():

    mode = get_deployment_mode()

    if mode == "future_forecast":
        input_path = FUTURE_INPUT_PATH
        output_path = (
            ROOT / "module5_edi" / "output"
            / "future_priority_with_flow.parquet"
        )
    else:
        input_path = HISTORICAL_INPUT_PATH
        output_path = (
            ROOT / "module5_edi" / "output" / "edi_scores_with_flow.parquet"
        )

    logger.info(f"Deployment mode: {mode}")
    logger.info(f"Loading scores with location context from {input_path.name}")
    edi_df = pd.read_parquet(input_path)

    logger.info("Loading per-violation scored data for obstruction weighting")
    scored_df = pd.read_parquet(
        SCORED_PATH,
        columns=[
            "grid_cell_id", "created_datetime", "primary_violation",
            "vehicle_size_score",
        ],
    )

    scored_df["date_hour"] = (
        pd.to_datetime(scored_df["created_datetime"]).dt.floor("h")
    )

    scored_df["obstruction_weight"] = (
        scored_df["primary_violation"]
        .astype(str)
        .map(LANE_OBSTRUCTION_WEIGHTS)
        .fillna(0.0)
    )

    unmapped = scored_df.loc[
        ~scored_df["primary_violation"].astype(str).isin(
            LANE_OBSTRUCTION_WEIGHTS.keys()
        )
    ]
    if len(unmapped):
        logger.warning(
            f"{len(unmapped):,} rows had a primary_violation not in "
            f"the obstruction weight map (defaulted to 0): "
            f"{unmapped['primary_violation'].unique().tolist()}"
        )

    scored_df["size_multiplier"] = (
        scored_df["vehicle_size_score"]
        .map(VEHICLE_SIZE_FLOW_MULTIPLIER)
        .fillna(1.0)
    )

    scored_df["obstruction_score"] = (
        scored_df["obstruction_weight"] * scored_df["size_multiplier"]
    )

    # Build a per-grid-cell obstruction RATE (mean weighted severity
    # per violation, not a raw sum), using all historical violations
    # at that grid cell regardless of hour. This captures "when a
    # violation happens here, how obstructive does it tend to be"
    # as a stable spatial property of the location.
    #
    # First version summed obstruction_score directly from
    # observed_violations in the same hour — but observed_violations
    # is exactly the quantity EDI penalizes for being LOW in RED
    # zones (RED = high predicted demand, low observed activity).
    # That made RED zones mechanically show near-zero obstruction by
    # construction, not because they're actually low-impact — it was
    # re-deriving part of EDI's own definition and contradicting it.
    # Confirmed: RED mean was 0.73 vs GREEN's 3.98, exactly backwards.
    grid_obstruction_rate = (
        scored_df.groupby("grid_cell_id", observed=True)
        .agg(
            mean_obstruction_per_violation=("obstruction_score", "mean"),
            zero_obstruction_violation_pct=(
                "obstruction_weight",
                lambda s: (s == 0).mean() * 100,
            ),
        )
        .reset_index()
    )

    df = edi_df.merge(grid_obstruction_rate, on="grid_cell_id", how="left")

    df["mean_obstruction_per_violation"] = (
        df["mean_obstruction_per_violation"].fillna(0)
    )
    df["zero_obstruction_violation_pct"] = (
        df["zero_obstruction_violation_pct"].fillna(0)
    )

    # Predicted flow impact = forecast demand (predicted_violations,
    # the forward-looking number) weighted by this location's typical
    # obstruction severity. This is the metric that should track
    # alongside RED/priority correctly, since it's driven by demand
    # rather than the under-reporting that defines RED in the first
    # place.
    df["predicted_flow_impact"] = (
        df["predicted_violations"] * df["mean_obstruction_per_violation"]
    )

    priority_col = "priority_score" if mode == "future_forecast" else "edi_priority"

    logger.info(
        f"Mean predicted_flow_impact: "
        f"{df['predicted_flow_impact'].mean():.2f}"
    )
    logger.info(
        f"Correlation between predicted_flow_impact and {priority_col}: "
        f"{df['predicted_flow_impact'].corr(df[priority_col]):.3f}"
    )

    red_mean = df.loc[df["zone_color"] == "RED", "predicted_flow_impact"].mean()
    green_mean = (
        df.loc[df["zone_color"] == "GREEN", "predicted_flow_impact"].mean()
    )
    logger.info(
        f"Mean predicted_flow_impact — RED: {red_mean:.2f}, "
        f"GREEN: {green_mean:.2f}"
    )

    df.to_parquet(output_path, index=False)

    logger.info(f"Saved {output_path}")


if __name__ == "__main__":
    main()
