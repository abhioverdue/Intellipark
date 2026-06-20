from pathlib import Path
import json
import logging
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

HISTORICAL_INPUT_PATH = (
    ROOT / "module5_edi" / "output" / "edi_scores.parquet"
)

FUTURE_INPUT_PATH = (
    ROOT / "module5_edi" / "output" / "future_priority_scores.parquet"
)

JUNCTION_LOOKUP_PATH = (
    ROOT / "module1_pipeline" / "output" / "cleaned.parquet"
)

OPTIMIZER_CONFIG_PATH = (
    ROOT / "module6_optimizer" / "config.json"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# Keyword groups derived by inspecting the actual 170 junction_name
# values in this dataset (see exploration notes) — not an external
# gazetteer, so this stays within the "use only the provided data"
# constraint. Order matters: first match wins, ordered roughly by
# specificity (METRO before generic TRANSIT, etc).
# Keyword groups derived by inspecting the actual 170 junction_name
# values in this dataset (see exploration notes) — not an external
# gazetteer, so this stays within the "use only the provided data"
# constraint.
#
# Order matters and this took one real bug to get right: almost
# every junction name in this dataset ends in the generic word
# "Junction" or "Circle" regardless of what it actually is (e.g.
# "KR Market Junction", "Sagar Theatre Junction", "Mysore Bank
# Junction" are all genuinely commercial places that happen to also
# contain "junction"). Checking generic transit words first swallowed
# every commercial/institutional location before they were ever
# tested — confirmed when an early version showed 0% COMMERCIAL
# matches despite "market", "theatre", and "bank" clearly appearing
# in the junction names. Specific business-type keywords must be
# checked BEFORE the generic junction/circle fallback.
LOCATION_CONTEXT_KEYWORDS = [
    ("METRO", ["metro station", "metro"]),
    ("COMMERCIAL", [
        "market", "plaza", "mall", "bank", "theatre", "bar",
        "complex", "shopping", "poornima",
    ]),
    ("INSTITUTIONAL", [
        "hospital", "school", "college", "university", "kalyana mantapa",
    ]),
    ("GOVERNMENT", [
        "soudha", "gpo", "police", "cto",
    ]),
    ("TRANSIT_HUB", ["bus stand", "railway", "junction", "circle"]),
]


def tag_location_context(junction_name: str) -> str:
    if not isinstance(junction_name, str) or junction_name == "No Junction" or junction_name == "UNKNOWN":
        return "UNCLASSIFIED"

    name_lower = junction_name.lower()

    for tag, keywords in LOCATION_CONTEXT_KEYWORDS:
        if any(kw in name_lower for kw in keywords):
            return tag

    return "OTHER_JUNCTION"


def get_deployment_mode() -> str:
    """
    Reads the same deployment_mode the optimizer uses, so this script
    and the optimizer always agree on which pipeline (historical gap
    analysis vs future forecast) is currently active, rather than
    needing the mode set in two separate places that can drift apart.
    """
    if OPTIMIZER_CONFIG_PATH.exists():
        with open(OPTIMIZER_CONFIG_PATH) as f:
            config = json.load(f)
        return config.get("deployment_mode", "historical_gap")
    return "historical_gap"


def main():

    mode = get_deployment_mode()

    if mode == "future_forecast":
        input_path = FUTURE_INPUT_PATH
        priority_col = "priority_score"
        output_path = (
            ROOT / "module5_edi" / "output"
            / "future_priority_with_context.parquet"
        )
    else:
        input_path = HISTORICAL_INPUT_PATH
        priority_col = "edi_priority"
        output_path = (
            ROOT / "module5_edi" / "output" / "edi_scores_with_context.parquet"
        )

    logger.info(f"Deployment mode: {mode}")
    logger.info(f"Loading scores from {input_path.name}")

    edi_df = pd.read_parquet(input_path)

    logger.info("Loading junction name lookup")
    junction_df = pd.read_parquet(
        JUNCTION_LOOKUP_PATH,
        columns=["grid_cell_id", "junction_name"],
    )

    # A grid cell can technically span more than one junction_name in
    # the raw data (rounding to ~1km bins). Use the most frequent
    # junction name per grid cell as its representative label —
    # documented here so it's clear this is a simplification, not a
    # precise geocode.
    grid_junction = (
        junction_df.groupby("grid_cell_id", observed=True)["junction_name"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
    )

    logger.info(
        f"Mapped {grid_junction['grid_cell_id'].nunique():,} grid cells "
        f"to representative junction names"
    )

    grid_junction["location_context"] = grid_junction["junction_name"].apply(
        tag_location_context
    )

    df = edi_df.merge(
        grid_junction[["grid_cell_id", "junction_name", "location_context"]],
        on="grid_cell_id",
        how="left",
    )

    df["junction_name"] = df["junction_name"].fillna("UNKNOWN")
    df["location_context"] = df["location_context"].fillna("UNCLASSIFIED")

    context_dist = (
        df["location_context"].value_counts(normalize=True).mul(100).round(2)
    )

    logger.info("Location context distribution (grid-hour rows)")
    for tag, pct in context_dist.items():
        logger.info(f"{tag}: {pct}%")

    # The metric that actually matters for the pitch: among RED zones
    # specifically, what fraction sit near a metro/commercial/
    # institutional area vs an unclassified street segment? This is
    # the number that answers the brief's "near commercial areas,
    # metro stations" framing directly.
    red_df = df[df["zone_color"] == "RED"]

    if len(red_df):
        red_context_dist = (
            red_df["location_context"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )
        logger.info("Location context among RED zones specifically")
        for tag, pct in red_context_dist.items():
            logger.info(f"{tag}: {pct}%")

    df.to_parquet(output_path, index=False)

    logger.info(f"Saved {output_path}")


if __name__ == "__main__":
    main()