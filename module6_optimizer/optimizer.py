from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EDI_PATH = (
    ROOT
    / "module5_edi"
    / "output"
    / "edi_scores.parquet"
)

FUTURE_PRIORITY_PATH = (
    ROOT
    / "module5_edi"
    / "output"
    / "future_priority_scores.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "module6_optimizer"
    / "output"
)

OUTPUT_PATH = OUTPUT_DIR / "allocations.json"

CONFIG_PATH = (
    ROOT
    / "module6_optimizer"
    / "config.json"
)


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


def compute_base_impact(
    df: pd.DataFrame,
    config: dict,
    priority_col: str = "edi_priority",
) -> pd.Series:
    """
    Score each grid-hour by how much intercepting violations there
    would matter, combining four signals:

    1. priority_col (edi_priority or priority_score) — the core
       Module 5 signal.
    2. predicted_violations (demand) — normalized 0-1.
    3. predicted_flow_impact — the lane-obstruction proxy from
       flow_proxy.py, normalized 0-1. Checked its raw scale first
       (ranges roughly 0-25 in future mode, 0-15 in historical mode)
       before combining — an earlier version of this function
       multiplied predicted_violations raw against a 0-100 priority
       score and let unbounded volume dominate everything, so every
       new signal added here goes through min_max_scale before
       being combined, not used raw.
    4. location_context — zones flagged METRO/COMMERCIAL get a
       small boost (more people/vehicles affected per violation
       intercepted), via context_multipliers in config. Defaults to
       1.0 (no effect) for any context not explicitly listed, so
       adding this signal can't silently zero out locations that
       weren't anticipated when the config was written.

    predicted_violations is normalized (0-1) before combining with
    the priority column (already 0-100). An earlier version
    multiplied the two raw together — since predicted_violations
    ranges up to 133 while the priority score tops out at 100, that
    let a handful of very high-volume but only moderately-prioritized
    GREEN zones dominate every result, drowning out the RED zones EDI
    was built to surface. Normalizing first means priority and demand
    contribute on comparable footing, controlled by demand_weight in
    config.
    """

    zone_multiplier = df["zone_color"].map(
        config["zone_multipliers"]
    ).fillna(1.0)

    context_multipliers = config.get("context_multipliers", {})
    context_multiplier = df["location_context"].map(
        context_multipliers
    ).fillna(1.0) if "location_context" in df.columns else 1.0

    demand_norm = min_max_scale(df["predicted_violations"])

    flow_weight = config.get("flow_weight", 0.0)
    demand_weight = config.get("demand_weight", 0.3)

    if "predicted_flow_impact" in df.columns and flow_weight > 0:
        flow_norm = min_max_scale(df["predicted_flow_impact"])
    else:
        flow_norm = 0.0
        flow_weight = 0.0

    priority_weight = 1 - demand_weight - flow_weight

    if priority_weight < 0:
        raise ValueError(
            f"demand_weight ({demand_weight}) + flow_weight ({flow_weight}) "
            f"exceed 1.0 — reduce one of them in config.json"
        )

    combined_signal = (
        priority_weight * (df[priority_col] / 100)
        + demand_weight * demand_norm
        + flow_weight * flow_norm
    )

    return (
        combined_signal
        * config["intercept_rate"]
        * zone_multiplier
        * context_multiplier
        * 100  # rescale to a 0-100ish interpretable "priority score"
    )


def _build_candidates(
    sub_df: pd.DataFrame,
    effectiveness: list,
    n_slots: int,
) -> pd.DataFrame:
    """One row per (grid-hour, officer slot) candidate, ranked later."""

    base_impact = sub_df["base_impact"].to_numpy()
    grid_ids = sub_df["grid_cell_id"].to_numpy()
    date_hours = sub_df["date_hour"].astype(str).to_numpy()
    zone_colors = sub_df["zone_color"].to_numpy()

    candidates = []
    for slot in range(n_slots):
        eff = effectiveness[slot]
        candidates.append(
            pd.DataFrame(
                {
                    "grid_cell_id": grid_ids,
                    "date_hour": date_hours,
                    "zone_color": zone_colors,
                    "officer_slot": slot + 1,
                    "marginal_gain": base_impact * eff,
                }
            )
        )

    return (
        pd.concat(candidates, ignore_index=True)
        .sort_values("marginal_gain", ascending=False)
        .reset_index(drop=True)
    )


def _greedy_select(
    candidates: pd.DataFrame,
    budget: int,
    max_per_location: int | None,
    location_counts: dict,
) -> list:
    """Walk ranked candidates, accepting up to `budget` slots while
    respecting the running per-location cap in `location_counts`
    (mutated in place so callers can chain multiple passes)."""

    accepted = []
    if budget <= 0:
        return accepted

    for row in candidates.itertuples(index=False):
        if len(accepted) >= budget:
            break

        current = location_counts.get(row.grid_cell_id, 0)
        if max_per_location is not None and current >= max_per_location:
            continue

        location_counts[row.grid_cell_id] = current + 1
        accepted.append(row)

    return accepted


def allocate_officers(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Greedy allocation with diminishing marginal returns per
    grid-hour slot, plus three coverage controls:

    1. max_officers_per_zone — caps officers at a single grid-hour
       (already existed, now actually enforced).

    2. max_officers_per_location — caps total officers a single
       grid_cell_id can absorb across ALL its hours combined. Without
       this, a handful of chronically high-priority cells (checked:
       grid 12.97_77.55 alone showed up across 9 different dates
       spanning Nov 2023-Apr 2024) can absorb 60%+ of the entire
       officer budget, since each of its many high-EDI hours
       out-competes most other locations' best hour.

    3. zone_allocation_quota — caps officers PER ZONE COLOR.

       Without this, the old single-pool global ranking sent 100% of
       every run's officers to RED (confirmed on a real run: 50/50
       officers RED in both historical_gap and future_forecast — 0
       YELLOW, 0 GREEN). That happened because zone_multipliers
       (RED=1.5 vs GREEN=1.0) plus RED's much higher mean base_impact
       (28.4 vs GREEN's 13.3) means literally every RED candidate
       outranks literally every YELLOW/GREEN candidate, and the
       officer budget (50) is tiny next to the RED candidate pool
       (thousands of grid-hours) — per-location capping alone just
       spreads those 50 across many *different* RED streets, it
       doesn't touch the other zones at all.

       Fix: officers are now allocated zone-by-zone against a quota
       (default RED 60% / YELLOW 30% / GREEN 10%, configurable),
       ranked by marginal_gain *within* each zone. RED still gets the
       majority share — that's the intended behavior, illegal parking
       in RED zones really does matter most — but YELLOW and GREEN are
       now guaranteed a non-zero floor instead of being mathematically
       unreachable. If a zone can't fill its quota (e.g. too few
       distinct locations once max_officers_per_location applies), the
       shortfall is topped up from the remaining ranked pool so the
       total officer count always equals available_officers exactly.
    """

    effectiveness = config["officer_effectiveness"]
    max_per_zone = config["max_officers_per_zone"]
    max_per_location = config.get("max_officers_per_location")
    available_officers = config["default_officer_count"]
    zone_quota = config.get(
        "zone_allocation_quota", {"RED": 0.6, "YELLOW": 0.3, "GREEN": 0.1}
    )

    df = df.reset_index(drop=True)
    n_slots = min(max_per_zone, len(effectiveness))

    zone_order = ["RED", "YELLOW", "GREEN"]

    # Largest-remainder rounding so per-zone targets sum exactly to
    # available_officers (straight floor()/round() can under- or
    # over-allocate by a couple of officers).
    raw_targets = {z: zone_quota.get(z, 0) * available_officers for z in zone_order}
    targets = {z: int(np.floor(v)) for z, v in raw_targets.items()}
    leftover = available_officers - sum(targets.values())
    by_remainder = sorted(
        zone_order, key=lambda z: raw_targets[z] - targets[z], reverse=True
    )
    for z in by_remainder[:leftover]:
        targets[z] += 1

    location_counts: dict[str, int] = {}
    accepted_rows = []
    used_keys: list[tuple] = []

    for zone in zone_order:
        zone_df = df.loc[df["zone_color"] == zone]
        if zone_df.empty or targets[zone] <= 0:
            continue

        candidates = _build_candidates(zone_df, effectiveness, n_slots)
        picked = _greedy_select(
            candidates, targets[zone], max_per_location, location_counts
        )
        accepted_rows.extend(picked)
        used_keys.extend(
            (r.grid_cell_id, r.date_hour, r.officer_slot) for r in picked
        )

    # Top-up pass: if any zone fell short of its quota (not enough
    # distinct eligible locations once the per-location cap applies),
    # fill the remaining budget from the full ranked pool across all
    # zones so the total officer count still equals available_officers.
    shortfall = available_officers - len(accepted_rows)
    if shortfall > 0:
        all_candidates = _build_candidates(df, effectiveness, n_slots)
        if used_keys:
            used_df = pd.DataFrame(
                used_keys, columns=["grid_cell_id", "date_hour", "officer_slot"]
            )
            all_candidates = all_candidates.merge(
                used_df,
                on=["grid_cell_id", "date_hour", "officer_slot"],
                how="left",
                indicator=True,
            )
            all_candidates = all_candidates.loc[
                all_candidates["_merge"] == "left_only"
            ].drop(columns="_merge")

        accepted_rows.extend(
            _greedy_select(
                all_candidates, shortfall, max_per_location, location_counts
            )
        )

    selected = pd.DataFrame(accepted_rows)

    allocations = (
        selected.groupby(
            ["grid_cell_id", "date_hour", "zone_color"],
            as_index=False,
        )
        .agg(
            officers_allocated=("officer_slot", "count"),
            expected_priority_score=("marginal_gain", "sum"),
        )
        .sort_values("expected_priority_score", ascending=False)
    )

    return allocations


def run_for_mode(deployment_mode: str, config: dict) -> dict | None:
    """
    Runs the full scoring + allocation pipeline for a single
    deployment mode, writes its output JSON, and returns the summary
    dict (or None if that mode's required input files don't exist
    yet, e.g. someone hasn't run forecast_future.py for
    future_forecast mode — logged as a clear warning rather than
    raising, so main() can still complete successfully for whichever
    mode IS ready).

    This is the same logic that previously lived directly in main()
    when the optimizer only supported one mode per run. Splitting it
    out means generating allocations_historical_gap.json AND
    allocations_future_forecast.json in a single `python optimizer.py`
    invocation, instead of needing to edit deployment_mode in
    config.json and rerun twice — which was the actual cause of
    allocations_historical_gap.json going stale/missing earlier in
    this project, since nobody remembered to flip the mode back.
    """

    available_officers = config["default_officer_count"]

    logger.info(f"--- Running mode: {deployment_mode} ---")
    logger.info(f"Available officers: {available_officers}")

    if deployment_mode == "future_forecast":
        input_path = (
            ROOT / "module5_edi" / "output"
            / "future_priority_with_flow.parquet"
        )
        priority_col = "priority_score"
    else:
        input_path = (
            ROOT / "module5_edi" / "output" / "edi_scores_with_flow.parquet"
        )
        priority_col = "edi_priority"

    if not input_path.exists():
        logger.warning(
            f"Skipping '{deployment_mode}' mode — {input_path.name} not "
            f"found. Run, in order: "
            f"{'module4_hotspot_forecast/forecast_future.py, ' if deployment_mode == 'future_forecast' else ''}"
            f"module5_edi/{'future_priority.py' if deployment_mode == 'future_forecast' else 'edi.py'}, "
            f"module5_edi/location_context.py, module5_edi/flow_proxy.py"
        )
        return None

    df = pd.read_parquet(input_path)

    logger.info(f"Loaded {len(df):,} grid-hours from {input_path.name}")

    missing_cols = [
        c for c in ["location_context", "predicted_flow_impact"]
        if c not in df.columns
    ]
    if missing_cols:
        logger.warning(
            f"Skipping '{deployment_mode}' mode — {missing_cols} missing "
            f"from {input_path.name}. Run module5_edi/location_context.py "
            f"and module5_edi/flow_proxy.py for this mode first."
        )
        return None

    if deployment_mode == "future_forecast":
        logger.info(
            f"Planning window: {df['date_hour'].min()} to "
            f"{df['date_hour'].max()}"
        )

    df = df.loc[
        df[priority_col] >= config["minimum_edi_priority"]
    ].copy()

    logger.info(f"Eligible grid-hours: {len(df):,}")

    if df.empty:
        logger.warning(
            f"Skipping '{deployment_mode}' mode — no grid-hours meet "
            f"minimum_edi_priority. Lower the threshold in config.json "
            f"if this mode should produce results."
        )
        return None

    df["base_impact"] = compute_base_impact(df, config, priority_col)

    logger.info(
        "Base impact score by zone (mean):\n%s",
        df.groupby("zone_color")["base_impact"].mean().round(2).to_string(),
    )

    allocations = allocate_officers(df, config)

    total_score = allocations["expected_priority_score"].sum()

    unique_locations_covered = allocations["grid_cell_id"].nunique()
    total_grid_locations = df["grid_cell_id"].nunique()

    logger.info(f"Allocated officers: {available_officers}")
    logger.info(f"Zones (grid-hours) selected: {len(allocations):,}")
    logger.info(
        f"Unique physical locations covered: {unique_locations_covered} "
        f"of {total_grid_locations} eligible-pool locations"
    )
    logger.info(
        f"Total expected priority score covered: {total_score:.2f}"
    )

    logger.info(
        "Allocations by zone color:\n%s",
        allocations.groupby("zone_color")["officers_allocated"]
        .sum()
        .to_string(),
    )

    logger.info(
        "\n%s",
        allocations.head(10).to_string(index=False),
    )

    summary = {
        "available_officers": int(available_officers),
        "zones_selected": int(len(allocations)),
        "unique_locations_covered": int(unique_locations_covered),
        "total_eligible_locations": int(total_grid_locations),
        "total_expected_priority_score": float(round(total_score, 2)),
        "officers_by_zone_color": (
            allocations.groupby("zone_color")["officers_allocated"]
            .sum()
            .astype(int)
            .to_dict()
        ),
    }

    output = {
        "summary": summary,
        "allocations": allocations.to_dict(orient="records"),
    }

    output_path = OUTPUT_DIR / f"allocations_{deployment_mode}.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved {output_path}")

    return summary


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()

    configured_mode = config.get("deployment_mode", "historical_gap")
    logger.info(
        f"config.json has deployment_mode='{configured_mode}' — "
        f"running BOTH modes regardless, so the dashboard's live "
        f"toggle always has both options available after a single run."
    )

    results = {}
    for deployment_mode in ["historical_gap", "future_forecast"]:
        results[deployment_mode] = run_for_mode(deployment_mode, config)
        logger.info("")  # spacer between modes in the log

    ran = [m for m, r in results.items() if r is not None]
    skipped = [m for m, r in results.items() if r is None]

    logger.info("=== Summary ===")
    logger.info(f"Modes generated: {ran if ran else 'none'}")
    if skipped:
        logger.info(
            f"Modes skipped (missing upstream files — see warnings above): "
            f"{skipped}"
        )

    if not ran:
        raise RuntimeError(
            "Neither mode produced output — run the module5_edi pipeline "
            "scripts for at least one mode before the optimizer."
        )


if __name__ == "__main__":
    main()