"""
Generates landing/stats.json from the real pipeline outputs, instead of
hand-typed numbers. Run this any time after run_pipeline.py finishes.

Usage (from the project root):
    python landing/generate_stats.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

EDI_WITH_FLOW_PATH = ROOT / "module5_edi" / "output" / "edi_scores_with_flow.parquet"
FUTURE_PRIORITY_WITH_FLOW_PATH = (
    ROOT / "module5_edi" / "output" / "future_priority_with_flow.parquet"
)
VEHICLE_RISK_PATH = ROOT / "module3_repeat_offender" / "output" / "vehicle_risk.parquet"
ALLOCATIONS_FUTURE_PATH = (
    ROOT / "module6_optimizer" / "output" / "allocations_future_forecast.json"
)
ALLOCATIONS_HISTORICAL_PATH = (
    ROOT / "module6_optimizer" / "output" / "allocations_historical_gap.json"
)

OUTPUT_PATH = Path(__file__).resolve().parent / "stats.json"

# Prefer future_forecast for the landing page's headline numbers (it's
# what config.json defaults to as the "live" deployment mode), falling
# back to historical_gap if that mode hasn't been generated yet.
PREFERRED_MODE_ORDER = [
    (FUTURE_PRIORITY_WITH_FLOW_PATH, ALLOCATIONS_FUTURE_PATH),
    (EDI_WITH_FLOW_PATH, ALLOCATIONS_HISTORICAL_PATH),
]


def main():
    grid_path = alloc_path = None
    for gp, ap in PREFERRED_MODE_ORDER:
        if gp.exists() and ap.exists():
            grid_path, alloc_path = gp, ap
            break

    if grid_path is None:
        print(
            "Neither mode's outputs exist yet. Run run_pipeline.py first "
            "(needs at least one of edi_scores_with_flow.parquet / "
            "future_priority_with_flow.parquet plus its matching "
            "allocations_*.json)."
        )
        return

    grid_df = pd.read_parquet(grid_path)

    with open(alloc_path) as f:
        alloc_data = json.load(f)

    stats = {
        "grid_hours_scored": int(len(grid_df)),
        "red_zones": int((grid_df["zone_color"] == "RED").sum()),
        "officers_allocated": int(alloc_data["summary"]["available_officers"]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    if VEHICLE_RISK_PATH.exists():
        vehicle_df = pd.read_parquet(VEHICLE_RISK_PATH, columns=["risk_level"])
        stats["critical_vehicles"] = int((vehicle_df["risk_level"] == "CRITICAL").sum())
    else:
        print(
            f"Warning: {VEHICLE_RISK_PATH.name} not found — "
            "critical_vehicles omitted (run module3_repeat_offender/vehicle_risk.py)."
        )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote {OUTPUT_PATH}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
