"""
Precomputes two small grid_cell_id-keyed lookups that the backend
currently derives on the fly from cleaned.parquet (19MB+) and
scored.parquet (20MB+):

  - module1_pipeline/output/deploy_lookups.json
        junction_by_grid: grid_cell_id -> representative junction_name
        cleaned_summary: the handful of scalar stats /api/limitations needs
  - module2_impact_score/output/breakdown_shapes.json
        grid_cell_id -> normalized share of each module-2 score component

Both are ~560 entries (one per grid cell) -- tens of KB total, vs the
~40MB of per-violation data they're derived from. Run this once after
run_pipeline.py finishes, and a deployment only needs to ship these
two small files plus the already-small module5/module6 outputs --
not the full cleaned.parquet / scored.parquet / risk_tagged.parquet /
featured.parquet (~98MB combined).

backend/app/loaders.py automatically prefers these files when present
and falls back to deriving from the full parquet files when they're
not (so nothing changes for local dev -- this is purely a deploy-size
optimization).

Usage:
    python backend/build_deploy_lookups.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CLEANED_PATH = ROOT / "module1_pipeline" / "output" / "cleaned.parquet"
SCORED_PATH = ROOT / "module2_impact_score" / "output" / "scored.parquet"

DEPLOY_LOOKUPS_PATH = ROOT / "module1_pipeline" / "output" / "deploy_lookups.json"
BREAKDOWN_SHAPES_PATH = (
    ROOT / "module2_impact_score" / "output" / "breakdown_shapes.json"
)

SCORE_COMPONENT_COLS = [
    "impact_junction",
    "impact_severity",
    "impact_vehicle",
    "impact_peak",
    "impact_hotspot",
    "impact_repeat",
]


def build_junction_and_summary():
    if not CLEANED_PATH.exists():
        print(f"Skipping: {CLEANED_PATH} not found.")
        return

    df = pd.read_parquet(
        CLEANED_PATH, columns=["grid_cell_id", "junction_name", "created_datetime"]
    )

    junction_by_grid = (
        df.groupby("grid_cell_id", observed=True)["junction_name"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )

    junction_named_pct = (~df["junction_name"].isin(["UNKNOWN", "No Junction"])).mean() * 100
    dt = pd.to_datetime(df["created_datetime"])

    output = {
        "junction_by_grid": junction_by_grid,
        "cleaned_summary": {
            "unique_grid_cells": int(df["grid_cell_id"].nunique()),
            "junction_named_pct": round(float(junction_named_pct), 1),
            "coverage_start": dt.min().isoformat(),
            "coverage_end": dt.max().isoformat(),
        },
    }

    with open(DEPLOY_LOOKUPS_PATH, "w") as f:
        json.dump(output, f)

    print(f"Wrote {DEPLOY_LOOKUPS_PATH} ({len(junction_by_grid)} grid cells)")


def build_breakdown_shapes():
    if not SCORED_PATH.exists():
        print(f"Skipping: {SCORED_PATH} not found.")
        return

    df = pd.read_parquet(SCORED_PATH, columns=["grid_cell_id"] + SCORE_COMPONENT_COLS)
    means = df.groupby("grid_cell_id", observed=True)[SCORE_COMPONENT_COLS].mean()
    row_sums = means.sum(axis=1).replace(0, 1.0)
    shares = means.div(row_sums, axis=0)

    with open(BREAKDOWN_SHAPES_PATH, "w") as f:
        json.dump(shares.to_dict(orient="index"), f)

    print(f"Wrote {BREAKDOWN_SHAPES_PATH} ({len(shares)} grid cells)")


if __name__ == "__main__":
    build_junction_and_summary()
    build_breakdown_shapes()
