"""
Cached loaders for the pipeline's parquet/json/csv outputs, plus a few
shared helpers (grid_cell_id -> lat/lng, grid_cell_id -> junction_name,
module-2 score-breakdown shapes).

Caching strategy: lru_cache keyed on (path, mtime). Whenever you re-run
the pipeline, the file's mtime changes, the cache key changes, and the
next request reads fresh data automatically — no manual cache-busting,
no restart required, but repeated requests against unchanged files don't
re-hit disk every time.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json

import pandas as pd
from fastapi import HTTPException

from . import config


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return -1.0


def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Required pipeline output not found: {path}. "
                f"Run {hint} first (see run_pipeline.py)."
            ),
        )
    return path


@lru_cache(maxsize=64)
def _read_parquet_cached(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path_str)


def read_parquet(path: Path, hint: str) -> pd.DataFrame:
    _require(path, hint)
    return _read_parquet_cached(str(path), _mtime(path)).copy()


@lru_cache(maxsize=64)
def _read_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path_str)


def read_csv(path: Path, hint: str) -> pd.DataFrame:
    _require(path, hint)
    return _read_csv_cached(str(path), _mtime(path)).copy()


@lru_cache(maxsize=64)
def _read_json_cached(path_str: str, mtime: float) -> dict:
    with open(path_str) as f:
        return json.load(f)


def read_json(path: Path, hint: str) -> dict:
    _require(path, hint)
    return _read_json_cached(str(path), _mtime(path))


def read_json_optional(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Shared derived lookups
# ---------------------------------------------------------------------------

def parse_grid_cell_id(grid_cell_id: str) -> tuple[float, float]:
    """grid_cell_id is created in clean.py as f'{lat_grid}_{lon_grid}'
    (round(lat, 2), round(lon, 2)) — so it IS the coordinate, no separate
    geocode table needed."""
    try:
        lat_str, lon_str = str(grid_cell_id).split("_", 1)
        return float(lat_str), float(lon_str)
    except (ValueError, AttributeError):
        return 0.0, 0.0


@lru_cache(maxsize=4)
def _junction_lookup_cached(path_str: str, mtime: float) -> dict[str, str]:
    df = pd.read_parquet(path_str, columns=["grid_cell_id", "junction_name"])
    grouped = (
        df.groupby("grid_cell_id", observed=True)["junction_name"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    return grouped.to_dict()


def junction_lookup() -> dict[str, str]:
    """grid_cell_id -> representative junction_name, same logic as
    module5_edi/location_context.py's grid_junction aggregation.

    Prefers the precomputed deploy_lookups.json (built by
    build_deploy_lookups.py) when present, so a deployment doesn't
    need to ship the full cleaned.parquet just for this. Falls back
    to deriving it live from cleaned.parquet for local dev."""
    deploy_data = read_json_optional(config.DEPLOY_LOOKUPS_PATH, {})
    if "junction_by_grid" in deploy_data:
        return deploy_data["junction_by_grid"]

    path = config.CLEANED_PATH
    _require(path, "module1_pipeline/clean.py")
    return _junction_lookup_cached(str(path), _mtime(path))


SCORE_COMPONENT_COLS = [
    "impact_junction",
    "impact_severity",
    "impact_vehicle",
    "impact_peak",
    "impact_hotspot",
    "impact_repeat",
]

SCORE_COMPONENT_TO_BREAKDOWN_KEY = {
    "impact_junction": "junctionType",
    "impact_severity": "violationSeverity",
    "impact_vehicle": "vehicleSize",
    "impact_peak": "peakHour",
    "impact_hotspot": "hotspotDensity",
    "impact_repeat": "repeatOffenders",
}


@lru_cache(maxsize=4)
def _breakdown_shape_cached(path_str: str, mtime: float) -> dict[str, dict]:
    """Per-grid_cell_id mean of module2's six impact_* components,
    normalized to shares that sum to 1. These are the per-violation
    score components from scorer.py — module5's priority score is a
    *different* formula (EDI / forecast-priority), so the breakdown
    is reported as "this grid cell's historical mix of what drives
    impact", rescaled at request time to sum to that grid-hour's
    actual priorityScore. It's a representative breakdown, not a
    literal decomposition of the EDI formula — documented here so
    that's clear to whoever wires the chart tooltip copy."""
    df = pd.read_parquet(path_str, columns=["grid_cell_id"] + SCORE_COMPONENT_COLS)
    means = df.groupby("grid_cell_id", observed=True)[SCORE_COMPONENT_COLS].mean()
    row_sums = means.sum(axis=1).replace(0, 1.0)
    shares = means.div(row_sums, axis=0)
    return shares.to_dict(orient="index")


def breakdown_shape_lookup() -> dict[str, dict]:
    """Prefers the precomputed breakdown_shapes.json over deriving
    from the full scored.parquet, for the same deploy-size reason as
    junction_lookup() above."""
    deploy_shapes = read_json_optional(config.BREAKDOWN_SHAPES_PATH, None)
    if deploy_shapes is not None:
        return deploy_shapes

    path = config.SCORED_PATH
    _require(path, "module2_impact_score/scorer.py")
    return _breakdown_shape_cached(str(path), _mtime(path))


def score_breakdown_for(grid_cell_id: str, priority_score: float, shapes: dict) -> dict:
    shape = shapes.get(grid_cell_id)
    if shape is None:
        # No per-violation history for this cell (shouldn't normally
        # happen since hotspots come from grid cells that have
        # violations) — split evenly rather than guessing a skew.
        even = priority_score / len(SCORE_COMPONENT_COLS)
        return {key: round(even, 2) for key in SCORE_COMPONENT_TO_BREAKDOWN_KEY.values()}

    return {
        SCORE_COMPONENT_TO_BREAKDOWN_KEY[col]: round(shape[col] * priority_score, 2)
        for col in SCORE_COMPONENT_COLS
    }
