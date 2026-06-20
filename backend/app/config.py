"""
Central path configuration for the IntelliPark backend.

Mirrors the exact `ROOT = Path(__file__).resolve().parents[N]` convention
already used in every module*_*.py script, so this file is the ONE place
that knows where backend/ sits relative to module1_pipeline ... module6_optimizer.

backend/app/config.py -> parents[0]=backend/app, [1]=backend, [2]=ROOT
"""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]

# Allow overriding via env var (e.g. if the pipeline lives somewhere else),
# but default to the sibling-folder convention every module script assumes.
PROJECT_ROOT = Path(os.getenv("INTELLIPARK_ROOT", str(ROOT)))

# --- Module 1: cleaning ---------------------------------------------------
M1_DIR = PROJECT_ROOT / "module1_pipeline" / "output"
CLEANED_PATH = M1_DIR / "cleaned.parquet"
CLEAN_QUALITY_PATH = M1_DIR / "clean_quality.json"
FEATURED_PATH = M1_DIR / "featured.parquet"
FEATURE_QUALITY_PATH = M1_DIR / "feature_quality.json"
DEPLOY_LOOKUPS_PATH = M1_DIR / "deploy_lookups.json"

# --- Module 2: impact score ------------------------------------------------
M2_DIR = PROJECT_ROOT / "module2_impact_score" / "output"
SCORED_PATH = M2_DIR / "scored.parquet"
BREAKDOWN_SHAPES_PATH = M2_DIR / "breakdown_shapes.json"

# --- Module 3: repeat offenders --------------------------------------------
M3_DIR = PROJECT_ROOT / "module3_repeat_offender" / "output"
VEHICLE_RISK_PATH = M3_DIR / "vehicle_risk.parquet"
RISK_TAGGED_PATH = M3_DIR / "risk_tagged.parquet"

# --- Module 4: hotspot forecast --------------------------------------------
M4_DIR = PROJECT_ROOT / "module4_hotspot_forecast" / "output"
FEATURE_IMPORTANCE_PATH = M4_DIR / "feature_importance.csv"

# --- Module 5: EDI / priority ----------------------------------------------
M5_DIR = PROJECT_ROOT / "module5_edi" / "output"
EDI_WITH_FLOW_PATH = M5_DIR / "edi_scores_with_flow.parquet"
FUTURE_PRIORITY_WITH_FLOW_PATH = M5_DIR / "future_priority_with_flow.parquet"

# --- Module 6: optimizer ----------------------------------------------------
M6_DIR = PROJECT_ROOT / "module6_optimizer"
M6_CONFIG_PATH = M6_DIR / "config.json"
M6_OUTPUT_DIR = M6_DIR / "output"
ALLOCATIONS_HISTORICAL_PATH = M6_OUTPUT_DIR / "allocations_historical_gap.json"
ALLOCATIONS_FUTURE_PATH = M6_OUTPUT_DIR / "allocations_future_forecast.json"

VALID_MODES = ("historical_gap", "future_forecast")


def allocations_path_for_mode(mode: str) -> Path:
    return (
        ALLOCATIONS_FUTURE_PATH
        if mode == "future_forecast"
        else ALLOCATIONS_HISTORICAL_PATH
    )


def hotspot_path_for_mode(mode: str) -> Path:
    return (
        FUTURE_PRIORITY_WITH_FLOW_PATH
        if mode == "future_forecast"
        else EDI_WITH_FLOW_PATH
    )


def priority_col_for_mode(mode: str) -> str:
    return "priority_score" if mode == "future_forecast" else "edi_priority"


# CORS — the Vite dev server's default origin, plus localhost variants.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "INTELLIPARK_CORS_ORIGINS",
        "https://intellipark-ufw7.onrender.com/",
    ).split(",")
    if o.strip()
]
