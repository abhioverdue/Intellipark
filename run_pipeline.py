"""
Runs the full IntelliPark pipeline, modules 1 through 6, in the order
each module's output requires.

Why this exists (not just "convenience"):

location_context.py and flow_proxy.py each call get_deployment_mode(),
which reads module6_optimizer/config.json ONCE per run and produces
ONLY that mode's output file. Unlike optimizer.py (which loops over
both modes internally), those two scripts do not. Run them once and
you get exactly one of {edi_scores_with_flow.parquet,
future_priority_with_flow.parquet} on disk — never both. The backend's
/hotspots and /limitations endpoints need both to exist for the
dashboard's mode toggle to work, so this script runs that pair TWICE,
flipping deployment_mode in config.json between runs, and restores the
original value afterward so nothing else is left in a surprising state.

Usage:
    python run_pipeline.py                 # run everything
    python run_pipeline.py --from edi      # resume from module5 onward
    python run_pipeline.py --skip-train    # reuse existing trained models
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

M6_CONFIG_PATH = ROOT / "module6_optimizer" / "config.json"

# (label, script path relative to ROOT)
STEPS = [
    ("clean", "module1_pipeline/clean.py"),
    ("features", "module1_pipeline/features.py"),
    ("score", "module2_impact_score/scorer.py"),
    ("vehicle_risk", "module3_repeat_offender/vehicle_risk.py"),
    ("grid_features", "module4_hotspot_forecast/grid_features.py"),
    ("train", "module4_hotspot_forecast/train_model.py"),
    ("shap_explain", "module4_hotspot_forecast/shap_explain.py"),
    ("forecast_future", "module4_hotspot_forecast/forecast_future.py"),
    ("edi", "module5_edi/edi.py"),
    ("future_priority", "module5_edi/future_priority.py"),
    # location_context + flow_proxy run twice (see module docstring above)
    ("optimizer", "module6_optimizer/optimizer.py"),
]

CONTEXT_AND_FLOW = ("module5_edi/location_context.py", "module5_edi/flow_proxy.py")


def run_script(rel_path: str):
    script = ROOT / rel_path
    logger.info(f"--- Running {rel_path} ---")
    result = subprocess.run([sys.executable, str(script)], cwd=str(script.parent))
    if result.returncode != 0:
        raise RuntimeError(f"{rel_path} failed (exit code {result.returncode})")


def set_deployment_mode(mode: str):
    with open(M6_CONFIG_PATH) as f:
        config = json.load(f)
    config["deployment_mode"] = mode
    with open(M6_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"module6_optimizer/config.json deployment_mode -> {mode}")


def get_deployment_mode() -> str:
    with open(M6_CONFIG_PATH) as f:
        return json.load(f).get("deployment_mode", "historical_gap")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from",
        dest="from_step",
        default=None,
        choices=[s[0] for s in STEPS],
        help="Resume from this step onward (skips everything before it).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip module4's train (and grid_features) — reuse existing model files.",
    )
    args = parser.parse_args()

    steps = STEPS
    if args.from_step:
        start = next(i for i, s in enumerate(steps) if s[0] == args.from_step)
        steps = steps[start:]

    if args.skip_train:
        steps = [s for s in steps if s[0] not in ("grid_features", "train")]

    original_mode = get_deployment_mode()

    for label, rel_path in steps:
        run_script(rel_path)

        if label == "future_priority":
            # Both mode-dependent location/flow files need to exist before
            # optimizer.py runs (it reads both unconditionally).
            for mode in ("historical_gap", "future_forecast"):
                set_deployment_mode(mode)
                for ctx_script in CONTEXT_AND_FLOW:
                    run_script(ctx_script)

    set_deployment_mode(original_mode)

    logger.info("--- Building slim deploy lookups ---")
    run_script("backend/build_deploy_lookups.py")

    logger.info("=== Pipeline complete ===")
    logger.info(
        "Start the backend now: cd backend && uvicorn app.main:app --reload --port 8000"
    )
    logger.info("Then check http://localhost:8000/health to confirm every output exists.")


if __name__ == "__main__":
    main()
