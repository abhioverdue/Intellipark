from fastapi import APIRouter, Query
from typing import Literal

from .. import config, loaders
from ..schemas import SimulatorDefaults, SimulatorBaseResult

router = APIRouter()


@router.get("/simulator-defaults", response_model=SimulatorDefaults)
def get_simulator_defaults(
    mode: Literal["historical_gap", "future_forecast"] = Query("future_forecast"),
):
    opt_config = loaders.read_json(
        config.M6_CONFIG_PATH, "module6_optimizer/optimizer.py (needs its config.json)"
    )

    alloc_path = config.allocations_path_for_mode(mode)
    alloc_data = loaders.read_json(alloc_path, "module6_optimizer/optimizer.py")
    summary = alloc_data["summary"]

    covered = int(summary["unique_locations_covered"])
    eligible = int(summary["total_eligible_locations"]) or 1

    return SimulatorDefaults(
        officerCount=int(opt_config["default_officer_count"]),
        demandWeight=float(opt_config.get("demand_weight", 0.25)),
        flowWeight=float(opt_config.get("flow_weight", 0.15)),
        maxOfficersPerLocation=int(opt_config.get("max_officers_per_location", 2)),
        minimumPriorityThreshold=float(opt_config.get("minimum_edi_priority", 40)),
        baseResult=SimulatorBaseResult(
            locationsCovered=covered,
            totalPriorityScore=float(summary["total_expected_priority_score"]),
            coverageUtilizationPct=round(covered / eligible * 100, 1),
        ),
    )
