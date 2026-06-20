from fastapi import APIRouter, Query
from typing import Literal

from .. import config, loaders
from ..schemas import AllocationResponse, AllocationSummary, AllocationRow, ZoneOfficers

router = APIRouter()

ZONE_ORDER = ["RED", "YELLOW", "GREEN"]


@router.get("/allocations", response_model=AllocationResponse)
def get_allocations(
    mode: Literal["historical_gap", "future_forecast"] = Query("historical_gap"),
):
    path = config.allocations_path_for_mode(mode)
    data = loaders.read_json(path, f"module6_optimizer/optimizer.py (produces both modes in one run)")

    summary = data["summary"]
    by_zone_raw = summary.get("officers_by_zone_color", {})

    junctions = loaders.junction_lookup()

    allocations = [
        AllocationRow(
            gridCellId=row["grid_cell_id"],
            junctionName=junctions.get(row["grid_cell_id"], "UNKNOWN"),
            zone=row["zone_color"],
            dateHour=str(row["date_hour"]),
            officersAllocated=int(row["officers_allocated"]),
            expectedPriorityScore=float(row["expected_priority_score"]),
        )
        for row in data["allocations"]
    ]

    return AllocationResponse(
        mode=mode,
        summary=AllocationSummary(
            availableOfficers=int(summary["available_officers"]),
            locationsCovered=int(summary["unique_locations_covered"]),
            totalEligibleLocations=int(summary["total_eligible_locations"]),
            totalPriorityScore=float(summary["total_expected_priority_score"]),
        ),
        byZone=[
            ZoneOfficers(zone=zone, officers=int(by_zone_raw.get(zone, 0)))
            for zone in ZONE_ORDER
        ],
        allocations=allocations,
    )
