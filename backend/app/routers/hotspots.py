from fastapi import APIRouter, Query
from typing import Literal
import pandas as pd

from .. import config, loaders
from ..schemas import HotspotResponse, HotspotLocation, ScoreBreakdown

router = APIRouter()


@router.get("/hotspots", response_model=HotspotResponse)
def get_hotspots(
    mode: Literal["historical_gap", "future_forecast"] = Query("historical_gap"),
    limit: int = Query(300, ge=1, le=5000),
):
    path = config.hotspot_path_for_mode(mode)
    priority_col = config.priority_col_for_mode(mode)

    hint = (
        "module5_edi/location_context.py then module5_edi/flow_proxy.py "
        f"(with deployment_mode='{mode}' in module6_optimizer/config.json "
        "at the time they ran)"
    )
    df = loaders.read_parquet(path, hint)

    df["date_hour"] = pd.to_datetime(df["date_hour"])

    # One representative row per physical location: its most recent
    # known hour, not every grid-hour (the map shows current state per
    # location, not a time series).
    df = df.sort_values("date_hour")
    latest_per_cell = df.groupby("grid_cell_id", observed=True).tail(1)

    data_as_of = df["date_hour"].max()

    latest_per_cell = latest_per_cell.sort_values(priority_col, ascending=False).head(limit)

    shapes = loaders.breakdown_shape_lookup()

    has_observed = "observed_violations" in latest_per_cell.columns

    locations = []
    for row in latest_per_cell.itertuples(index=False):
        r = row._asdict()
        lat, lng = loaders.parse_grid_cell_id(r["grid_cell_id"])
        priority_score = float(r[priority_col])

        locations.append(
            HotspotLocation(
                gridCellId=r["grid_cell_id"],
                junctionName=r.get("junction_name", "UNKNOWN") or "UNKNOWN",
                lat=lat,
                lng=lng,
                zone=r["zone_color"],
                priorityScore=priority_score,
                predictedViolations=float(r.get("predicted_violations", 0) or 0),
                observedViolations=(
                    float(r["observed_violations"])
                    if has_observed and pd.notna(r.get("observed_violations"))
                    else None
                ),
                flowImpact=float(r.get("predicted_flow_impact", 0) or 0),
                recommendedPatrolUnits=int(r.get("recommended_patrol_units", 1) or 1),
                locationContext=r.get("location_context", "UNCLASSIFIED") or "UNCLASSIFIED",
                scoreBreakdown=ScoreBreakdown(
                    **loaders.score_breakdown_for(r["grid_cell_id"], priority_score, shapes)
                ),
            )
        )

    return HotspotResponse(
        mode=mode,
        dataAsOf=data_as_of.isoformat(),
        locations=locations,
    )
