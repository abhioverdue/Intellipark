from fastapi import APIRouter, Query

from .. import config, loaders
from ..schemas import RepeatOffendersResponse, RiskTierSummary, VehicleRisk

router = APIRouter()

TIER_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


@router.get("/repeat-offenders", response_model=RepeatOffendersResponse)
def get_repeat_offenders(limit: int = Query(50, ge=1, le=1000)):
    df = loaders.read_parquet(
        config.VEHICLE_RISK_PATH, "module3_repeat_offender/vehicle_risk.py"
    )

    total = len(df)
    counts = df["risk_level"].value_counts()

    tier_summary = [
        RiskTierSummary(
            tier=tier,
            count=int(counts.get(tier, 0)),
            percentage=round(float(counts.get(tier, 0)) / total * 100, 1) if total else 0.0,
        )
        for tier in TIER_ORDER
    ]

    critical_df = (
        df[df["risk_level"] == "CRITICAL"]
        .sort_values("future_risk_score", ascending=False)
        .head(limit)
    )

    critical_vehicles = [
        VehicleRisk(
            vehicleNumber=row["vehicle_number"],
            totalViolations=int(row["total_violations"]),
            criticalRatio=round(float(row["critical_ratio"]), 3),
            topGridConcentration=round(float(row["top_grid_concentration"]), 3),
            meanImpactScore=round(float(row["mean_impact_score"]), 1),
            riskLevel=row["risk_level"],
            recommendedAction=row["recommended_action"],
            riskExplanation=row["risk_explanation"],
        )
        for row in critical_df.to_dict(orient="records")
    ]

    return RepeatOffendersResponse(
        tierSummary=tier_summary,
        criticalVehicles=critical_vehicles,
    )
