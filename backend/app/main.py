"""
IntelliPark backend.

Run with:
    uvicorn app.main:app --reload --port 8000

from inside the backend/ directory. Endpoints are served under /api to
match VITE_API_BASE=http://localhost:8000/api in the dashboard's .env.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .routers import hotspots, allocations, repeat_offenders, simulator, limitations

app = FastAPI(
    title="IntelliPark API",
    description="Bridges module1_pipeline ... module6_optimizer outputs to the dashboard.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(hotspots.router, prefix="/api")
app.include_router(allocations.router, prefix="/api")
app.include_router(repeat_offenders.router, prefix="/api")
app.include_router(simulator.router, prefix="/api")
app.include_router(limitations.router, prefix="/api")


@app.get("/health")
def health():
    """Quick check of which pipeline outputs currently exist, so you can
    see at a glance what's missing before poking the dashboard.

    cleaned.parquet/scored.parquet are satisfied by EITHER the full
    file OR its slim deploy-lookup equivalent (deploy_lookups.json /
    breakdown_shapes.json from build_deploy_lookups.py) -- a slim
    deployment intentionally ships only the latter."""
    checks = {
        "cleaned.parquet (or deploy_lookups.json)": (
            config.CLEANED_PATH.exists() or config.DEPLOY_LOOKUPS_PATH.exists()
        ),
        "scored.parquet (or breakdown_shapes.json)": (
            config.SCORED_PATH.exists() or config.BREAKDOWN_SHAPES_PATH.exists()
        ),
        "vehicle_risk.parquet": config.VEHICLE_RISK_PATH.exists(),
        "feature_importance.csv": config.FEATURE_IMPORTANCE_PATH.exists(),
        "edi_scores_with_flow.parquet": config.EDI_WITH_FLOW_PATH.exists(),
        "future_priority_with_flow.parquet": config.FUTURE_PRIORITY_WITH_FLOW_PATH.exists(),
        "module6_optimizer/config.json": config.M6_CONFIG_PATH.exists(),
        "allocations_historical_gap.json": config.ALLOCATIONS_HISTORICAL_PATH.exists(),
        "allocations_future_forecast.json": config.ALLOCATIONS_FUTURE_PATH.exists(),
    }
    return {
        "status": "ok" if all(checks.values()) else "incomplete",
        "project_root": str(config.PROJECT_ROOT),
        "outputs": checks,
    }
