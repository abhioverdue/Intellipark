# backend/app/routers/shap.py
#
# Serves shap_explain.py's output (module4_hotspot_forecast/output/
# shap_top_features.parquet) so the dashboard can show "why is this
# zone flagged" -- the top-3 features driving the violation-count and
# impact-score predictions for a given grid cell.
#
# New file, new endpoint -- does not modify hotspots.py or any existing
# router. Registered in main.py alongside the other routers.

from typing import Literal, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import loaders, shap_paths

router = APIRouter()


class ShapContributor(BaseModel):
    feature: str
    shapValue: float


class ShapExplanation(BaseModel):
    gridCellId: str
    dateHour: str
    model: Literal["violation_count", "impact_score"]
    baseValue: float
    predictedValue: float
    topContributors: list[ShapContributor]


class ShapExplanationResponse(BaseModel):
    gridCellId: str
    explanations: list[ShapExplanation]


def _row_to_explanation(row: dict) -> ShapExplanation:
    contributors = []
    for rank in (1, 2, 3):
        feature = row.get(f"top{rank}_feature")
        value = row.get(f"top{rank}_shap")
        if feature is None or pd.isna(feature):
            continue
        contributors.append(ShapContributor(feature=feature, shapValue=float(value)))

    return ShapExplanation(
        gridCellId=row["grid_cell_id"],
        dateHour=pd.Timestamp(row["date_hour"]).isoformat(),
        model=row["model"],
        baseValue=float(row["base_value"]),
        predictedValue=float(row["predicted_value"]),
        topContributors=contributors,
    )


@router.get("/shap/{grid_cell_id}", response_model=ShapExplanationResponse)
def get_shap_explanation(
    grid_cell_id: str,
    model: Optional[Literal["violation_count", "impact_score"]] = None,
):
    """Returns the most recent SHAP explanation(s) for a grid cell.

    By default returns both models (violation_count and impact_score).
    Pass ?model=violation_count or ?model=impact_score to filter to one.
    """
    hint = (
        "module4_hotspot_forecast/shap_explain.py (after "
        "module4_hotspot_forecast/train_model.py has produced "
        "count_model.pkl / impact_model.pkl)"
    )

    df = loaders.read_parquet(shap_paths.SHAP_TOP_FEATURES_PATH, hint)

    df = df[df["grid_cell_id"] == grid_cell_id]

    if model is not None:
        df = df[df["model"] == model]

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No SHAP explanation found for grid_cell_id={grid_cell_id!r}.",
        )

    df["date_hour"] = pd.to_datetime(df["date_hour"])

    latest = (
        df.sort_values("date_hour")
        .groupby("model", observed=True)
        .tail(1)
    )

    explanations = [
        _row_to_explanation(row) for row in latest.to_dict(orient="records")
    ]

    return ShapExplanationResponse(
        gridCellId=grid_cell_id,
        explanations=explanations,
    )
