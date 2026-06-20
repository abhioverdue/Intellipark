from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Query

from .. import config, loaders
from ..schemas import LimitationsResponse, FeatureImportanceRow, DataQualityMetric

router = APIRouter()

TOP_N_FEATURES_PER_MODEL = 8


@lru_cache(maxsize=2)
def _cleaned_summary_cached(path_str: str, mtime: float) -> dict:
    df = pd.read_parquet(
        path_str, columns=["grid_cell_id", "junction_name", "created_datetime"]
    )
    junction_named = (~df["junction_name"].isin(["UNKNOWN", "No Junction"])).mean() * 100
    dt = pd.to_datetime(df["created_datetime"])
    return {
        "unique_grid_cells": int(df["grid_cell_id"].nunique()),
        "junction_named_pct": round(float(junction_named), 1),
        "coverage_start": dt.min(),
        "coverage_end": dt.max(),
    }


def _cleaned_summary() -> dict:
    deploy_data = loaders.read_json_optional(config.DEPLOY_LOOKUPS_PATH, {})
    if "cleaned_summary" in deploy_data:
        s = deploy_data["cleaned_summary"]
        return {
            "unique_grid_cells": s["unique_grid_cells"],
            "junction_named_pct": s["junction_named_pct"],
            "coverage_start": pd.to_datetime(s["coverage_start"]),
            "coverage_end": pd.to_datetime(s["coverage_end"]),
        }

    path = config.CLEANED_PATH
    loaders._require(path, "module1_pipeline/clean.py")
    return _cleaned_summary_cached(str(path), loaders._mtime(path))


@router.get("/limitations", response_model=LimitationsResponse)
def get_limitations(
    mode: Literal["historical_gap", "future_forecast"] = Query("historical_gap"),
):
    fi_df = loaders.read_csv(
        config.FEATURE_IMPORTANCE_PATH, "module4_hotspot_forecast/train_model.py"
    )

    top_rows = (
        fi_df.sort_values("importance", ascending=False)
        .groupby("model", sort=False)
        .head(TOP_N_FEATURES_PER_MODEL)
    )

    feature_importance = [
        FeatureImportanceRow(
            feature=row["feature"],
            importance=round(float(row["importance"]), 4),
            model=row["model"],
        )
        for row in top_rows.to_dict(orient="records")
    ]

    priority_col = config.priority_col_for_mode(mode)
    hotspot_df = loaders.read_parquet(
        config.hotspot_path_for_mode(mode),
        "module5_edi/flow_proxy.py",
    )
    scores = hotspot_df[priority_col]

    summary = _cleaned_summary()
    coverage = (
        f"{summary['coverage_start'].strftime('%b %Y')} – "
        f"{summary['coverage_end'].strftime('%b %Y')}"
    )

    data_quality = [
        DataQualityMetric(label="Mean priority score", value=f"{scores.mean():.1f}"),
        DataQualityMetric(label="Median priority score", value=f"{scores.median():.1f}"),
        DataQualityMetric(label="Max priority score", value=f"{scores.max():.1f}"),
        DataQualityMetric(
            label="Grid cells with named junction",
            value=f"~{summary['junction_named_pct']:.0f}%",
        ),
        DataQualityMetric(label="Dataset coverage", value=coverage),
        DataQualityMetric(
            label="Unique grid cells", value=str(summary["unique_grid_cells"])
        ),
    ]

    return LimitationsResponse(
        featureImportance=feature_importance,
        dataQuality=data_quality,
    )
