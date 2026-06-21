# module4_hotspot_forecast/shap_explain.py
#
# Adds per-prediction explainability on top of the existing
# count_model.pkl / impact_model.pkl trained by train_model.py.
#
# train_model.py already saves a GLOBAL feature_importance.csv (average
# importance across every row). This script adds LOCAL explanations --
# for one specific grid_cell_id at one specific date_hour, which features
# pushed the prediction up or down, and by how much.
#
# This is purely additive: it does not modify train_model.py,
# forecast_future.py, or their outputs. It only reads the models and
# training_data.parquet that train_model.py already produces, and writes
# one new file: output/shap_top_features.parquet
#
# Run after train_model.py (or after run_pipeline.py finishes module 4):
#   pip install -r shap-requirements.txt
#   python shap_explain.py

from pathlib import Path
import logging

import joblib
import numpy as np
import pandas as pd

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "shap is not installed. Run "
        "`pip install -r module4_hotspot_forecast/shap-requirements.txt` "
        "before running shap_explain.py."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "module4_hotspot_forecast" / "output"

TRAINING_DATA_PATH = OUTPUT_DIR / "training_data.parquet"
COUNT_MODEL_PATH = OUTPUT_DIR / "count_model.pkl"
IMPACT_MODEL_PATH = OUTPUT_DIR / "impact_model.pkl"

SHAP_TOP_FEATURES_PATH = OUTPUT_DIR / "shap_top_features.parquet"

# Same feature lists as train_model.py / forecast_future.py, duplicated
# here intentionally so this script has zero import dependency on those
# files and can never accidentally change their behavior.
BASE_FEATURES = [
    "hour", "dow", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "critical_ratio", "repeat_offender_ratio",
    "violation_count_lag_1", "violation_count_lag_24",
    "violation_count_rolling_6", "violation_count_rolling_24",
    "avg_impact_score_lag_1", "avg_impact_score_lag_24",
    "avg_impact_score_rolling_6", "avg_impact_score_rolling_24",
]

OPTIONAL_FEATURES = [
    "violation_count_lag_2", "violation_count_lag_3",
    "violation_count_lag_6", "violation_count_lag_168",
    "avg_impact_score_lag_2", "avg_impact_score_lag_3",
    "avg_impact_score_lag_6", "avg_impact_score_lag_168",
    "violation_count_rolling_12", "avg_impact_score_rolling_12",
    "grid_mean_count", "grid_mean_impact",
]

TOP_N = 3


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def get_features(df: pd.DataFrame) -> list[str]:
    features = BASE_FEATURES.copy()
    for col in OPTIONAL_FEATURES:
        if col in df.columns:
            features.append(col)
    return features


def require(path: Path, hint: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run {hint} first."
        )


def explain_model(model, X: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Returns one row per (grid_cell_id, date_hour) with the top-N
    SHAP contributors for this model's prediction at that row."""

    logger.info(f"Building TreeExplainer for {model_name}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[FEATURES])
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    predictions = model.predict(X[FEATURES])

    feature_names = np.array(FEATURES)

    records = []

    for i in range(len(X)):
        row_shap = shap_values[i]
        order = np.argsort(-np.abs(row_shap))[:TOP_N]

        record = {
            "grid_cell_id": X.iloc[i]["grid_cell_id"],
            "date_hour": X.iloc[i]["date_hour"],
            "model": model_name,
            "base_value": round(base_value, 4),
            "predicted_value": round(float(predictions[i]), 4),
        }

        for rank, idx in enumerate(order, start=1):
            record[f"top{rank}_feature"] = feature_names[idx]
            record[f"top{rank}_shap"] = round(float(row_shap[idx]), 4)

        # Pad if fewer than TOP_N features exist (shouldn't happen here,
        # but keeps the schema stable if FEATURES ever shrinks below 3).
        for rank in range(len(order) + 1, TOP_N + 1):
            record[f"top{rank}_feature"] = None
            record[f"top{rank}_shap"] = None

        records.append(record)

    return pd.DataFrame(records)


def main():
    require(TRAINING_DATA_PATH, "module4_hotspot_forecast/train_model.py")
    require(COUNT_MODEL_PATH, "module4_hotspot_forecast/train_model.py")
    require(IMPACT_MODEL_PATH, "module4_hotspot_forecast/train_model.py")

    logger.info("Loading training data and trained models")

    df = pd.read_parquet(TRAINING_DATA_PATH)
    df["date_hour"] = pd.to_datetime(df["date_hour"])
    df = df.sort_values("date_hour").reset_index(drop=True)

    global FEATURES
    FEATURES = get_features(df)

    logger.info(f"Explaining {len(df):,} rows across {len(FEATURES)} features")

    count_model = joblib.load(COUNT_MODEL_PATH)
    impact_model = joblib.load(IMPACT_MODEL_PATH)

    count_explained = explain_model(count_model, df, "violation_count")
    impact_explained = explain_model(impact_model, df, "impact_score")

    result = pd.concat([count_explained, impact_explained], ignore_index=True)

    result.to_parquet(SHAP_TOP_FEATURES_PATH, index=False)

    logger.info(f"Saved {SHAP_TOP_FEATURES_PATH} ({len(result):,} rows)")
    logger.info(
        "Each grid_cell_id/date_hour now has its top-3 SHAP contributors "
        "for both the violation_count and impact_score models."
    )


if __name__ == "__main__":
    main()
