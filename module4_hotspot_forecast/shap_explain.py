# module4_hotspot_forecast/shap_explain.py

from pathlib import Path
import logging
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# ---------------- CONFIG ---------------- #

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "module4_hotspot_forecast" / "output"

TRAINING_DATA_PATH = OUTPUT_DIR / "training_data.parquet"
COUNT_MODEL_PATH = OUTPUT_DIR / "count_model.pkl"
IMPACT_MODEL_PATH = OUTPUT_DIR / "impact_model.pkl"
OUTPUT_PATH = OUTPUT_DIR / "shap_top_features.parquet"

TOP_N = 3

# ⚠️ IMPORTANT: Windows-safe thread setting
NTHREAD = 6

CHUNK_SIZE = 8000   # prevents RAM spikes / freezes

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

# ---------------- LOGGING ---------------- #

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def get_features(df):
    return BASE_FEATURES + [c for c in OPTIONAL_FEATURES if c in df.columns]


def load_data():
    df = pd.read_parquet(TRAINING_DATA_PATH)
    df["date_hour"] = pd.to_datetime(df["date_hour"])
    return df.sort_values("date_hour").reset_index(drop=True)


def topk_shap(shap_vals, k=3):
    abs_vals = np.abs(shap_vals)
    idx = np.argpartition(-abs_vals, k - 1, axis=1)[:, :k]
    row = np.arange(shap_vals.shape[0])[:, None]

    order = np.argsort(-abs_vals[row, idx], axis=1)
    return idx[row, order]


def explain_model(model, df, features, model_name):

    logger.info(f"Starting SHAP for {model_name}")

    booster = model.get_booster()
    booster.set_param({"nthread": NTHREAD, "tree_method": "hist"})

    results = []

    feature_names = np.array(features)

    for start in range(0, len(df), CHUNK_SIZE):

        end = min(start + CHUNK_SIZE, len(df))
        chunk = df.iloc[start:end]

        logger.info(f"{model_name} chunk {start} → {end}")

        dmat = xgb.DMatrix(chunk[features])

        contribs = booster.predict(dmat, pred_contribs=True)

        shap_vals = contribs[:, :-1]
        base_value = float(contribs[0, -1])
        preds = model.predict(chunk[features])

        top_idx = topk_shap(shap_vals, TOP_N)

        rows = np.arange(len(chunk))[:, None]

        out = pd.DataFrame({
            "grid_cell_id": chunk["grid_cell_id"].values,
            "date_hour": chunk["date_hour"].values,
            "model": model_name,
            "base_value": base_value,
            "predicted_value": preds,
        })

        for i in range(TOP_N):
            col = top_idx[:, i]
            out[f"top{i+1}_feature"] = feature_names[col]
            out[f"top{i+1}_shap"] = shap_vals[rows[:, 0], col]

        results.append(out)

    return pd.concat(results, ignore_index=True)


def main():

    logger.info("Loading data + models")

    df = load_data()
    features = get_features(df)

    logger.info(f"Rows: {len(df):,}, Features: {len(features)}")

    count_model = joblib.load(COUNT_MODEL_PATH)
    impact_model = joblib.load(IMPACT_MODEL_PATH)

    count_exp = explain_model(count_model, df, features, "violation_count")
    impact_exp = explain_model(impact_model, df, features, "impact_score")

    final = pd.concat([count_exp, impact_exp], ignore_index=True)

    final.to_parquet(OUTPUT_PATH, index=False)

    logger.info(f"DONE → {OUTPUT_PATH}")
    logger.info(f"Total rows written: {len(final):,}")


if __name__ == "__main__":
    main()
