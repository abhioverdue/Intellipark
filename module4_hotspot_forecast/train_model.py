from pathlib import Path
import logging

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)

from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "module4_hotspot_forecast"
    / "output"
    / "training_data.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "module4_hotspot_forecast"
    / "output"
)

FORECAST_PATH = OUTPUT_DIR / "grid_forecasts.parquet"

COUNT_MODEL_PATH = OUTPUT_DIR / "count_model.pkl"

IMPACT_MODEL_PATH = OUTPUT_DIR / "impact_model.pkl"

FEATURE_IMPORTANCE_PATH = (
    OUTPUT_DIR / "feature_importance.csv"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


BASE_FEATURES = [
    "hour",
    "dow",
    "month",
    "is_weekend",

    "hour_sin",
    "hour_cos",

    "dow_sin",
    "dow_cos",

    "critical_ratio",
    "repeat_offender_ratio",

    "violation_count_lag_1",
    "violation_count_lag_24",

    "violation_count_rolling_6",
    "violation_count_rolling_24",

    "avg_impact_score_lag_1",
    "avg_impact_score_lag_24",

    "avg_impact_score_rolling_6",
    "avg_impact_score_rolling_24"
]


OPTIONAL_FEATURES = [
    "violation_count_lag_2",
    "violation_count_lag_3",
    "violation_count_lag_6",
    "violation_count_lag_168",

    "avg_impact_score_lag_2",
    "avg_impact_score_lag_3",
    "avg_impact_score_lag_6",
    "avg_impact_score_lag_168",

    "violation_count_rolling_12",
    "avg_impact_score_rolling_12",

    "grid_mean_count",
    "grid_mean_impact",
]


def get_features(df: pd.DataFrame) -> list[str]:
    features = BASE_FEATURES.copy()

    for col in OPTIONAL_FEATURES:
        if col in df.columns:
            features.append(col)

    return features


def validate_schema(df: pd.DataFrame, features: list[str]):

    required = features + [
        "date_hour",
        "grid_cell_id",
        "violation_count",
        "avg_impact_score"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )


def build_count_model():

    return XGBRegressor(
        objective="count:poisson",
        eval_metric="mae",

        n_estimators=700,
        learning_rate=0.03,

        max_depth=10,
        min_child_weight=3,

        subsample=0.85,
        colsample_bytree=0.85,

        random_state=42,
        n_jobs=-1
    )


def build_impact_model():

    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",

        n_estimators=400,
        learning_rate=0.05,

        max_depth=8,
        min_child_weight=5,

        subsample=0.8,
        colsample_bytree=0.8,

        random_state=42,
        n_jobs=-1
    )


def save_feature_importance(
    model,
    features,
    model_name
):

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
            "model": model_name
        }
    )

    return importance.sort_values(
        "importance",
        ascending=False
    )


def evaluate(
    y_true,
    y_pred,
    name
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = root_mean_squared_error(
        y_true,
        y_pred
    )

    logger.info(f"{name} MAE: {mae:.3f}")
    logger.info(f"{name} RMSE: {rmse:.3f}")

    return mae, rmse


def main():

    logger.info("Loading training data")

    df = pd.read_parquet(INPUT_PATH)

    logger.info(f"Loaded {len(df):,} rows")

    df["date_hour"] = pd.to_datetime(
        df["date_hour"]
    )

    df = df.sort_values(
        "date_hour"
    ).reset_index(drop=True)

    FEATURES = get_features(df)

    logger.info(
        f"Using {len(FEATURES)} features"
    )

    validate_schema(df, FEATURES)

    logger.info("Schema validation passed")

    cutoff = df["date_hour"].quantile(0.80)

    train = df[
        df["date_hour"] < cutoff
    ].copy()

    test = df[
        df["date_hour"] >= cutoff
    ].copy()

    logger.info(
        f"Train rows: {len(train):,}"
    )

    logger.info(
        f"Test rows: {len(test):,}"
    )

    logger.info(
        f"Train end: {train['date_hour'].max()}"
    )

    logger.info(
        f"Test start: {test['date_hour'].min()}"
    )

    X_train = train[FEATURES]
    X_test = test[FEATURES]

    y_train_count = train["violation_count"]
    y_test_count = test["violation_count"]

    y_train_impact = train["avg_impact_score"]
    y_test_impact = test["avg_impact_score"]

    logger.info(
        "Training violation count model"
    )

    count_model = build_count_model()

    count_model.fit(
        X_train,
        y_train_count,
        eval_set=[(X_test, y_test_count)],
        verbose=False
    )

    logger.info(
        "Training impact model"
    )

    impact_model = build_impact_model()

    impact_model.fit(
        X_train,
        y_train_impact,
        eval_set=[(X_test, y_test_impact)],
        verbose=False
    )

    pred_count = count_model.predict(X_test)

    pred_count = np.clip(
        pred_count,
        0,
        None
    )

    pred_impact = impact_model.predict(X_test)

    pred_impact = np.clip(
        pred_impact,
        0,
        100
    )

    evaluate(
        y_test_count,
        pred_count,
        "Violation"
    )

    evaluate(
        y_test_impact,
        pred_impact,
        "Impact"
    )

    logger.info(
        "Generating forecasts"
    )

    df["predicted_violations"] = np.clip(
        count_model.predict(df[FEATURES]),
        0,
        None
    )

    df["predicted_impact"] = np.clip(
        impact_model.predict(df[FEATURES]),
        0,
        100
    )

    df["forecast_priority"] = pd.qcut(
        df["predicted_impact"],
        q=[0, 0.50, 0.80, 0.95, 1.0],
        labels=[
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
        ],
        duplicates="drop"
    )

    forecast_df = df[
        [
            "grid_cell_id",
            "date_hour",
            "hour",
            "dow",
            "predicted_violations",
            "predicted_impact",
            "forecast_priority"
        ]
    ].copy()

    forecast_df.to_parquet(
        FORECAST_PATH,
        index=False
    )

    joblib.dump(
        count_model,
        COUNT_MODEL_PATH
    )

    joblib.dump(
        impact_model,
        IMPACT_MODEL_PATH
    )

    importance = pd.concat(
        [
            save_feature_importance(
                count_model,
                FEATURES,
                "violation_count"
            ),

            save_feature_importance(
                impact_model,
                FEATURES,
                "impact_score"
            )
        ],
        ignore_index=True
    )

    importance.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False
    )

    logger.info(
        f"Saved forecasts: {FORECAST_PATH}"
    )

    logger.info(
        f"Saved feature importance: "
        f"{FEATURE_IMPORTANCE_PATH}"
    )


if __name__ == "__main__":
    main()