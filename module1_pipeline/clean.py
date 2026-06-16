from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd


# ==========================================================
# CONFIG
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data" / "raw" / "data.csv"

OUTPUT_DIR = ROOT / "module1_pipeline" / "output"

CLEANED_PATH = OUTPUT_DIR / "cleaned.parquet"
QUALITY_PATH = OUTPUT_DIR / "clean_quality.json"

IST_TZ = "Asia/Kolkata"

REQUIRED_COLUMNS = [
    "id",
    "latitude",
    "longitude",
    "vehicle_number",
    "vehicle_type",
    "created_datetime",
]


# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ==========================================================
# HELPERS
# ==========================================================

def parse_datetime(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    return dt.dt.tz_convert(IST_TZ)


def create_grid_cell_id(
    latitude: pd.Series,
    longitude: pd.Series,
    precision: int = 2,
) -> pd.Series:

    lat_grid = latitude.round(precision)
    lon_grid = longitude.round(precision)

    return (
        lat_grid.astype(str)
        + "_"
        + lon_grid.astype(str)
    )


def normalize_vehicle_type(series: pd.Series) -> pd.Series:

    mapping = {
        "SCOOTER": "TWO_WHEELER",
        "MOTOR CYCLE": "TWO_WHEELER",
        "MOTORCYCLE": "TWO_WHEELER",
        "MOPED": "TWO_WHEELER",

        "CAR": "CAR",
        "JEEP": "CAR",
        "VAN": "CAR",

        "PASSENGER AUTO": "AUTO",
        "GOODS AUTO": "AUTO",
        "AUTO RICKSHAW": "AUTO",

        "MAXI-CAB": "COMMERCIAL",
        "TEMPO": "COMMERCIAL",

        "LGV": "GOODS",
        "HGV": "GOODS",
        "TRUCK": "GOODS",

        "PRIVATE BUS": "BUS",
        "BUS": "BUS",
        "SCHOOL VEHICLE": "BUS",
    }

    clean = (
        series.fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return clean.map(mapping).fillna("OTHER")


def assign_vehicle_size(series: pd.Series) -> pd.Series:

    mapping = {
        "TWO_WHEELER": "SMALL",
        "AUTO": "MEDIUM",
        "CAR": "MEDIUM",
        "COMMERCIAL": "LARGE",
        "GOODS": "XLARGE",
        "BUS": "XLARGE",
        "OTHER": "MEDIUM",
        "UNKNOWN": "MEDIUM",
    }

    return series.map(mapping).fillna("MEDIUM")


# ==========================================================
# MAIN
# ==========================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw dataset")

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    logger.info(f"Loaded {len(df):,} rows")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}"
        )

    duplicate_ids = int(df["id"].duplicated().sum())

    if duplicate_ids:
        logger.warning(f"Removing {duplicate_ids:,} duplicate IDs")

        df = df.drop_duplicates(subset=["id"])

    df["vehicle_type_final"] = np.where(
        df["updated_vehicle_type"].notna(),
        df["updated_vehicle_type"],
        df["vehicle_type"],
    )

    datetime_cols = [
        "created_datetime",
        "modified_datetime",
        "validation_timestamp",
    ]

    for col in datetime_cols:
        if col in df.columns:
            df[col] = parse_datetime(df[col])

    df["validation_status"] = (
        df["validation_status"]
        .fillna("not_validated")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    categorical_cols = [
    "junction_name",
    "police_station",
    "location",
]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("UNKNOWN")
                .astype(str)
            )
    if "center_code" in df.columns:
        df["center_code"] = (
        df["center_code"]
        .astype("Int64")
        .astype(str)
        .replace("<NA>", "UNKNOWN")
    )

    df = df[
        df["latitude"].between(12.80, 13.30)
        & df["longitude"].between(77.40, 77.80)
    ].copy()

    df["grid_cell_id"] = create_grid_cell_id(
        df["latitude"],
        df["longitude"],
    )

    df["vehicle_category"] = normalize_vehicle_type(
        df["vehicle_type_final"]
    )

    df["vehicle_size_class"] = assign_vehicle_size(
        df["vehicle_category"]
    )

    category_cols = [
        "validation_status",
        "vehicle_category",
        "vehicle_size_class",
        "police_station",
        "junction_name",
        "grid_cell_id",
    ]

    for col in category_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    quality_report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_ids_removed": duplicate_ids,
        "missing_created_datetime": int(
            df["created_datetime"].isna().sum()
        ),
        "unique_vehicles": int(
            df["vehicle_number"].nunique()
        ),
        "unique_grids": int(
            df["grid_cell_id"].nunique()
        ),
    }

    with open(QUALITY_PATH, "w") as f:
        json.dump(quality_report, f, indent=2)

    df.to_parquet(
        CLEANED_PATH,
        index=False,
    )

    logger.info(f"Saved {CLEANED_PATH}")
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Quality report: {QUALITY_PATH}")


if __name__ == "__main__":
    main()