# backend/app/shap_paths.py
#
# Path config for the SHAP explainability output, kept in its own file
# so backend/app/config.py (the existing path registry) doesn't need
# to be touched.
#
# Mirrors config.py's ROOT-resolution convention exactly:
# backend/app/shap_paths.py -> parents[0]=backend/app, [1]=backend, [2]=ROOT

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.getenv("INTELLIPARK_ROOT", str(ROOT)))

M4_DIR = PROJECT_ROOT / "module4_hotspot_forecast" / "output"
SHAP_TOP_FEATURES_PATH = M4_DIR / "shap_top_features.parquet"
