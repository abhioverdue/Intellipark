# IntelliPark — Enforcement Intelligence

## Project structure

```
intellipark/
├── data/raw/                       ← put your data.csv here (HackerEarth link)
├── module1_pipeline/                clean.py, features.py
├── module2_impact_score/            scorer.py + config.json
├── module3_repeat_offender/         vehicle_risk.py + config.json
├── module4_hotspot_forecast/        grid_features.py, train_model.py, forecast_future.py + config.json
├── module5_edi/                     edi.py, future_priority.py, location_context.py, flow_proxy.py + config.json
├── module6_optimizer/                optimizer.py + config.json
├── run_pipeline.py                  ← NEW: runs all 6 modules in the right order
├── backend/                         ← NEW: FastAPI bridge (pipeline outputs → dashboard contract)
│   ├── app/
│   │   ├── main.py                  FastAPI app + CORS + /health
│   │   ├── config.py                resolves all pipeline output paths
│   │   ├── schemas.py               Pydantic models mirroring intellipark.ts exactly
│   │   ├── loaders.py               cached parquet/json/csv readers + shared lookups
│   │   └── routers/
│   │       ├── hotspots.py          GET /api/hotspots?mode=...
│   │       ├── allocations.py       GET /api/allocations?mode=...
│   │       ├── repeat_offenders.py  GET /api/repeat-offenders
│   │       ├── simulator.py         GET /api/simulator-defaults
│   │       └── limitations.py       GET /api/limitations
│   ├── requirements.txt
│   └── .env.example
├── dashboard/                       React/Vite dashboard (from earlier)
│   └── .env                         ← VITE_API_BASE=http://localhost:8000/api (already set)
└── landing/                         static marketing page (from earlier)
    └── generate_stats.py           ← NEW: builds stats.json from real pipeline outputs
```

## How the pieces connect

```
module1 → module2 → module3 → module4 → module5 (edi/future_priority,
then location_context+flow_proxy run TWICE — see run_pipeline.py) → module6
                                              │
                                              ▼
                                   backend/ (FastAPI, reads the parquet/JSON)
                                              │
                          ┌───────────────────┴───────────────────┐
                          ▼                                       ▼
                 dashboard/ (Vite/React,                landing/stats.json
                 VITE_API_BASE set)                      (generate_stats.py)
```

## Run it

```bash
# 1. Get the dataset
#    Download from the HackerEarth link and save as:
mkdir -p data/raw
# place your data.csv at data/raw/data.csv

# 2. Run the full pipeline (modules 1-6, both deployment modes, plus
#    the slim deploy-lookup files backend/build_deploy_lookups.py needs)
pip install -r module-requirements.txt   # pandas, xgboost, scikit-learn, joblib, pyarrow
python run_pipeline.py

# 3. Generate the landing page stats from real output
python landing/generate_stats.py

# 4. Start the backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# sanity check: http://localhost:8000/health should show all outputs = true

# 5. Start the dashboard (separate terminal)
cd dashboard
npm install
npm run dev
# open http://localhost:5173 — it's now reading real pipeline data, not mock data

# 6. Serve the landing page (separate terminal, optional)
cd landing
python3 -m http.server 4173
# open http://localhost:4173
```

## Deploying

See [DEPLOY.md](./DEPLOY.md) — Render (backend, via the included
`Dockerfile`) + Vercel (dashboard), tested against a simulated copy of
the exact container filesystem layout.

## Re-running after pipeline changes

`run_pipeline.py` supports resuming partway through, so you don't have to
re-train models every time you tweak a downstream config:

```bash
python run_pipeline.py --from edi          # skip straight to module5 onward
python run_pipeline.py --skip-train        # reuse existing count_model.pkl / impact_model.pkl
```

The backend's caches auto-invalidate off file mtimes — no restart needed
after re-running the pipeline, just refresh the dashboard.

## module4_hotspot_forecast/config.json — now actually wired (previously dead)

Three real fixes here, tested by actually running the scripts against
synthetic data (not just code review):

1. **`forecast_horizon_hours` was a genuine contradiction, not just unused.**
   `config.json` said `1`; `forecast_future.py` hardcoded `24*7 = 168`.
   Now the script reads it from config — and the config's value has
   been corrected to `168` so existing behavior (one week of forecast)
   doesn't change underneath you. Edit this value if you want a
   different forecast window.
2. **`grid_features.use_lag_1/use_lag_24/use_rolling_24` couldn't express
   most of what the code actually computes** (lags 1/3/6/24/168, rolling
   6/12/24 — the old boolean flags only covered 3 of 8). Replaced with
   explicit `"lags": [...]` and `"rolling_windows": [...]` lists that
   `grid_features.py` now reads directly — verified: narrowing the list
   measurably shrinks `training_data.parquet`'s column count.
3. **`train_fraction` is now live** in `train_model.py` (verified: changing
   it from 0.8 to 0.5 visibly shifted the train/test split). `model` and
   `targets` are validated against what the script can actually run
   (it only implements XGBoost with these two specific targets) and
   raise a clear error if config disagrees, rather than silently
   training the same thing regardless of what config says.

If you've already trained models under the old hardcoded 168-hour
horizon, no action needed — the corrected config matches that. If you
want a different horizon, lag set, or train split, those config edits
now actually take effect on the next `run_pipeline.py`.
