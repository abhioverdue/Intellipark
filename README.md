# IntelliPark

Enforcement-intelligence dashboard: hotspot forecasts, officer allocation, repeat-offender risk, and a policy simulator, backed by a FastAPI API serving precomputed pipeline outputs.

## 1. Run the dashboard (quickest way to look at it)

```bash
cd dashboard
npm install
npm run dev
```

Opens at `http://localhost:5173`.

## 2. Run the backend (to see real data instead of mocks)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
INTELLIPARK_CORS_ORIGINS=http://localhost:5173 uvicorn app.main:app --reload --port 8000
```

The `INTELLIPARK_CORS_ORIGINS` override is required, without it the API only allows the deployed frontend's origin, not your local dev server.

Check `http://localhost:8000/health` — it lists which pipeline output files are present.

Then point the dashboard at it:

```bash
cd dashboard
echo "VITE_API_BASE=http://localhost:8000/api" > .env
npm run dev
```

## 3. (Optional) Regenerate pipeline outputs from raw data

Already-computed outputs ship in each `module*/output/` folder, so this is only needed if you have your own raw dataset (`data/raw/data.csv`, not included) and want to recompute everything.

```bash
pip install -r module-requirements.txt
python module1_pipeline/clean.py
python module1_pipeline/features.py
python module2_impact_score/scorer.py
python module3_repeat_offender/vehicle_risk.py
python module4_hotspot_forecast/grid_features.py
python module4_hotspot_forecast/train_model.py
python module4_hotspot_forecast/forecast_future.py
python module4_hotspot_forecast/shap_explain.py   # needs shap, see shap-requirements.txt
python module5_edi/location_context.py
python module5_edi/edi.py
python module5_edi/future_priority.py
python module5_edi/flow_proxy.py
python module6_optimizer/optimizer.py
```

## 4. Production view

```bash
cd dashboard
npm run dev   
```

