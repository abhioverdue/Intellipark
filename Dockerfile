WORKDIR /srv

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app

# The backend resolves its data root via app/config.py:
#   ROOT = Path(__file__).resolve().parents[2]   (used for local dev,
#   where backend/app/config.py sits 2 levels under the project root)
# That convention doesn't hold once app/ is copied into its own image
# layer, so we override it explicitly with INTELLIPARK_ROOT instead --
# config.py already supports this env var for exactly this reason.
ENV INTELLIPARK_ROOT=/data

# Only the slim deploy bundle from build_deploy_lookups.py + module
# 3/5/6 outputs (~7MB) -- not the ~98MB of full per-violation parquet
# files, and not the pipeline scripts themselves (this image only
# serves already-computed results).
COPY module1_pipeline/output/deploy_lookups.json /data/module1_pipeline/output/deploy_lookups.json
COPY module2_impact_score/output/breakdown_shapes.json /data/module2_impact_score/output/breakdown_shapes.json
COPY module3_repeat_offender/output/vehicle_risk.parquet /data/module3_repeat_offender/output/vehicle_risk.parquet
COPY module4_hotspot_forecast/output/feature_importance.csv /data/module4_hotspot_forecast/output/feature_importance.csv
COPY module4_hotspot_forecast/output/shap_top_features.parquet /data/module4_hotspot_forecast/output/shap_top_features.parquet
COPY module5_edi/output/edi_scores_with_flow.parquet /data/module5_edi/output/edi_scores_with_flow.parquet
COPY module5_edi/output/future_priority_with_flow.parquet /data/module5_edi/output/future_priority_with_flow.parquet
COPY module6_optimizer/config.json /data/module6_optimizer/config.json
COPY module6_optimizer/output/allocations_historical_gap.json /data/module6_optimizer/output/allocations_historical_gap.json
COPY module6_optimizer/output/allocations_future_forecast.json /data/module6_optimizer/output/allocations_future_forecast.json

# Set this to your deployed frontend's actual origin (Render/Vercel
# give you the URL after first deploy; redeploy once you have it).
ENV INTELLIPARK_CORS_ORIGINS=http://localhost:5173

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

