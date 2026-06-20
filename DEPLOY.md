# Deploying IntelliPark

Tested path: **Render** (backend, via Docker) + **Vercel** (dashboard).
Both have free tiers, no credit card required, and deploy straight from
GitHub — good fit under hackathon time pressure. The backend image was
verified by simulating the exact container filesystem layout locally
(see note at the bottom) since this sandbox has no Docker daemon — if
something doesn't match your actual Render build, that's the one step
I couldn't test directly.

## 0. One-time prep (do this locally first)

```bash
# Full pipeline run against your real dataset (also builds the slim
# deploy lookups automatically as its last step)
pip install -r module-requirements.txt
python run_pipeline.py

# Generate the landing page stats
python landing/generate_stats.py
```

Push to GitHub. With the `.gitignore` already in this repo, `git add -A`
will only pick up the slim deploy bundle (~7MB) — not the full
per-violation parquet files or your raw `data.csv`. Verify before
pushing:

```bash
git add -A
git status --short | grep "^A"
# should show deploy_lookups.json, breakdown_shapes.json,
# vehicle_risk.parquet, feature_importance.csv,
# edi_scores_with_flow.parquet, future_priority_with_flow.parquet,
# module6_optimizer/config.json, allocations_*.json — nothing bigger.
```

## 1. Backend → Render

1. [render.com](https://render.com) → New → Web Service → connect your repo.
2. Render auto-detects the `Dockerfile` at the repo root — leave "Docker" as the environment.
3. Instance type: Free is fine for a hackathon demo.
4. Leave build/start commands blank — they come from the `Dockerfile`'s `CMD`.
5. Deploy. Render gives you a URL like `https://your-app.onrender.com`.
6. Check it: `https://your-app.onrender.com/health` should show `"status": "ok"`.
7. **Note the CORS origin is currently hardcoded to `localhost:5173`** in the Dockerfile. Once you have your Vercel URL (step 2 below), come back to Render → Environment → add:
   ```
   INTELLIPARK_CORS_ORIGINS=https://your-dashboard.vercel.app
   ```
   and redeploy (or trigger a manual redeploy from the Render dashboard).

Free-tier Render web services spin down after inactivity and take ~30-60s
to wake on the next request — if a judge's first click is slow, that's
why, not a bug. Worth a heads-up slide if you're presenting live.

## 2. Dashboard → Vercel

1. [vercel.com](https://vercel.com) → New Project → import the same repo.
2. **Root Directory**: set to `dashboard` (Vercel builds from there, not the repo root).
3. Framework preset: Vite (auto-detected from `package.json`).
4. Environment variable:
   ```
   VITE_API_BASE=https://your-app.onrender.com/api
   ```
   (your actual Render URL from step 1, with `/api` appended — same convention as local `.env`.)
5. Deploy. Vercel gives you `https://your-dashboard.vercel.app`.
6. Go back to Render and set `INTELLIPARK_CORS_ORIGINS` to this exact URL (step 1.7 above), then redeploy the backend.

## 3. Landing page (optional)

If you want the marketing landing page live too: same Vercel flow, Root
Directory `landing`, Framework preset "Other" (static HTML — no build
command needed). Before deploying, point its CTA at your real dashboard
URL:

```bash
# in landing/main.js, change:
const DASHBOARD_URL = 'http://localhost:5173';
# to:
const DASHBOARD_URL = 'https://your-dashboard.vercel.app';
```

## Alternative: Railway instead of Render

Same Dockerfile works unchanged on [railway.app](https://railway.app) if
you hit Render's free-tier sleep behavior during judging and want an
always-on alternative (Railway's free tier has usage-based limits
instead of spin-down, more predictable for a live demo). Steps are
the same: New Project → Deploy from GitHub repo → set
`INTELLIPARK_CORS_ORIGINS` as an environment variable → it auto-detects
the Dockerfile.

## How I verified the Dockerfile without a Docker daemon

This sandbox doesn't have `docker` installed, so I couldn't literally
run `docker build`. Instead I reproduced the exact filesystem the image
would have — `backend/app` under `/srv/app`, the slim bundle under
`/data/...`, `INTELLIPARK_ROOT=/data` set — and ran the same
`uvicorn app.main:app --host 0.0.0.0 --port $PORT` command the
Dockerfile's `CMD` uses. All 5 endpoints × both modes returned 200,
including `/health` reporting `"status": "ok"`. This is strong evidence
the image will behave correctly, but if your actual Render/Railway build
log shows something different, that's the one layer between "verified"
and "guaranteed" — check the build logs on first deploy.
