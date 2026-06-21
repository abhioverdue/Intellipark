/**
 * ShapExplanation.tsx
 *
 * Self-contained "why is this prediction what it is" panel, fed by the
 * new backend endpoint GET /api/shap/{gridCellId} (backend/app/routers/shap.py),
 * which itself reads module4_hotspot_forecast/output/shap_top_features.parquet
 * (produced by module4_hotspot_forecast/shap_explain.py).
 *
 * This is a NEW file and does not modify api/intellipark.ts — it has its
 * own tiny fetch-or-mock helper so the existing API layer stays untouched.
 * Drop <ShapExplanationPanel gridCellId={selected.gridCellId} /> into any
 * detail view; it fetches and renders on its own.
 */

import { useEffect, useState } from "react";

// ---------------------------------------------------------------------------
// Types — mirror backend/app/routers/shap.py's response_model exactly
// ---------------------------------------------------------------------------

export interface ShapContributor {
  feature: string;
  shapValue: number;
}

export interface ShapExplanation {
  gridCellId: string;
  dateHour: string;
  model: "violation_count" | "impact_score";
  baseValue: number;
  predictedValue: number;
  topContributors: ShapContributor[];
}

export interface ShapExplanationResponse {
  gridCellId: string;
  explanations: ShapExplanation[];
}

// ---------------------------------------------------------------------------
// Fetch-or-mock — same BASE convention as api/intellipark.ts, kept local
// so that file doesn't need to change.
// ---------------------------------------------------------------------------

const BASE = (import.meta.env?.VITE_API_BASE as string) ?? "";

function mockShapExplanation(gridCellId: string): ShapExplanationResponse {
  return {
    gridCellId,
    explanations: [
      {
        gridCellId,
        dateHour: new Date().toISOString(),
        model: "violation_count",
        baseValue: 5.1,
        predictedValue: 9.4,
        topContributors: [
          { feature: "violation_count_rolling_24", shapValue: 2.3 },
          { feature: "is_weekend", shapValue: 1.1 },
          { feature: "critical_ratio", shapValue: 0.9 },
        ],
      },
      {
        gridCellId,
        dateHour: new Date().toISOString(),
        model: "impact_score",
        baseValue: 48.2,
        predictedValue: 67.5,
        topContributors: [
          { feature: "avg_impact_score_rolling_24", shapValue: 11.4 },
          { feature: "hour_sin", shapValue: 6.2 },
          { feature: "repeat_offender_ratio", shapValue: -2.1 },
        ],
      },
    ],
  };
}

async function fetchShapExplanation(gridCellId: string): Promise<ShapExplanationResponse | null> {
  if (!BASE) return mockShapExplanation(gridCellId);
  try {
    const res = await fetch(`${BASE}/shap/${encodeURIComponent(gridCellId)}`);
    if (res.status === 404) return null; // no shap_explain.py run yet for this cell
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return (await res.json()) as ShapExplanationResponse;
  } catch (err) {
    console.warn(
      `[IntelliPark API] Failed to reach ${BASE}/shap/${gridCellId} — falling back to mock data.`,
      err
    );
    return mockShapExplanation(gridCellId);
  }
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

const FEATURE_LABELS: Record<string, string> = {
  violation_count_rolling_24: "24h rolling violation count",
  violation_count_rolling_6: "6h rolling violation count",
  violation_count_lag_1: "Violations, 1h ago",
  violation_count_lag_24: "Violations, 24h ago",
  avg_impact_score_rolling_24: "24h rolling impact score",
  avg_impact_score_rolling_6: "6h rolling impact score",
  avg_impact_score_lag_1: "Impact score, 1h ago",
  avg_impact_score_lag_24: "Impact score, 24h ago",
  is_weekend: "Weekend",
  critical_ratio: "Critical-violation ratio",
  repeat_offender_ratio: "Repeat-offender ratio",
  hour_sin: "Time of day",
  hour_cos: "Time of day",
  dow_sin: "Day of week",
  dow_cos: "Day of week",
};

function labelFor(feature: string): string {
  return FEATURE_LABELS[feature] ?? feature;
}

const MODEL_LABELS: Record<ShapExplanation["model"], string> = {
  violation_count: "Predicted violation count",
  impact_score: "Predicted impact score",
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ShapExplanationPanel({ gridCellId }: { gridCellId: string }) {
  const [data, setData] = useState<ShapExplanationResponse | null | undefined>(undefined);

  useEffect(() => {
    let cancelled = false;
    setData(undefined);
    fetchShapExplanation(gridCellId).then(res => {
      if (!cancelled) setData(res);
    });
    return () => {
      cancelled = true;
    };
  }, [gridCellId]);

  // Loading
  if (data === undefined) {
    return (
      <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--ip-text-dim)", padding: "8px 0" }}>
        Loading model explanation…
      </div>
    );
  }

  // No SHAP output yet for this cell (404) — don't break the rest of the UI
  if (data === null || data.explanations.length === 0) {
    return null;
  }

  return (
    <div style={{ marginTop: 20 }}>
      <div style={{
        fontFamily: "var(--font-mono)", fontSize: "0.7rem", letterSpacing: "0.06em",
        color: "var(--ip-text-dim)", marginBottom: 10,
      }}>
        MODEL EXPLAINABILITY (SHAP)
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {data.explanations.map(exp => {
          const maxAbs = Math.max(1, ...exp.topContributors.map(c => Math.abs(c.shapValue)));
          return (
            <div
              key={exp.model}
              style={{
                background: "var(--ip-surface-2)",
                border: "1px solid var(--ip-hairline)",
                borderRadius: 8,
                padding: "12px 14px",
              }}
            >
              <div style={{
                display: "flex", justifyContent: "space-between", alignItems: "baseline",
                marginBottom: 8,
              }}>
                <div style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--ip-text)" }}>
                  {MODEL_LABELS[exp.model]}
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem", color: "var(--ip-text-dim)" }}>
                  base {exp.baseValue.toFixed(1)} → predicted {exp.predictedValue.toFixed(1)}
                </div>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {exp.topContributors.map(c => {
                  const positive = c.shapValue >= 0;
                  const widthPct = (Math.abs(c.shapValue) / maxAbs) * 100;
                  return (
                    <div key={c.feature} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <div style={{ width: 170, fontSize: "0.74rem", color: "var(--ip-text)", flexShrink: 0 }}>
                        {labelFor(c.feature)}
                      </div>
                      <div style={{ flex: 1, display: "flex", background: "var(--ip-surface)", borderRadius: 3, height: 6, overflow: "hidden" }}>
                        <div
                          style={{
                            width: `${widthPct}%`,
                            height: "100%",
                            background: positive ? "#FF4D2E" : "#39B88A",
                            borderRadius: 3,
                            marginLeft: positive ? "auto" : 0,
                          }}
                        />
                      </div>
                      <div style={{
                        fontFamily: "var(--font-mono)", fontSize: "0.68rem", width: 50, textAlign: "right",
                        color: positive ? "#FF4D2E" : "#39B88A",
                      }}>
                        {positive ? "+" : ""}{c.shapValue.toFixed(2)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
      <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.64rem", color: "var(--ip-text-dim)", marginTop: 8 }}>
        Red = pushed prediction up · Green = pushed prediction down
      </div>
    </div>
  );
}
