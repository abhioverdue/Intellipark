/**
 * ShapExplanation.tsx
 *
 * "Why is this prediction what it is" panel.
 * Fetches GET /api/shap/{gridCellId} — served by backend/app/routers/shap.py
 * which reads module4_hotspot_forecast/output/shap_top_features.parquet.
 *
 * Key behaviour:
 * - BASE set  → hits the real backend. If 404 (cell not in parquet) shows
 *   a clear "no SHAP data" message. Never silently falls back to mock.
 * - BASE empty → shows mock data with a visible MOCK badge so you know.
 */

import { useEffect, useState } from "react";
import {
  fetchShapExplanation,
  type ShapExplanation,
  type ShapFetchResult,
} from "../api/intellipark";

type FetchState =
  | { status: "loading" }
  | ShapFetchResult;


// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

const FEATURE_LABELS: Record<string, string> = {
  violation_count_rolling_24:  "24h rolling violation count",
  violation_count_rolling_6:   "6h rolling violation count",
  violation_count_lag_1:       "Violations, 1h ago",
  violation_count_lag_24:      "Violations, 24h ago",
  avg_impact_score_rolling_24: "24h rolling impact score",
  avg_impact_score_rolling_6:  "6h rolling impact score",
  avg_impact_score_lag_1:      "Impact score, 1h ago",
  avg_impact_score_lag_24:     "Impact score, 24h ago",
  is_weekend:                  "Weekend",
  critical_ratio:              "Critical-violation ratio",
  repeat_offender_ratio:       "Repeat-offender ratio",
  hour_sin:                    "Time of day (sin)",
  hour_cos:                    "Time of day (cos)",
  dow_sin:                     "Day of week (sin)",
  dow_cos:                     "Day of week (cos)",
};

function labelFor(f: string) { return FEATURE_LABELS[f] ?? f; }

const MODEL_LABELS: Record<ShapExplanation["model"], string> = {
  violation_count: "Predicted violation count",
  impact_score:    "Predicted impact score",
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ShapExplanationPanel({ gridCellId }: { gridCellId: string }) {
  const [state, setState] = useState<FetchState>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading" });
    fetchShapExplanation(gridCellId).then(s => { if (!cancelled) setState(s); });
    return () => { cancelled = true; };
  }, [gridCellId]);

  // --- Loading ---
  if (state.status === "loading") {
    return (
      <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--ip-text-dim)", padding: "12px 0" }}>
        Loading SHAP explanation…
      </div>
    );
  }

  // --- Not found in parquet ---
  if (state.status === "not_found") {
    return (
      <div style={{
        marginTop: 20,
        padding: "10px 14px",
        background: "rgba(138,147,143,0.08)",
        border: "1px solid var(--ip-hairline)",
        borderRadius: 6,
        fontFamily: "var(--font-mono)",
        fontSize: "0.72rem",
        color: "var(--ip-text-dim)",
      }}>
        <strong style={{ color: "var(--ip-text)" }}>No SHAP data for this cell.</strong>
        {" "}Grid cell <code style={{ color: "var(--ip-yellow)" }}>{state.gridCellId}</code> was not found in{" "}
        <code>shap_top_features.parquet</code>. Check that <code>shap_explain.py</code> ran on the same
        training data that produced this hotspot.
      </div>
    );
  }

  // --- Backend error ---
  if (state.status === "error") {
    return (
      <div style={{
        marginTop: 20,
        padding: "10px 14px",
        background: "rgba(255,77,46,0.07)",
        border: "1px solid rgba(255,77,46,0.3)",
        borderRadius: 6,
        fontFamily: "var(--font-mono)",
        fontSize: "0.72rem",
        color: "#FF4D2E",
      }}>
        SHAP fetch failed: {state.message}
      </div>
    );
  }

  // --- Real or mock data ---
  const { data } = state;
  const isMock = state.status === "mock";

  return (
    <div style={{ marginTop: 20 }}>
      <div style={{
        display: "flex", alignItems: "center", gap: 10,
        fontFamily: "var(--font-mono)", fontSize: "0.7rem",
        letterSpacing: "0.06em", color: "var(--ip-text-dim)", marginBottom: 10,
      }}>
        MODEL EXPLAINABILITY — SHAP
        {isMock && (
          <span style={{
            background: "rgba(245,197,24,0.15)",
            color: "var(--ip-yellow)",
            fontSize: "0.6rem", fontWeight: 700,
            letterSpacing: "0.08em",
            padding: "2px 6px", borderRadius: 3,
          }}>
            MOCK — set VITE_API_BASE to see real data
          </span>
        )}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {data.explanations.map(exp => {
          const maxAbs = Math.max(1, ...exp.topContributors.map(c => Math.abs(c.shapValue)));
          return (
            <div key={exp.model} style={{
              background: "var(--ip-surface-2)",
              border: "1px solid var(--ip-hairline)",
              borderRadius: 8, padding: "12px 14px",
            }}>
              <div style={{
                display: "flex", justifyContent: "space-between",
                alignItems: "baseline", marginBottom: 8,
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
                      <div style={{ width: 180, fontSize: "0.74rem", color: "var(--ip-text)", flexShrink: 0 }}>
                        {labelFor(c.feature)}
                      </div>
                      <div style={{ flex: 1, display: "flex", background: "var(--ip-surface)", borderRadius: 3, height: 6, overflow: "hidden" }}>
                        <div style={{
                          width: `${widthPct}%`,
                          height: "100%",
                          background: positive ? "#FF4D2E" : "#39B88A",
                          borderRadius: 3,
                          marginLeft: positive ? "auto" : 0,
                        }} />
                      </div>
                      <div style={{
                        fontFamily: "var(--font-mono)", fontSize: "0.68rem",
                        width: 54, textAlign: "right",
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
