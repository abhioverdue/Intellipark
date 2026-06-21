/**
 * ShapExplanation.tsx
 *
 * "Why is this prediction what it is" panel.
 * Fetches GET /api/shap/{gridCellId} — served by backend/app/routers/shap.py
 * which reads module4_hotspot_forecast/output/shap_top_features.parquet.
 *
 * This panel is read by patrol officers with no data-science background, so
 * raw SHAP feature names/values are never shown directly. Every contributing
 * feature is translated into a plain-English sentence, and the numeric SHAP
 * value is only used internally to (a) decide direction — red/up = raised
 * the risk, green/down = lowered it — and (b) size a relative bar. No
 * decimal SHAP units, base values, or feature codes reach the screen.
 *
 * Key behaviour:
 * - BASE set  → hits the real backend. If 404 (cell not in parquet) shows
 *   a clear "no data" message. Never silently falls back to mock.
 * - BASE empty → shows mock data with a visible MOCK badge so you know.
 */

import { useEffect, useState } from "react";
import { ArrowUpRight, ArrowDownRight } from "lucide-react";
import {
  fetchShapExplanation,
  type ShapExplanation,
  type ShapContributor,
  type ShapFetchResult,
} from "../api/intellipark";

type FetchState =
  | { status: "loading" }
  | ShapFetchResult;

// ---------------------------------------------------------------------------
// Plain-English translations
//
// Every feature the pipeline can produce (module4_hotspot_forecast/
// shap_explain.py — BASE_FEATURES + OPTIONAL_FEATURES) maps to a pair of
// human sentences: one for when it pushed the prediction UP, one for DOWN.
// ---------------------------------------------------------------------------

const FEATURE_EXPLANATIONS: Record<string, { up: string; down: string }> = {
  // time of day / week / year
  hour:                        { up: "This hour of day is usually busier here",            down: "This hour of day is usually quieter here" },
  hour_sin:                    { up: "The time of day points to more activity",             down: "The time of day points to less activity" },
  hour_cos:                    { up: "The time of day points to more activity",             down: "The time of day points to less activity" },
  dow:                         { up: "This day of the week is usually busier here",         down: "This day of the week is usually quieter here" },
  dow_sin:                     { up: "The day-of-week pattern points to more activity",     down: "The day-of-week pattern points to less activity" },
  dow_cos:                     { up: "The day-of-week pattern points to more activity",     down: "The day-of-week pattern points to less activity" },
  month:                       { up: "This time of year tends to see more violations",      down: "This time of year tends to see fewer violations" },
  is_weekend:                  { up: "It's a weekend, when this area tends to get busier",  down: "It's a weekday, when this area tends to be quieter" },

  // location history
  grid_mean_count:             { up: "This spot has a history of more violations than most",          down: "This spot has a history of fewer violations than most" },
  grid_mean_impact:            { up: "Violations at this spot tend to be more serious than most",      down: "Violations at this spot tend to be less serious than most" },
  critical_ratio:              { up: "A high share of past violations here were serious",              down: "Most past violations here have been minor" },
  repeat_offender_ratio:       { up: "Many of the vehicles here are repeat offenders",                 down: "Few repeat offenders in this area" },

  // recent violation counts
  violation_count_lag_1:       { up: "Violations spiked in the last hour",            down: "Violations were quiet in the last hour" },
  violation_count_lag_2:       { up: "Violations were elevated 2 hours ago",          down: "Violations were low 2 hours ago" },
  violation_count_lag_3:       { up: "Violations were elevated 3 hours ago",          down: "Violations were low 3 hours ago" },
  violation_count_lag_6:       { up: "Violations were elevated 6 hours ago",          down: "Violations were low 6 hours ago" },
  violation_count_lag_24:      { up: "This same hour yesterday saw more violations",  down: "This same hour yesterday saw fewer violations" },
  violation_count_lag_168:     { up: "This same hour last week saw more violations",  down: "This same hour last week saw fewer violations" },
  violation_count_rolling_6:   { up: "Violations have trended up over the last 6 hours",  down: "Violations have trended down over the last 6 hours" },
  violation_count_rolling_12:  { up: "Violations have trended up over the last 12 hours", down: "Violations have trended down over the last 12 hours" },
  violation_count_rolling_24:  { up: "Violations have trended up over the last 24 hours", down: "Violations have trended down over the last 24 hours" },

  // recent severity
  avg_impact_score_lag_1:      { up: "The most recent violation (last hour) was serious",   down: "The most recent violation (last hour) was minor" },
  avg_impact_score_lag_2:      { up: "Violations 2 hours ago were serious",                  down: "Violations 2 hours ago were minor" },
  avg_impact_score_lag_3:      { up: "Violations 3 hours ago were serious",                  down: "Violations 3 hours ago were minor" },
  avg_impact_score_lag_6:      { up: "Violations 6 hours ago were serious",                  down: "Violations 6 hours ago were minor" },
  avg_impact_score_lag_24:     { up: "This same hour yesterday saw serious violations",      down: "This same hour yesterday saw minor violations" },
  avg_impact_score_lag_168:    { up: "This same hour last week saw serious violations",      down: "This same hour last week saw minor violations" },
  avg_impact_score_rolling_6:  { up: "Violations over the last 6 hours have been more serious than usual",  down: "Violations over the last 6 hours have been less serious than usual" },
  avg_impact_score_rolling_12: { up: "Violations over the last 12 hours have been more serious than usual", down: "Violations over the last 12 hours have been less serious than usual" },
  avg_impact_score_rolling_24: { up: "Violations over the last 24 hours have been more serious than usual", down: "Violations over the last 24 hours have been less serious than usual" },
};

/** Turns a snake_case feature code into readable words, as a last-resort fallback. */
function humanize(feature: string): string {
  return feature.replace(/_/g, " ");
}

function explainContributor(c: ShapContributor): string {
  const positive = c.shapValue >= 0;
  const entry = FEATURE_EXPLANATIONS[c.feature];
  if (entry) return positive ? entry.up : entry.down;
  return positive
    ? `${humanize(c.feature)} is higher than usual here`
    : `${humanize(c.feature)} is lower than usual here`;
}

const MODEL_LABELS: Record<ShapExplanation["model"], string> = {
  violation_count: "How many violations are expected",
  impact_score:    "How serious those violations are likely to be",
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
        Loading explanation…
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
      }}>
        <div style={{ fontSize: "0.82rem", color: "var(--ip-text)", marginBottom: 4 }}>
          A detailed breakdown isn't available for this specific spot yet.
        </div>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.66rem", color: "var(--ip-text-dim)" }}>
          Grid cell <code style={{ color: "var(--ip-yellow)" }}>{state.gridCellId}</code> was not found in{" "}
          <code>shap_top_features.parquet</code>. Check that <code>shap_explain.py</code> ran on the same
          training data that produced this hotspot.
        </div>
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
      }}>
        <div style={{ fontSize: "0.82rem", color: "#FF4D2E", marginBottom: 4 }}>
          Couldn't load the explanation for this spot.
        </div>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.66rem", color: "#FF4D2E" }}>
          {state.message}
        </div>
      </div>
    );
  }

  // --- Real or mock data ---
  const { data } = state;
  const isMock = state.status === "mock";

  return (
    <div style={{ marginTop: 20 }}>
      <div style={{
        display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap",
        marginBottom: 4,
      }}>
        <div style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--ip-text)" }}>
          What's driving this forecast
        </div>
        {isMock && (
          <span style={{
            background: "rgba(245,197,24,0.15)",
            color: "var(--ip-yellow)",
            fontFamily: "var(--font-mono)",
            fontSize: "0.6rem", fontWeight: 700,
            letterSpacing: "0.08em",
            padding: "2px 6px", borderRadius: 3,
          }}>
            MOCK — set VITE_API_BASE to see real data
          </span>
        )}
      </div>
      <div style={{ fontSize: "0.74rem", color: "var(--ip-text-dim)", marginBottom: 14 }}>
        Based on recent activity patterns near this spot — separate from the fixed location score above.
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {data.explanations.map(exp => {
          const maxAbs = Math.max(1e-9, ...exp.topContributors.map(c => Math.abs(c.shapValue)));
          return (
            <div key={exp.model} style={{
              background: "var(--ip-surface-2)",
              border: "1px solid var(--ip-hairline)",
              borderRadius: 8, padding: "12px 14px",
            }}>
              <div style={{ fontSize: "0.76rem", fontWeight: 600, color: "var(--ip-text)", marginBottom: 10 }}>
                {MODEL_LABELS[exp.model]}
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {exp.topContributors.map(c => {
                  const positive = c.shapValue >= 0;
                  const widthPct = Math.max(8, (Math.abs(c.shapValue) / maxAbs) * 100);
                  const color = positive ? "#FF4D2E" : "#39B88A";
                  return (
                    <div key={c.feature} style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
                      <div style={{ flexShrink: 0, marginTop: 1 }}>
                        {positive
                          ? <ArrowUpRight size={15} color={color} />
                          : <ArrowDownRight size={15} color={color} />}
                      </div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: "0.8rem", color: "var(--ip-text)", lineHeight: 1.35 }}>
                          {explainContributor(c)}
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 5 }}>
                          <div style={{ flex: 1, maxWidth: 140, background: "var(--ip-surface)", borderRadius: 3, height: 5, overflow: "hidden" }}>
                            <div style={{ width: `${widthPct}%`, height: "100%", background: color, borderRadius: 3 }} />
                          </div>
                          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", color, flexShrink: 0 }}>
                            {positive ? "Raised the risk" : "Lowered the risk"}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      <div style={{
        display: "flex", alignItems: "center", gap: 16, marginTop: 10,
        fontFamily: "var(--font-mono)", fontSize: "0.64rem", color: "var(--ip-text-dim)",
      }}>
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <ArrowUpRight size={11} color="#FF4D2E" /> Raised the risk
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <ArrowDownRight size={11} color="#39B88A" /> Lowered the risk
        </span>
      </div>
    </div>
  );
}

