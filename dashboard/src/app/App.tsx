/**
 * IntelliPark Dashboard — App.tsx
 *
 * All data comes through src/app/api/intellipark.ts.
 * Replace the mock data in that file (or set VITE_API_BASE) to wire
 * this UI to real pipeline outputs without changing this component.
 */

import { useState, useMemo, useCallback, useEffect, useRef } from "react";
import {
  MapPin, Users, Car, Sliders, Info,
  Sun, Moon, ChevronUp, ChevronDown,
  ArrowUpRight, ArrowDownRight,
  AlertCircle, Loader2,
} from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer,
} from "recharts";
import {
  MapContainer, TileLayer, CircleMarker, Popup, useMap,
} from "react-leaflet";
import L from "leaflet";

import {
  fetchHotspots,
  fetchAllocations,
  fetchRepeatOffenders,
  fetchSimulatorDefaults,
  fetchLimitations,
  simulateAllocation,
  type DeployMode,
  type ZoneColor,
  type RiskLevel,
  type HotspotLocation,
  type AllocationResponse,
  type RepeatOffendersResponse,
  type SimulatorDefaults,
  type LimitationsResponse,
  type SimulatorInputs,
} from "./api/intellipark";
import { ShapExplanationPanel } from "./components/ShapExplanation";

/* =========================================================================
   DESIGN TOKENS
   ========================================================================= */

function cssVars(dark: boolean): React.CSSProperties {
  return {
    "--ip-base":      dark ? "#0A0E0F" : "#F7F8FA",
    "--ip-surface":   dark ? "#141A1C" : "#FFFFFF",
    "--ip-surface-2": dark ? "#1C2528" : "#EEF0F3",
    "--ip-hairline":  dark ? "#2A3438" : "#D8DCE3",
    "--ip-text":      dark ? "#E8E4DC" : "#1A2433",
    "--ip-text-dim":  dark ? "#8A938F" : "#5A6678",
    "--ip-red":   "#FF4D2E",
    "--ip-yellow": dark ? "#F5C518" : "#C49E00",
    "--ip-green":  dark ? "#39B88A" : "#2A9972",
    "--ip-sim-bg": dark ? "#101618" : "#F0EEE8",
    "--font-sans": "'IBM Plex Sans', system-ui, sans-serif",
    "--font-mono": "'IBM Plex Mono', monospace",
  } as React.CSSProperties;
}

/* =========================================================================
   SHARED UTILITIES
   ========================================================================= */

function zoneColor(zone: ZoneColor): { bg: string; text: string; solid: string } {
  const map: Record<ZoneColor, { bg: string; text: string; solid: string }> = {
    RED:    { bg: "rgba(255,77,46,0.15)",   text: "#FF4D2E",              solid: "#FF4D2E" },
    YELLOW: { bg: "rgba(245,197,24,0.15)",  text: "var(--ip-yellow)",     solid: "#F5C518" },
    GREEN:  { bg: "rgba(57,184,138,0.15)",  text: "var(--ip-green)",      solid: "#39B88A" },
  };
  return map[zone] ?? map.GREEN;
}

function riskColor(tier: RiskLevel): { bg: string; text: string } {
  const map: Record<RiskLevel, { bg: string; text: string }> = {
    CRITICAL: { bg: "rgba(232,25,44,0.12)",   text: "#E8192C" },
    HIGH:     { bg: "rgba(255,77,46,0.10)",   text: "#FF4D2E" },
    MEDIUM:   { bg: "rgba(245,197,24,0.10)",  text: "#C49E00" },
    LOW:      { bg: "rgba(57,184,138,0.10)",  text: "#2A9972" },
  };
  return map[tier] ?? map.LOW;
}

const ACTION_LABEL: Record<string, string> = {
  IMMEDIATE_TOW:   "IMMEDIATE TOW",
  PRIORITY_TICKET: "PRIORITY TICKET",
  MONITOR:         "MONITOR",
  IGNORE:          "IGNORE",
};

function actionChipStyle(action: string): React.CSSProperties {
  const map: Record<string, { bg: string; color: string }> = {
    IMMEDIATE_TOW:   { bg: "rgba(232,25,44,0.15)",   color: "#E8192C" },
    PRIORITY_TICKET: { bg: "rgba(255,77,46,0.12)",   color: "#FF4D2E" },
    MONITOR:         { bg: "rgba(245,197,24,0.12)",  color: "var(--ip-yellow)" },
    IGNORE:          { bg: "rgba(138,147,143,0.12)", color: "var(--ip-text-dim)" },
  };
  const s = map[action] ?? map.MONITOR;
  return {
    display: "inline-block",
    background: s.bg,
    color: s.color,
    fontFamily: "var(--font-mono)",
    fontSize: "0.66rem",
    fontWeight: 700,
    letterSpacing: "0.05em",
    padding: "3px 8px",
    borderRadius: 4,
    whiteSpace: "nowrap",
  };
}

function parseRiskExplanation(v: RepeatOffendersResponse["criticalVehicles"][0]): string {
  return (
    `This vehicle has accumulated ${v.totalViolations} violations, ` +
    `of which ${(v.criticalRatio * 100).toFixed(0)}% were critical severity. ` +
    `Its violations are highly concentrated — ` +
    `${(v.topGridConcentration * 100).toFixed(0)}% occur at a single location, ` +
    `suggesting a habitual pattern rather than incidental offences. ` +
    `Mean enforcement impact score: ${v.meanImpactScore.toFixed(1)}.`
  );
}

/* =========================================================================
   SMALL REUSABLE COMPONENTS
   ========================================================================= */

function ZoneBadge({ zone }: { zone: ZoneColor }) {
  const c = zoneColor(zone);
  return (
    <span style={{
      display: "inline-block",
      background: c.bg,
      color: c.text,
      fontFamily: "var(--font-mono)",
      fontSize: "0.68rem",
      fontWeight: 600,
      letterSpacing: "0.06em",
      padding: "2px 8px",
      borderRadius: 4,
      whiteSpace: "nowrap",
    }}>
      {zone}
    </span>
  );
}

function ActionChip({ action }: { action: string }) {
  return <span style={actionChipStyle(action)}>{ACTION_LABEL[action] ?? action}</span>;
}

function SectionEyebrow({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontFamily: "var(--font-mono)",
      fontSize: "0.7rem",
      letterSpacing: "0.1em",
      color: "var(--ip-yellow)",
      marginBottom: 12,
    }}>
      {children}
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h2 style={{
      fontFamily: "var(--font-mono)",
      fontWeight: 700,
      fontSize: "clamp(1.2rem, 2vw, 1.6rem)",
      color: "var(--ip-text)",
      margin: "0 0 8px",
      letterSpacing: "-0.01em",
    }}>
      {children}
    </h2>
  );
}

function StatCard({ value, label, sub, accent }: {
  value: string | number; label: string; sub?: string; accent?: string;
}) {
  return (
    <div style={{
      background: "var(--ip-surface)",
      border: "1px solid var(--ip-hairline)",
      borderRadius: 8,
      padding: "22px 20px",
    }}>
      <div style={{
        fontFamily: "var(--font-mono)",
        fontSize: "2.2rem",
        fontWeight: 700,
        color: accent ?? "var(--ip-text)",
        lineHeight: 1.1,
        marginBottom: 8,
        fontVariantNumeric: "tabular-nums",
      }}>
        {value}
      </div>
      <div style={{ fontSize: "0.85rem", color: "var(--ip-text)", fontWeight: 500, marginBottom: 3 }}>{label}</div>
      {sub && (
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--ip-text-dim)", letterSpacing: "0.02em" }}>
          {sub}
        </div>
      )}
    </div>
  );
}

function SortIcon({ active, dir }: { active: boolean; dir: "asc" | "desc" }) {
  if (!active) return <span style={{ opacity: 0.25, marginLeft: 4 }}>↕</span>;
  return dir === "asc"
    ? <ChevronUp size={12} style={{ marginLeft: 4, verticalAlign: "middle" }} />
    : <ChevronDown size={12} style={{ marginLeft: 4, verticalAlign: "middle" }} />;
}

function SegControl({ options, value, onChange }: {
  options: string[]; value: string; onChange: (v: string) => void;
}) {
  return (
    <div style={{
      display: "inline-flex",
      background: "var(--ip-surface-2)",
      border: "1px solid var(--ip-hairline)",
      borderRadius: 7,
      padding: 3,
      gap: 2,
    }}>
      {options.map(o => (
        <button key={o} onClick={() => onChange(o)} style={{
          padding: "5px 13px",
          borderRadius: 5,
          border: "none",
          fontFamily: "var(--font-mono)",
          fontSize: "0.74rem",
          fontWeight: value === o ? 600 : 400,
          letterSpacing: "0.02em",
          cursor: "pointer",
          background: value === o ? "var(--ip-surface)" : "transparent",
          color: value === o ? "var(--ip-text)" : "var(--ip-text-dim)",
          boxShadow: value === o ? "0 1px 3px rgba(0,0,0,0.15)" : "none",
          transition: "all 0.15s",
        }}>
          {o}
        </button>
      ))}
    </div>
  );
}

function NavItem({ icon, label, active, onClick }: {
  icon: React.ReactNode; label: string; active: boolean; onClick: () => void;
}) {
  return (
    <button onClick={onClick} style={{
      display: "flex",
      alignItems: "center",
      gap: 10,
      padding: "9px 12px",
      borderRadius: 7,
      border: "none",
      cursor: "pointer",
      width: "100%",
      textAlign: "left",
      background: active ? "rgba(245,197,24,0.1)" : "transparent",
      color: active ? "var(--ip-yellow)" : "var(--ip-text-dim)",
      transition: "all 0.12s",
      fontFamily: "var(--font-mono)",
      fontSize: "0.78rem",
      fontWeight: active ? 600 : 400,
      letterSpacing: "0.02em",
      borderLeft: active ? "2px solid var(--ip-yellow)" : "2px solid transparent",
    }}>
      {icon}
      {label}
    </button>
  );
}

function LoadingState({ label }: { label?: string }) {
  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", gap: 14, padding: "60px 0",
      color: "var(--ip-text-dim)",
    }}>
      <Loader2 size={28} style={{ animation: "spin 1s linear infinite" }} />
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem" }}>{label ?? "Loading…"}</span>
    </div>
  );
}

function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", gap: 12, padding: "60px 0",
      color: "var(--ip-text-dim)",
    }}>
      <AlertCircle size={28} />
      <div style={{ fontFamily: "var(--font-mono)", fontWeight: 600, color: "var(--ip-text)", fontSize: "0.92rem" }}>{title}</div>
      <div style={{ fontSize: "0.84rem", maxWidth: 380, textAlign: "center", lineHeight: 1.6 }}>{detail}</div>
    </div>
  );
}

/* =========================================================================
   SECTION 1 — HOTSPOT MAP
   ========================================================================= */

/* =========================================================================
   RADAR OVERLAY — canvas-based gaussian intensity field (smooth view)
   Mirrors the landing page's radial-gradient weather-radar aesthetic but
   driven by real lat/lon data. Renders on a Leaflet canvas overlay so it
   moves and zooms with the map tiles.
   ========================================================================= */

interface RadarPoint {
  lat: number;
  lng: number;
  priorityScore: number;  // 0-100
  zone: ZoneColor;
}

// Zone → radar color (RGB, will be used in canvas gradient stops)
const RADAR_COLOR: Record<ZoneColor, [number, number, number]> = {
  RED:    [255, 70,  50],
  YELLOW: [245, 197, 24],
  GREEN:  [57,  184, 138],
};

function RadarOverlay({ points }: { points: RadarPoint[] }) {
  const map = useMap();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const layerRef = useRef<L.Layer | null>(null);

  useEffect(() => {
    if (!map) return;

    // Create a custom Leaflet layer that owns a canvas element
    const RadarLayer = L.Layer.extend({
      onAdd(m: L.Map) {
        const canvas = document.createElement("canvas");
        canvas.style.position = "absolute";
        canvas.style.top = "0";
        canvas.style.left = "0";
        canvas.style.pointerEvents = "none";
        canvas.style.zIndex = "400";
        // Attach above tile pane, below marker pane
        m.getPanes().overlayPane.appendChild(canvas);
        canvasRef.current = canvas;
        this._canvas = canvas;
        this._map = m;
        this._draw();
        m.on("moveend zoomend resize", this._draw, this);
      },
      onRemove(m: L.Map) {
        m.off("moveend zoomend resize", this._draw, this);
        if (this._canvas && this._canvas.parentNode) {
          this._canvas.parentNode.removeChild(this._canvas);
        }
        canvasRef.current = null;
      },
      _draw() {
        const canvas = this._canvas as HTMLCanvasElement;
        const m = this._map as L.Map;
        if (!canvas || !m) return;

        const size = m.getSize();
        canvas.width  = size.x;
        canvas.height = size.y;
        canvas.style.width  = size.x + "px";
        canvas.style.height = size.y + "px";

        // Align canvas with the map's top-left corner
        const topLeft = m.containerPointToLayerPoint([0, 0]);
        L.DomUtil.setPosition(canvas, topLeft);

        const ctx = canvas.getContext("2d")!;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // source-over: normal alpha compositing. "screen" was causing
        // overlapping blobs to add RGB channels and blow out to white.
        ctx.globalCompositeOperation = "source-over";

        const sortedPoints = [...points].sort((a, b) => a.priorityScore - b.priorityScore);
        for (const pt of sortedPoints) {
          const px = m.latLngToContainerPoint([pt.lat, pt.lng]);

          const zoom = m.getZoom();
          const zoomScale = Math.pow(2, zoom - 11);
          const intensity = Math.max(0.35, Math.min(1, pt.priorityScore / 100));
          const radius = Math.max(46, Math.min(170, 48 * zoomScale * (0.95 + intensity * 0.75)));

          const [r, g, b] = RADAR_COLOR[pt.zone];
          const grad = ctx.createRadialGradient(px.x, px.y, 0, px.x, px.y, radius);
          if (pt.zone === "RED") {
            grad.addColorStop(0, `rgba(255,77,46,${(0.30 * intensity).toFixed(3)})`);
            grad.addColorStop(0.26, `rgba(245,197,24,${(0.22 * intensity).toFixed(3)})`);
            grad.addColorStop(0.62, `rgba(57,184,138,${(0.14 * intensity).toFixed(3)})`);
          } else if (pt.zone === "YELLOW") {
            grad.addColorStop(0, `rgba(245,197,24,${(0.26 * intensity).toFixed(3)})`);
            grad.addColorStop(0.38, `rgba(255,145,48,${(0.16 * intensity).toFixed(3)})`);
            grad.addColorStop(0.72, `rgba(57,184,138,${(0.12 * intensity).toFixed(3)})`);
          } else {
            grad.addColorStop(0, `rgba(${r},${g},${b},${(0.18 * intensity).toFixed(3)})`);
            grad.addColorStop(0.48, `rgba(90,210,160,${(0.11 * intensity).toFixed(3)})`);
            grad.addColorStop(0.78, `rgba(245,197,24,${(0.06 * intensity).toFixed(3)})`);
          }
          grad.addColorStop(1, `rgba(${r},${g},${b},0)`);

          ctx.beginPath();
          ctx.arc(px.x, px.y, radius, 0, Math.PI * 2);
          ctx.fillStyle = grad;
          ctx.fill();
        }

        // Gentle blur for feathered radar look — low enough not to wash out colours.
        canvas.style.filter = "blur(3px) saturate(1.45) contrast(1.12)";
      },
    });

    const layer = new (RadarLayer as unknown as new () => L.Layer)();
    // Attach points so _draw can see them via closure
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (layer as any)._points = points;
    // Patch _draw to use current points from closure
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const proto = layer as any;
    const originalDraw = proto._draw.bind(proto);
    proto._draw = originalDraw;

    layer.addTo(map);
    layerRef.current = layer;

    return () => {
      layer.remove();
      layerRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, JSON.stringify(points.map(p => [p.lat, p.lng, p.priorityScore, p.zone]))]);

  return null;
}

/**
 * Re-centers/zooms the Leaflet map whenever the underlying data
 * changes (e.g. switching historical_gap <-> future_forecast can
 * shift which grid cells are present). Depends on a stringified
 * snapshot of the positions rather than the array reference itself,
 * since a new array is created every render but should only trigger
 * a refit when the actual coordinates change.
 */
function FitBounds({ positions }: { positions: [number, number][] }) {
  const map = useMap();
  const key = JSON.stringify(positions);

  useEffect(() => {
    if (positions.length === 0) return;
    if (positions.length === 1) {
      map.setView(positions[0], 14);
    } else {
      map.fitBounds(positions, { padding: [28, 28] });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, map]);

  return null;
}

function HotspotMap({ mode, dark }: { mode: DeployMode; dark: boolean }) {
  const [data, setData] = useState<Awaited<ReturnType<typeof fetchHotspots>> | null>(null);
  const [loading, setLoading] = useState(true);
  const [mapView, setMapView] = useState<"smooth" | "exact">("smooth");
  const [metric, setMetric] = useState("Priority score");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setSelectedId(null);
    fetchHotspots(mode).then(setData).finally(() => setLoading(false));
  }, [mode]);

  if (loading) return <LoadingState label="Loading hotspot data…" />;
  if (!data || data.locations.length === 0) {
    return (
      <EmptyState
        title="No hotspot data available"
        detail="Run the Module 4 → 5 pipeline to generate grid forecasts and EDI scores. This view will populate once those outputs exist."
      />
    );
  }

  const locs = data.locations;
  const isPast = mode === "historical_gap";
  const selected = selectedId !== null ? (locs.find(l => l.gridCellId === selectedId) ?? null) : null;

  function verdict(loc: HotspotLocation) {
    if (loc.observedViolations === null) return null;
    const diff = loc.observedViolations - loc.predictedViolations;
    if (diff < -15) return { label: "This location was under-patrolled.", color: "#FF4D2E" };
    if (diff > 15)  return { label: "This location was over-patrolled.",  color: "#39B88A" };
    return { label: "Coverage was roughly appropriate.", color: "#F5C518" };
  }

  const positions: [number, number][] = locs.map(loc => [loc.lat, loc.lng]);
  const smoothPositions: [number, number][] = useMemo(
    () => [...locs]
      .sort((a, b) => b.priorityScore - a.priorityScore)
      .slice(0, Math.min(160, locs.length))
      .map(loc => [loc.lat, loc.lng]),
    [locs],
  );
  const tileClass = `ip-leaflet-tiles${dark ? " ip-leaflet-tiles-dark" : ""}`;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Controls */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
        <SegControl
          options={["Smooth view", "Exact locations"]}
          value={mapView === "smooth" ? "Smooth view" : "Exact locations"}
          onChange={v => setMapView(v === "Smooth view" ? "smooth" : "exact")}
        />
        <select
          value={metric}
          onChange={e => setMetric(e.target.value)}
          style={{
            background: "var(--ip-surface)",
            border: "1px solid var(--ip-hairline)",
            color: "var(--ip-text)",
            fontFamily: "var(--font-mono)",
            fontSize: "0.76rem",
            padding: "6px 10px",
            borderRadius: 6,
            cursor: "pointer",
          }}
        >
          {["Priority score", "Predicted violations", "Traffic-flow impact", "Recommended patrol units"].map(m => (
            <option key={m}>{m}</option>
          ))}
        </select>
      </div>

      {/* Map + ranked list */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 16, alignItems: "start" }}>
        {/* Map canvas */}
        <div style={{
          border: "1px solid var(--ip-hairline)",
          borderRadius: 10,
          overflow: "hidden",
          background: "var(--ip-surface)",
          position: "relative",
        }}>
          {/* Scoped dark-tile filter: inverts OSM's light tile colors so
              the basemap doesn't clash with the dashboard's dark theme,
              without touching marker/circle colors (those live in a
              separate Leaflet pane this selector doesn't target). */}
          <style>{`
            .ip-leaflet-tiles-dark .leaflet-tile-pane {
              filter: invert(1) hue-rotate(180deg) brightness(0.92) contrast(0.88) saturate(0.35);
            }
            .ip-leaflet-tiles .leaflet-control-attribution {
              background: rgba(20,26,28,0.75); color: #8A938F; font-size: 10px;
            }
            .ip-leaflet-tiles .leaflet-control-attribution a { color: #B7BEC2; }
          `}</style>
          <div className={tileClass} style={{ height: 400, position: "relative" }}>
            <MapContainer
              center={[13.05, 77.60]}
              zoom={11}
              style={{ height: "100%", width: "100%", background: "#1a2a38" }}
              scrollWheelZoom
            >
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              />
              <FitBounds positions={mapView === "smooth" ? smoothPositions : positions} />

              {mapView === "smooth" && (
                <RadarOverlay
                  points={locs.map(loc => ({
                    lat: loc.lat,
                    lng: loc.lng,
                    priorityScore: loc.priorityScore,
                    zone: loc.zone,
                  }))}
                />
              )}

              {mapView === "exact" && locs.map((loc, i) => {
                const c = zoneColor(loc.zone);
                const isSelected = selectedId === loc.gridCellId;
                const radius = Math.max(5, loc.priorityScore / 12);
                return (
                  <CircleMarker
                    key={loc.gridCellId + i}
                    center={[loc.lat, loc.lng]}
                    radius={radius}
                    pathOptions={{
                      color: isSelected ? "#FFFFFF" : c.solid,
                      weight: isSelected ? 2 : 1,
                      fillColor: c.solid,
                      fillOpacity: 0.85,
                    }}
                    eventHandlers={{
                      click: () => setSelectedId(isSelected ? null : loc.gridCellId),
                    }}
                  >
                    <Popup>
                      <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.78rem" }}>
                        <strong>{loc.junctionName}</strong><br />
                        {loc.predictedViolations} predicted &middot; {loc.recommendedPatrolUnits} officers<br />
                        Zone: {loc.zone}
                      </div>
                    </Popup>
                  </CircleMarker>
                );
              })}
            </MapContainer>

            <div style={{
              position: "absolute", bottom: 8, right: 10, zIndex: 1000, pointerEvents: "none",
              fontFamily: "var(--font-mono)", fontSize: "0.62rem",
              color: "rgba(255,255,255,0.55)", letterSpacing: "0.03em",
              background: "rgba(10,14,15,0.45)", padding: "2px 6px", borderRadius: 4,
            }}>
              {metric.toUpperCase()}
            </div>
          </div>
          <div style={{
            display: "flex", gap: 20, padding: "12px 16px",
            borderTop: "1px solid var(--ip-hairline)",
            fontFamily: "var(--font-mono)", fontSize: "0.72rem",
            color: "var(--ip-text-dim)", background: "var(--ip-surface)",
          }}>
            {(["GREEN", "YELLOW", "RED"] as ZoneColor[]).map(z => (
              <div key={z} style={{ display: "flex", alignItems: "center", gap: 7 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: zoneColor(z).solid, display: "inline-block" }} />
                {z} — {{ GREEN: "clear", YELLOW: "caution", RED: "priority" }[z]}
              </div>
            ))}
          </div>
        </div>

        {/* Ranked list */}
        <div style={{
          background: "var(--ip-surface)",
          border: "1px solid var(--ip-hairline)",
          borderRadius: 10, overflow: "hidden",
        }}>
          <div style={{
            padding: "12px 16px",
            borderBottom: "1px solid var(--ip-hairline)",
            fontFamily: "var(--font-mono)", fontSize: "0.72rem",
            letterSpacing: "0.06em", color: "var(--ip-text-dim)",
          }}>
            TOP {locs.length} LOCATIONS
          </div>
          {locs.map((loc, i) => (
            <div
              key={loc.gridCellId + i}
              onClick={() => setSelectedId(loc.gridCellId === selectedId ? null : loc.gridCellId)}
              style={{
                padding: "10px 16px",
                borderBottom: i < locs.length - 1 ? "1px solid var(--ip-hairline)" : "none",
                cursor: "pointer",
                background: selectedId === loc.gridCellId ? "rgba(245,197,24,0.07)" : "transparent",
                display: "flex", alignItems: "center", gap: 10,
                transition: "background 0.1s",
              }}
            >
              <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--ip-text-dim)", width: 16, flexShrink: 0 }}>
                {i + 1}
              </span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: "0.82rem", fontWeight: 500, color: "var(--ip-text)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                  {loc.junctionName}
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem", color: "var(--ip-text-dim)", marginTop: 2 }}>
                  {loc.predictedViolations} pred · {loc.recommendedPatrolUnits} officers
                </div>
              </div>
              <ZoneBadge zone={loc.zone} />
            </div>
          ))}
        </div>
      </div>

      {/* Detail panel */}
      {selected && (() => {
        const v = verdict(selected);
        const c = zoneColor(selected.zone);
        const breakdown = [
          { factor: "Junction type",      value: selected.scoreBreakdown.junctionType },
          { factor: "Violation severity", value: selected.scoreBreakdown.violationSeverity },
          { factor: "Vehicle size",       value: selected.scoreBreakdown.vehicleSize },
          { factor: "Peak hour",          value: selected.scoreBreakdown.peakHour },
          { factor: "Hotspot density",    value: selected.scoreBreakdown.hotspotDensity },
          { factor: "Repeat offenders",   value: selected.scoreBreakdown.repeatOffenders },
        ];
        return (
          <div style={{
            background: "var(--ip-surface)",
            border: "1px solid var(--ip-hairline)",
            borderLeft: `3px solid ${c.solid}`,
            borderRadius: 8, padding: "20px 24px",
          }}>
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 16, gap: 12, flexWrap: "wrap" }}>
              <div>
                <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--ip-text)", marginBottom: 4, fontFamily: "var(--font-mono)" }}>
                  {selected.junctionName}
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem", color: "var(--ip-text-dim)", marginBottom: 8 }}>
                  {selected.gridCellId} · {selected.locationContext}
                </div>
                <ZoneBadge zone={selected.zone} />
              </div>
              <div style={{ display: "flex", gap: 24 }}>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.8rem", fontWeight: 700, color: "var(--ip-text)", fontVariantNumeric: "tabular-nums" }}>
                    {selected.predictedViolations}
                  </div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem", color: "var(--ip-text-dim)" }}>
                    {isPast ? "PREDICTED" : "FORECAST"}
                  </div>
                </div>
                {isPast && selected.observedViolations !== null && (
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.8rem", fontWeight: 700, color: "var(--ip-text)", fontVariantNumeric: "tabular-nums" }}>
                      {selected.observedViolations}
                    </div>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem", color: "var(--ip-text-dim)" }}>ACTUAL</div>
                  </div>
                )}
              </div>
            </div>
            {isPast && v ? (
              <div style={{
                padding: "10px 14px", background: `${v.color}15`,
                border: `1px solid ${v.color}40`, borderRadius: 6,
                marginBottom: 16, fontSize: "0.88rem", color: v.color, fontWeight: 500,
              }}>
                {v.label} ({selected.predictedViolations} predicted{selected.observedViolations !== null ? `, ${selected.observedViolations} logged` : ""})
              </div>
            ) : (
              <div style={{
                padding: "10px 14px", background: "rgba(245,197,24,0.08)",
                border: "1px solid rgba(245,197,24,0.3)", borderRadius: 6,
                marginBottom: 16, fontSize: "0.82rem", color: "var(--ip-text-dim)",
              }}>
                <strong style={{ color: "var(--ip-yellow)" }}>Forecast — </strong>
                {selected.predictedViolations} violations expected. This is a projection for a future period, not an observation.
              </div>
            )}
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", letterSpacing: "0.06em", color: "var(--ip-text-dim)", marginBottom: 10 }}>
              WHY THIS LOCATION IS FLAGGED
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {breakdown.map(({ factor, value }) => (
                <div key={factor} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <div style={{ width: 150, fontSize: "0.8rem", color: "var(--ip-text)", flexShrink: 0 }}>{factor}</div>
                  <div style={{ flex: 1, background: "var(--ip-surface-2)", borderRadius: 3, height: 6, overflow: "hidden" }}>
                    <div style={{ width: `${(value / 25) * 100}%`, height: "100%", background: c.solid, borderRadius: 3 }} />
                  </div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--ip-text-dim)", width: 28, textAlign: "right" }}>
                    {value}
                  </div>
                </div>
              ))}
            </div>
            <ShapExplanationPanel gridCellId={selected.gridCellId} />
          </div>
        );
      })()}
    </div>
  );
}

/* =========================================================================
   SECTION 2 — OFFICER ALLOCATION
   ========================================================================= */

function OfficerAllocation({ mode }: { mode: DeployMode }) {
  const [data, setData] = useState<AllocationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [sortCol, setSortCol] = useState<string>("expectedPriorityScore");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  useEffect(() => {
    setLoading(true);
    fetchAllocations(mode).then(setData).finally(() => setLoading(false));
  }, [mode]);

  const sorted = useMemo(() => {
    if (!data) return [];
    return [...data.allocations].sort((a, b) => {
      const va = (a as Record<string, unknown>)[sortCol] as number | string;
      const vb = (b as Record<string, unknown>)[sortCol] as number | string;
      if (typeof va === "string") return sortDir === "asc" ? (va as string).localeCompare(vb as string) : (vb as string).localeCompare(va as string);
      return sortDir === "asc" ? (va as number) - (vb as number) : (vb as number) - (va as number);
    });
  }, [data, sortCol, sortDir]);

  const toggleSort = useCallback((col: string) => {
    if (sortCol === col) setSortDir(d => d === "asc" ? "desc" : "asc");
    else { setSortCol(col); setSortDir("desc"); }
  }, [sortCol]);

  if (loading) return <LoadingState label="Loading allocations…" />;
  if (!data) {
    return (
      <EmptyState
        title="Allocation data not ready"
        detail="Run module6_optimizer/optimizer.py to generate allocation JSON files. Both modes are produced in a single run."
      />
    );
  }

  const thStyle: React.CSSProperties = {
    padding: "10px 14px", fontFamily: "var(--font-mono)", fontSize: "0.68rem",
    letterSpacing: "0.06em", color: "var(--ip-text-dim)", textAlign: "left",
    borderBottom: "1px solid var(--ip-hairline)", cursor: "pointer",
    userSelect: "none", whiteSpace: "nowrap",
  };

  const zoneChartData = data.byZone.map(z => ({ zone: z.zone, officers: z.officers, fill: zoneColor(z.zone).solid }));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
        <StatCard value={data.summary.availableOfficers} label="Officers available" sub="total shift strength" />
        <StatCard value={data.summary.locationsCovered}  label="Locations covered"  sub="this allocation" />
        <StatCard value={`${data.summary.locationsCovered} / ${data.summary.totalEligibleLocations}`} label="Spread" sub="hotspot locations covered" accent="var(--ip-yellow)" />
        <StatCard value={data.summary.totalPriorityScore.toLocaleString()} label="Priority score addressed" sub="sum of impact values" />
      </div>

      <div style={{ background: "var(--ip-surface)", border: "1px solid var(--ip-hairline)", borderRadius: 8, padding: "18px 20px" }}>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", letterSpacing: "0.07em", color: "var(--ip-text-dim)", marginBottom: 14 }}>
          OFFICERS BY ZONE — red zones absorb the most resource
        </div>
        <ResponsiveContainer width="100%" height={80}>
          <BarChart data={zoneChartData} layout="vertical" barSize={20}>
            <XAxis type="number" hide />
            <YAxis type="category" dataKey="zone" width={60}
              tick={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--ip-text-dim)" }}
              axisLine={false} tickLine={false}
            />
            <Tooltip
              contentStyle={{ background: "var(--ip-surface)", border: "1px solid var(--ip-hairline)", fontFamily: "var(--font-mono)", fontSize: 12 }}
              formatter={(v: unknown) => [`${v} officers`, ""]}
              labelStyle={{ color: "var(--ip-text)" }}
            />
            <Bar dataKey="officers" radius={4}>
              {zoneChartData.map(entry => <Cell key={entry.zone} fill={entry.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ background: "var(--ip-surface)", border: "1px solid var(--ip-hairline)", borderRadius: 8, overflow: "hidden" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              {[
                { key: "junctionName",          label: "LOCATION" },
                { key: "zone",                  label: "ZONE" },
                { key: "officersAllocated",     label: "OFFICERS" },
                { key: "expectedPriorityScore", label: "IMPACT SCORE" },
              ].map(({ key, label }) => (
                <th key={key} style={thStyle} onClick={() => toggleSort(key)}>
                  {label}<SortIcon active={sortCol === key} dir={sortDir} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((row, i) => (
              <tr key={row.gridCellId + i} style={{ background: i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.015)" }}>
                <td style={{ padding: "10px 14px", fontSize: "0.84rem", color: "var(--ip-text)", maxWidth: 240, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {row.junctionName}
                </td>
                <td style={{ padding: "10px 14px" }}><ZoneBadge zone={row.zone} /></td>
                <td style={{ padding: "10px 14px", fontFamily: "var(--font-mono)", fontSize: "0.84rem", color: "var(--ip-text)", fontVariantNumeric: "tabular-nums" }}>
                  {row.officersAllocated}
                </td>
                <td style={{ padding: "10px 14px", fontFamily: "var(--font-mono)", fontSize: "0.84rem", color: "var(--ip-text)", fontVariantNumeric: "tabular-nums" }}>
                  {row.expectedPriorityScore}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* =========================================================================
   SECTION 3 — REPEAT OFFENDERS
   ========================================================================= */

function RepeatOffenders() {
  const [data, setData] = useState<RepeatOffendersResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedPlate, setSelectedPlate] = useState<string | null>(null);

  useEffect(() => {
    fetchRepeatOffenders().then(setData).finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingState label="Loading vehicle risk data…" />;
  if (!data) {
    return (
      <EmptyState
        title="Vehicle risk data not ready"
        detail="Run module3_repeat_offender/vehicle_risk.py to generate vehicle_risk.parquet."
      />
    );
  }

  const selected = data.criticalVehicles.find(v => v.vehicleNumber === selectedPlate) ?? null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
        {data.tierSummary.map(t => {
          const c = riskColor(t.tier);
          return (
            <div key={t.tier} style={{ background: c.bg, border: `1px solid ${c.text}40`, borderRadius: 8, padding: "20px 18px" }}>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "2rem", fontWeight: 700, color: c.text, fontVariantNumeric: "tabular-nums", marginBottom: 6 }}>
                {t.percentage}%
              </div>
              <div style={{ fontWeight: 600, color: "var(--ip-text)", marginBottom: 2, fontSize: "0.9rem" }}>{t.tier}</div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--ip-text-dim)" }}>{t.count.toLocaleString()} vehicles</div>
            </div>
          );
        })}
      </div>

      <div style={{ background: "var(--ip-surface)", border: "1px solid var(--ip-hairline)", borderRadius: 8, overflow: "hidden" }}>
        <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--ip-hairline)", fontFamily: "var(--font-mono)", fontSize: "0.7rem", letterSpacing: "0.07em", color: "var(--ip-text-dim)" }}>
          CRITICAL-RISK VEHICLES — immediate action required
        </div>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              {["PLATE", "VIOLATIONS", "CRITICAL RATIO", "CONCENTRATION", "MEAN IMPACT", "ACTION"].map(h => (
                <th key={h} style={{ padding: "10px 14px", fontFamily: "var(--font-mono)", fontSize: "0.65rem", letterSpacing: "0.06em", color: "var(--ip-text-dim)", textAlign: "left", borderBottom: "1px solid var(--ip-hairline)", whiteSpace: "nowrap" }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.criticalVehicles.map((v, i) => (
              <tr
                key={v.vehicleNumber}
                onClick={() => setSelectedPlate(v.vehicleNumber === selectedPlate ? null : v.vehicleNumber)}
                style={{
                  background: selectedPlate === v.vehicleNumber ? "rgba(232,25,44,0.07)" : i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.015)",
                  cursor: "pointer",
                }}
              >
                <td style={{ padding: "10px 14px", fontFamily: "var(--font-mono)", fontSize: "0.84rem", color: "var(--ip-text)", whiteSpace: "nowrap", fontWeight: 600 }}>{v.vehicleNumber}</td>
                <td style={{ padding: "10px 14px", fontFamily: "var(--font-mono)", fontSize: "0.84rem", color: "var(--ip-text)", fontVariantNumeric: "tabular-nums" }}>{v.totalViolations}</td>
                <td style={{ padding: "10px 14px", fontFamily: "var(--font-mono)", fontSize: "0.84rem", color: "#E8192C", fontVariantNumeric: "tabular-nums" }}>{(v.criticalRatio * 100).toFixed(0)}%</td>
                <td style={{ padding: "10px 14px", fontFamily: "var(--font-mono)", fontSize: "0.84rem", color: "var(--ip-text)", fontVariantNumeric: "tabular-nums" }}>{(v.topGridConcentration * 100).toFixed(0)}%</td>
                <td style={{ padding: "10px 14px", fontFamily: "var(--font-mono)", fontSize: "0.84rem", color: "var(--ip-text)", fontVariantNumeric: "tabular-nums" }}>{v.meanImpactScore.toFixed(1)}</td>
                <td style={{ padding: "10px 14px" }}><ActionChip action={v.recommendedAction} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selected && (
        <div style={{
          background: "var(--ip-surface)",
          border: "1px solid rgba(232,25,44,0.4)",
          borderLeft: "3px solid #E8192C",
          borderRadius: 8, padding: "20px 24px",
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
            <div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: 700, color: "var(--ip-text)", marginBottom: 6 }}>{selected.vehicleNumber}</div>
              <ActionChip action={selected.recommendedAction} />
            </div>
            <div style={{ display: "flex", gap: 24 }}>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.8rem", fontWeight: 700, color: "#E8192C" }}>{selected.totalViolations}</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.66rem", color: "var(--ip-text-dim)" }}>TOTAL VIOLATIONS</div>
              </div>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.8rem", fontWeight: 700, color: "var(--ip-text)" }}>{selected.meanImpactScore.toFixed(1)}</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.66rem", color: "var(--ip-text-dim)" }}>MEAN IMPACT</div>
              </div>
            </div>
          </div>
          <div style={{ marginTop: 16, padding: "12px 16px", background: "rgba(232,25,44,0.06)", borderRadius: 6, fontSize: "0.88rem", color: "var(--ip-text)", lineHeight: 1.6 }}>
            {parseRiskExplanation(selected)} Recommended action: <strong style={{ color: "#E8192C" }}>{ACTION_LABEL[selected.recommendedAction]}</strong>.
          </div>
        </div>
      )}
    </div>
  );
}

/* =========================================================================
   SECTION 4 — POLICY SIMULATOR
   ========================================================================= */

function PolicySimulator() {
  const [defaults, setDefaults] = useState<SimulatorDefaults | null>(null);
  const [loading, setLoading] = useState(true);
  const [inputs, setInputs] = useState<SimulatorInputs | null>(null);

  useEffect(() => {
    fetchSimulatorDefaults().then(d => {
      setDefaults(d);
      setInputs({ officerCount: d.officerCount, demandWeight: d.demandWeight, flowWeight: d.flowWeight, maxOfficersPerLocation: d.maxOfficersPerLocation, minimumPriorityThreshold: d.minimumPriorityThreshold });
    }).finally(() => setLoading(false));
  }, []);

  if (loading || !defaults || !inputs) return <LoadingState label="Loading simulator…" />;

  const result = simulateAllocation(defaults, inputs);
  const base = defaults.baseResult;

  function setInput<K extends keyof SimulatorInputs>(key: K, val: SimulatorInputs[K]) {
    setInputs(prev => prev ? { ...prev, [key]: val } : prev);
  }

  function DeltaBlock({ label, base: b, sim, unit = "" }: { label: string; base: number; sim: number; unit?: string }) {
    const d = sim - b;
    const p = Math.round(((sim - b) / (b || 1)) * 100);
    const isGood = d >= 0;
    return (
      <div style={{ background: "var(--ip-surface)", border: "1px solid var(--ip-hairline)", borderRadius: 8, padding: "18px 20px", flex: 1 }}>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem", letterSpacing: "0.06em", color: "var(--ip-text-dim)", marginBottom: 10 }}>{label.toUpperCase()}</div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 4 }}>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "1.6rem", fontWeight: 700, color: "var(--ip-text-dim)", fontVariantNumeric: "tabular-nums" }}>{b}{unit}</span>
          <span style={{ fontSize: "1.2rem", color: "var(--ip-text-dim)" }}>→</span>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "1.6rem", fontWeight: 700, color: "var(--ip-text)", fontVariantNumeric: "tabular-nums" }}>{sim}{unit}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6, color: isGood ? "#39B88A" : "#FF4D2E", fontFamily: "var(--font-mono)", fontSize: "0.8rem", fontWeight: 600 }}>
          {isGood ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
          {d > 0 ? "+" : ""}{d}{unit} ({p > 0 ? "+" : ""}{p}%)
        </div>
      </div>
    );
  }

  function SliderRow({ label, value, min, max, step, onChange, fmt }: {
    label: string; value: number; min: number; max: number; step: number;
    onChange: (v: number) => void; fmt?: (v: number) => string;
  }) {
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 14 }}>
        <div style={{ width: 200, fontSize: "0.84rem", color: "var(--ip-text)", flexShrink: 0 }}>{label}</div>
        <input type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(Number(e.target.value))}
          style={{ flex: 1, accentColor: "var(--ip-yellow)" }}
        />
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.84rem", color: "var(--ip-yellow)", fontWeight: 600, width: 52, textAlign: "right", flexShrink: 0 }}>
          {fmt ? fmt(value) : value}
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{
        display: "inline-flex", alignItems: "center", gap: 8,
        padding: "6px 14px", background: "rgba(245,197,24,0.10)",
        border: "1px solid rgba(245,197,24,0.3)", borderRadius: 6, alignSelf: "flex-start",
        fontFamily: "var(--font-mono)", fontSize: "0.72rem", fontWeight: 700,
        letterSpacing: "0.08em", color: "var(--ip-yellow)",
      }}>
        <Sliders size={12} />
        SIMULATION MODE — not live data
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div style={{ background: "var(--ip-surface)", border: "1px solid var(--ip-hairline)", borderRadius: 8, padding: "20px 22px", opacity: 0.75 }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", letterSpacing: "0.08em", color: "var(--ip-text-dim)", marginBottom: 16 }}>CURRENT SETTINGS (read-only)</div>
          {[
            ["Officer count",             String(defaults.officerCount)],
            ["Demand weighting",          defaults.demandWeight.toFixed(2)],
            ["Flow weighting",            defaults.flowWeight.toFixed(2)],
            ["Max officers per location", String(defaults.maxOfficersPerLocation)],
            ["Min priority threshold",    String(defaults.minimumPriorityThreshold)],
          ].map(([k, v]) => (
            <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "8px 0", borderBottom: "1px solid var(--ip-hairline)", fontSize: "0.84rem" }}>
              <span style={{ color: "var(--ip-text-dim)" }}>{k}</span>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600, color: "var(--ip-text)" }}>{v}</span>
            </div>
          ))}
        </div>
        <div style={{ background: "var(--ip-sim-bg)", border: "1px solid var(--ip-hairline)", borderRadius: 8, padding: "20px 22px" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", letterSpacing: "0.08em", color: "var(--ip-yellow)", marginBottom: 16 }}>TRY DIFFERENT SETTINGS</div>
          <SliderRow label="Officer count"        value={inputs.officerCount}             min={5}   max={100} step={1}    onChange={v => setInput("officerCount", v)} />
          <SliderRow label="Demand weighting"     value={inputs.demandWeight}             min={0.0} max={1.0} step={0.05} onChange={v => setInput("demandWeight", v)}             fmt={v => v.toFixed(2)} />
          <SliderRow label="Flow weighting"       value={inputs.flowWeight}               min={0.0} max={1.0} step={0.05} onChange={v => setInput("flowWeight", v)}               fmt={v => v.toFixed(2)} />
          <SliderRow label="Max per location"     value={inputs.maxOfficersPerLocation}   min={1}   max={8}   step={1}    onChange={v => setInput("maxOfficersPerLocation", v)} />
          <SliderRow label="Min threshold"        value={inputs.minimumPriorityThreshold} min={10}  max={80}  step={5}    onChange={v => setInput("minimumPriorityThreshold", v)} />
        </div>
      </div>

      <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
        <DeltaBlock label="Locations covered"        base={base.locationsCovered}       sim={result.locationsCovered} />
        <DeltaBlock label="Priority score addressed" base={base.totalPriorityScore}     sim={result.totalPriorityScore} />
        <DeltaBlock label="Coverage utilization"     base={base.coverageUtilizationPct} sim={result.coverageUtilizationPct} unit="%" />
      </div>

      <div style={{ padding: "12px 16px", background: "rgba(138,147,143,0.08)", border: "1px solid var(--ip-hairline)", borderRadius: 6, fontSize: "0.8rem", color: "var(--ip-text-dim)", lineHeight: 1.6 }}>
        <strong style={{ color: "var(--ip-text)" }}>Disclaimer:</strong> This simulator re-runs the scoring formula with the inputs you set. It shows how the recommendation would shift under different resourcing — it does not predict real-world enforcement outcomes or model second-order effects.
      </div>
    </div>
  );
}

/* =========================================================================
   SECTION 5 — KNOWN LIMITATIONS
   ========================================================================= */

const STATIC_LIMITATIONS = [
  { tag: "DATA WINDOW",       headline: "Dataset ends April 2024 — forecast mode is a demonstration, not a live feed.",               detail: "Forecast mode projects the week immediately following the dataset's cutoff. It illustrates the forecasting mechanism; it is not connected to a live data pipeline.", border: "#F5C518" },
  { tag: "FLOW METRIC",       headline: "Traffic-flow impact is inferred, not measured.",                                              detail: "No physical traffic sensors exist in this dataset. The obstruction-proxy score is computed from violation type and vehicle size — an explicitly labelled stand-in for sensor data.", border: "#FF4D2E" },
  { tag: "JUNCTION MATCHING", headline: "~40% of grid cells have no matched place name.",                                             detail: "Location names are matched from junction_name strings using keyword search. About 60% of zones get a real name; the rest remain UNCLASSIFIED with only a coordinate reference.", border: "#8A938F" },
  { tag: "MODEL ASSUMPTIONS", headline: "Scores are formula outputs, not causal predictions.",                                         detail: "The priority score is a weighted composite of demand, flow, junction type, and repeat-offender signals. It ranks locations relative to each other — not an absolute count of future violations.", border: "#8A938F" },
  { tag: "SIMULATOR",         headline: "Policy simulator re-runs the formula, not the real world.",                                  detail: "Changing sliders shows how the scoring algorithm re-distributes officers under different inputs. It does not model officer behaviour, traffic response, or second-order effects.", border: "#39B88A" },
];

function KnownLimitations() {
  const [data, setData] = useState<LimitationsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchLimitations().then(setData).finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingState label="Loading model evidence…" />;

  const fiData = (data?.featureImportance ?? [])
    .filter(f => f.model === "violation_count")
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 8);

  const qualityMetrics = data?.dataQuality ?? [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 14 }}>
        {STATIC_LIMITATIONS.map(lim => (
          <div key={lim.tag} style={{ background: "var(--ip-surface)", border: "1px solid var(--ip-hairline)", borderLeft: `3px solid ${lim.border}`, borderRadius: 8, padding: "20px 18px" }}>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.66rem", letterSpacing: "0.07em", color: lim.border, marginBottom: 10 }}>{lim.tag}</div>
            <div style={{ fontWeight: 600, color: "var(--ip-text)", fontSize: "0.9rem", marginBottom: 8, lineHeight: 1.4 }}>{lim.headline}</div>
            <div style={{ fontSize: "0.84rem", color: "var(--ip-text-dim)", lineHeight: 1.6 }}>{lim.detail}</div>
          </div>
        ))}
      </div>

      <div style={{ background: "var(--ip-surface)", border: "1px solid var(--ip-hairline)", borderRadius: 8, padding: "20px 22px" }}>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", letterSpacing: "0.07em", color: "var(--ip-text-dim)", marginBottom: 16 }}>
          MODEL EVIDENCE — what the forecasting model actually relies on
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 24, alignItems: "start" }}>
          <div>
            <div style={{ fontSize: "0.82rem", color: "var(--ip-text)", marginBottom: 12 }}>Feature importance (violation count model · normalised)</div>
            {fiData.length > 0 ? (
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={fiData} layout="vertical" margin={{ left: 0, right: 20 }}>
                  <XAxis type="number" hide />
                  <YAxis type="category" dataKey="feature" width={210}
                    tick={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--ip-text-dim)" }}
                    axisLine={false} tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{ background: "var(--ip-surface)", border: "1px solid var(--ip-hairline)", fontFamily: "var(--font-mono)", fontSize: 12 }}
                    formatter={(v: unknown) => [`${((v as number) * 100).toFixed(0)}%`, "weight"]}
                    labelStyle={{ color: "var(--ip-text)" }}
                  />
                  <Bar dataKey="importance" fill="var(--ip-yellow)" radius={3} barSize={12} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.78rem", color: "var(--ip-text-dim)", padding: "20px 0" }}>
                Run module4_hotspot_forecast/train_model.py to generate feature_importance.csv
              </div>
            )}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 12, minWidth: 200 }}>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem", letterSpacing: "0.06em", color: "var(--ip-text-dim)", marginBottom: 4 }}>DATA QUALITY METRICS</div>
            {qualityMetrics.map(({ label, value }) => (
              <div key={label} style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
                <span style={{ fontSize: "0.8rem", color: "var(--ip-text-dim)" }}>{label}</span>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.84rem", fontWeight: 600, color: "var(--ip-text)" }}>{value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* =========================================================================
   ROOT APP
   ========================================================================= */

const NAV_ITEMS = [
  { id: "hotspot",     label: "Hotspot Map",       icon: <MapPin size={15} /> },
  { id: "allocation",  label: "Officer Allocation", icon: <Users size={15} /> },
  { id: "offenders",   label: "Repeat Offenders",   icon: <Car size={15} /> },
  { id: "simulator",   label: "Policy Simulator",   icon: <Sliders size={15} /> },
  { id: "limitations", label: "Known Limitations",  icon: <Info size={15} /> },
] as const;

type Section = typeof NAV_ITEMS[number]["id"];

const SECTION_META: Record<Section, { eyebrow: string; title: string; sub: string }> = {
  hotspot:     { eyebrow: "SECTION 1", title: "Hotspot Map",        sub: "Where to send people, and why — priority intensity over Bengaluru." },
  allocation:  { eyebrow: "SECTION 2", title: "Officer Allocation",  sub: "How today's shift strength is distributed across active zones." },
  offenders:   { eyebrow: "SECTION 3", title: "Repeat Offenders",    sub: "Vehicles with habitual violation patterns requiring active enforcement." },
  simulator:   { eyebrow: "SECTION 4", title: "Policy Simulator",    sub: "What-if tool — adjust resourcing rules and see the modelled outcome." },
  limitations: { eyebrow: "SECTION 5", title: "Known Limitations",   sub: "What the model can and cannot tell you. Read before trusting a number." },
};

export default function App() {
  const [dark, setDark] = useState(true);
  const [section, setSection] = useState<Section>("hotspot");
  const [mode, setMode] = useState<DeployMode>("historical_gap");

  const meta = SECTION_META[section];

  return (
    <div className={dark ? "dark" : ""} style={{ display: "flex", height: "100vh", overflow: "hidden", background: "var(--ip-base)", color: "var(--ip-text)", fontFamily: "var(--font-sans)", ...cssVars(dark) }}>

      {/* ===== SIDEBAR ===== */}
      <aside style={{
        width: 228, flexShrink: 0, background: "var(--ip-surface)",
        borderRight: "1px solid var(--ip-hairline)",
        display: "flex", flexDirection: "column", height: "100vh", overflow: "hidden",
      }}>
        <div style={{ padding: "20px 16px 16px", borderBottom: "1px solid var(--ip-hairline)" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontWeight: 700, fontSize: "0.92rem", letterSpacing: "0.08em", color: "var(--ip-text)", marginBottom: 2 }}>INTELLIPARK</div>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", letterSpacing: "0.05em", color: "var(--ip-text-dim)" }}>ENFORCEMENT INTELLIGENCE</div>
        </div>

        <div style={{ padding: "14px 12px", borderBottom: "1px solid var(--ip-hairline)" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", letterSpacing: "0.06em", color: "var(--ip-text-dim)", marginBottom: 8 }}>DATASET MODE</div>
          <SegControl
            options={["Past perf.", "Next week"]}
            value={mode === "historical_gap" ? "Past perf." : "Next week"}
            onChange={v => setMode(v === "Past perf." ? "historical_gap" : "future_forecast")}
          />
        </div>

        <nav style={{ flex: 1, padding: "10px 8px", overflowY: "auto" }}>
          {NAV_ITEMS.map(item => (
            <NavItem key={item.id} icon={item.icon} label={item.label} active={section === item.id} onClick={() => setSection(item.id)} />
          ))}
        </nav>

        <div style={{ padding: "12px 14px", borderTop: "1px solid var(--ip-hairline)", fontFamily: "var(--font-mono)", fontSize: "0.66rem", color: "var(--ip-text-dim)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 4 }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#39B88A", boxShadow: "0 0 5px #39B88A", display: "inline-block", flexShrink: 0 }} />
            Pipeline synced
          </div>
          <div style={{ opacity: 0.7 }}>Apr 28, 2024 · 00:00 IST</div>
        </div>
      </aside>

      {/* ===== MAIN ===== */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", height: "100vh", overflow: "hidden" }}>
        <header style={{
          padding: "0 28px", height: 56, borderBottom: "1px solid var(--ip-hairline)",
          display: "flex", alignItems: "center", justifyContent: "space-between",
          flexShrink: 0, background: "var(--ip-surface)",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", letterSpacing: "0.06em", color: "var(--ip-text-dim)" }}>{meta.eyebrow}</span>
            <span style={{ color: "var(--ip-hairline)" }}>·</span>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem", fontWeight: 600, color: "var(--ip-text)" }}>{meta.title}</span>
            {mode === "future_forecast" && section !== "simulator" && section !== "limitations" && (
              <span style={{ background: "rgba(245,197,24,0.12)", color: "var(--ip-yellow)", fontFamily: "var(--font-mono)", fontSize: "0.64rem", fontWeight: 700, letterSpacing: "0.06em", padding: "2px 8px", borderRadius: 4 }}>
                FORECAST
              </span>
            )}
          </div>
          <button
            onClick={() => setDark(d => !d)}
            style={{
              background: "var(--ip-surface-2)", border: "1px solid var(--ip-hairline)",
              borderRadius: 7, padding: "6px 10px", cursor: "pointer",
              color: "var(--ip-text-dim)", display: "flex", alignItems: "center", gap: 6,
              fontFamily: "var(--font-mono)", fontSize: "0.72rem", transition: "all 0.15s",
            }}
          >
            {dark ? <Sun size={13} /> : <Moon size={13} />}
            {dark ? "Light" : "Dark"}
          </button>
        </header>

        <div style={{ padding: "18px 28px 14px", borderBottom: "1px solid var(--ip-hairline)", flexShrink: 0, background: "var(--ip-base)" }}>
          <SectionEyebrow>{meta.eyebrow} — {meta.title.toUpperCase()}</SectionEyebrow>
          <SectionTitle>{meta.title}</SectionTitle>
          <p style={{ fontSize: "0.85rem", color: "var(--ip-text-dim)", margin: 0, maxWidth: 600 }}>{meta.sub}</p>
        </div>

        <div style={{ flex: 1, overflowY: "auto", padding: "24px 28px", background: "var(--ip-base)" }}>
          {section === "hotspot"     && <HotspotMap mode={mode} dark={dark} />}
          {section === "allocation"  && <OfficerAllocation mode={mode} />}
          {section === "offenders"   && <RepeatOffenders />}
          {section === "simulator"   && <PolicySimulator />}
          {section === "limitations" && <KnownLimitations />}
        </div>
      </main>
    </div>
  );
}

