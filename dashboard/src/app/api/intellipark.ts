/**
 * IntelliPark API Layer
 *
 * Maps to the outputs of the Python pipeline modules:
 *
 *   Module 1  → module1_pipeline/output/cleaned.parquet
 *   Module 2  → module2_impact_score/output/scored.parquet
 *   Module 3  → module3_repeat_offender/output/vehicle_risk.parquet
 *               module3_repeat_offender/output/risk_tagged.parquet
 *   Module 4  → module4_hotspot_forecast/output/grid_forecasts.parquet
 *               module4_hotspot_forecast/output/future_grid_forecasts.parquet
 *               module4_hotspot_forecast/output/feature_importance.csv
 *               module4_hotspot_forecast/output/training_data.parquet
 *   Module 5  → module5_edi/output/edi_scores.parquet            (historical)
 *               module5_edi/output/future_priority_scores.parquet (forecast)
 *               module5_edi/output/edi_scores_with_flow.parquet
 *               module5_edi/output/future_priority_with_flow.parquet
 *   Module 6  → module6_optimizer/output/allocations_historical_gap.json
 *               module6_optimizer/output/allocations_future_forecast.json
 *
 * In production, replace each `resolve*` function body with a real
 * fetch() call to your backend server / FastAPI / Flask endpoint that
 * reads the parquet/JSON files and returns JSON.  The shape of every
 * return type in this file defines the contract those endpoints must
 * honour — nothing else in the UI needs to change.
 *
 * During development (or demo mode with no live server) the functions
 * return the sample data below, so the UI is always functional.
 */

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

export type ZoneColor = "RED" | "YELLOW" | "GREEN";
export type RiskLevel = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
export type DeployMode = "historical_gap" | "future_forecast";

// ---------------------------------------------------------------------------
// Section 1 — Hotspot Map
// ---------------------------------------------------------------------------

export interface HotspotLocation {
  /** grid_cell_id from the pipeline, e.g. "12.97_77.59" */
  gridCellId: string;
  /** Human-readable junction name from location_context.py */
  junctionName: string;
  /** WGS-84 coordinates */
  lat: number;
  lng: number;
  zone: ZoneColor;
  /** module5 edi_priority / priority_score (0-100) */
  priorityScore: number;
  /** module4 predicted_violations */
  predictedViolations: number;
  /** module5 observed_violations (null in forecast mode — not yet real) */
  observedViolations: number | null;
  /** module5 predicted_flow_impact */
  flowImpact: number;
  /** module6 officers_allocated */
  recommendedPatrolUnits: number;
  /** module5 location_context */
  locationContext: string;
  /** Score breakdown from module2 weights — sums to priorityScore */
  scoreBreakdown: {
    junctionType: number;
    violationSeverity: number;
    vehicleSize: number;
    peakHour: number;
    hotspotDensity: number;
    repeatOffenders: number;
  };
}

export interface HotspotResponse {
  mode: DeployMode;
  /** ISO-8601 timestamp of last pipeline run */
  dataAsOf: string;
  locations: HotspotLocation[];
}

// ---------------------------------------------------------------------------
// Section 2 — Officer Allocation
// ---------------------------------------------------------------------------

export interface AllocationSummary {
  availableOfficers: number;
  locationsCovered: number;
  totalEligibleLocations: number;
  totalPriorityScore: number;
}

export interface AllocationRow {
  gridCellId: string;
  junctionName: string;
  zone: ZoneColor;
  dateHour: string;
  officersAllocated: number;
  expectedPriorityScore: number;
}

export interface ZoneOfficers {
  zone: ZoneColor;
  officers: number;
}

export interface AllocationResponse {
  mode: DeployMode;
  summary: AllocationSummary;
  byZone: ZoneOfficers[];
  allocations: AllocationRow[];
}

// ---------------------------------------------------------------------------
// Section 3 — Repeat Offenders
// ---------------------------------------------------------------------------

export interface RiskTierSummary {
  tier: RiskLevel;
  count: number;
  percentage: number;
}

export interface VehicleRisk {
  vehicleNumber: string;
  totalViolations: number;
  criticalRatio: number;
  topGridConcentration: number;
  meanImpactScore: number;
  riskLevel: RiskLevel;
  recommendedAction: "IMMEDIATE_TOW" | "PRIORITY_TICKET" | "MONITOR" | "IGNORE";
  /** Pre-built plain-English explanation from vehicle_risk.py */
  riskExplanation: string;
}

export interface RepeatOffendersResponse {
  tierSummary: RiskTierSummary[];
  criticalVehicles: VehicleRisk[];
}

// ---------------------------------------------------------------------------
// Section 4 — Policy Simulator
// ---------------------------------------------------------------------------

export interface SimulatorDefaults {
  officerCount: number;
  demandWeight: number;
  flowWeight: number;
  maxOfficersPerLocation: number;
  minimumPriorityThreshold: number;
  baseResult: {
    locationsCovered: number;
    totalPriorityScore: number;
    coverageUtilizationPct: number;
  };
}

// The simulator computes its results client-side using the formula below.
// Pass SimulatorInputs to `simulateAllocation` to get a SimulatorResult.
export interface SimulatorInputs {
  officerCount: number;
  demandWeight: number;
  flowWeight: number;
  maxOfficersPerLocation: number;
  minimumPriorityThreshold: number;
}

export interface SimulatorResult {
  locationsCovered: number;
  totalPriorityScore: number;
  coverageUtilizationPct: number;
}

// ---------------------------------------------------------------------------
// Section 5 — Known Limitations
// ---------------------------------------------------------------------------

export interface FeatureImportanceRow {
  feature: string;
  importance: number;
  model: "violation_count" | "impact_score";
}

export interface DataQualityMetric {
  label: string;
  value: string;
}

export interface LimitationsResponse {
  featureImportance: FeatureImportanceRow[];
  dataQuality: DataQualityMetric[];
}

// ---------------------------------------------------------------------------
// API CLIENT
// ---------------------------------------------------------------------------

/**
 * Base URL for the backend server that reads the pipeline parquet/JSON outputs.
 *
 * Point this at your FastAPI / Flask / Express server.
 * Example:  const BASE = "http://localhost:8000/api"
 *
 * If BASE is empty ("") all calls fall through to the mock data below —
 * useful for design/demo mode without a running server.
 */
const BASE = (import.meta.env?.VITE_API_BASE as string) ?? "";

async function fetchOrMock<T>(
  path: string,
  mockFn: () => T
): Promise<T> {
  if (!BASE) return mockFn();
  try {
    const res = await fetch(`${BASE}${path}`);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return (await res.json()) as T;
  } catch (err) {
    console.warn(
      `[IntelliPark API] Failed to reach ${BASE}${path} — falling back to mock data.`,
      err
    );
    return mockFn();
  }
}

// ---------------------------------------------------------------------------
// PUBLIC API FUNCTIONS
// These are what the React components call.
// ---------------------------------------------------------------------------

/**
 * Hotspot map data.
 * Backend reads:
 *   historical_gap → edi_scores_with_flow.parquet + location_context join
 *   future_forecast → future_priority_with_flow.parquet + location_context join
 */
export async function fetchHotspots(mode: DeployMode): Promise<HotspotResponse> {
  return fetchOrMock(`/hotspots?mode=${mode}`, () => mockHotspots(mode));
}

/**
 * Officer allocation data.
 * Backend reads:
 *   allocations_historical_gap.json  OR  allocations_future_forecast.json
 *   + junction name join from cleaned.parquet
 */
export async function fetchAllocations(mode: DeployMode): Promise<AllocationResponse> {
  return fetchOrMock(`/allocations?mode=${mode}`, () => mockAllocations(mode));
}

/**
 * Repeat offenders.
 * Backend reads: vehicle_risk.parquet
 */
export async function fetchRepeatOffenders(): Promise<RepeatOffendersResponse> {
  return fetchOrMock(`/repeat-offenders`, () => mockRepeatOffenders());
}

/**
 * Default config values for the Policy Simulator.
 * Backend reads: module6_optimizer/config.json
 */
export async function fetchSimulatorDefaults(): Promise<SimulatorDefaults> {
  return fetchOrMock(`/simulator-defaults`, () => mockSimulatorDefaults());
}

/**
 * Feature importance + data quality for Section 5.
 * Backend reads: feature_importance.csv + clean_quality.json
 */
export async function fetchLimitations(): Promise<LimitationsResponse> {
  return fetchOrMock(`/limitations`, () => mockLimitations());
}

// ---------------------------------------------------------------------------
// CLIENT-SIDE SIMULATOR FORMULA
// Mirrors the logic in module6_optimizer/optimizer.py:compute_base_impact
// and allocate_officers, simplified for instant interactivity.
// ---------------------------------------------------------------------------

export function simulateAllocation(
  defaults: SimulatorDefaults,
  inputs: SimulatorInputs
): SimulatorResult {
  const totalEligible = 90; // matches mock data — replace with live value

  // Officers drive coverage non-linearly (diminishing returns past ~1.7 per slot)
  const rawCovered = Math.min(
    totalEligible,
    Math.round(inputs.officerCount * (1.4 + inputs.demandWeight * 0.6))
  );

  // Score is weighted sum of demand + flow contributions
  const rawScore = Math.round(
    inputs.officerCount * 70 * inputs.demandWeight +
    inputs.officerCount * 30 * inputs.flowWeight
  );

  // Threshold raises the bar — fewer eligible slots means fewer locations covered
  const thresholdPenalty = Math.max(
    0,
    (inputs.minimumPriorityThreshold - defaults.minimumPriorityThreshold) / 100
  );

  const adjustedCovered = Math.max(
    1,
    Math.round(rawCovered * (1 - thresholdPenalty * 0.5))
  );

  return {
    locationsCovered: adjustedCovered,
    totalPriorityScore: rawScore,
    coverageUtilizationPct: Math.min(
      100,
      Math.round((adjustedCovered / totalEligible) * 100)
    ),
  };
}

// ===========================================================================
// MOCK DATA
// Shapes match real pipeline outputs — wire to live endpoints by replacing
// each mockFn() call in fetchOrMock with real parquet-backed responses.
// ===========================================================================

function mockHotspots(mode: DeployMode): HotspotResponse {
  const isFuture = mode === "future_forecast";
  return {
    mode,
    dataAsOf: "2024-04-28T00:00:00+05:30",
    locations: [
      {
        gridCellId: "12.96_77.58",
        junctionName: "KR Market Junction",
        lat: 12.9639,
        lng: 77.5834,
        zone: "RED",
        priorityScore: 94,
        predictedViolations: 142,
        observedViolations: isFuture ? null : 119,
        flowImpact: 18.4,
        recommendedPatrolUnits: 4,
        locationContext: "COMMERCIAL",
        scoreBreakdown: { junctionType: 22, violationSeverity: 19, vehicleSize: 15, peakHour: 18, hotspotDensity: 13, repeatOffenders: 7 },
      },
      {
        gridCellId: "12.91_77.62",
        junctionName: "Silk Board Junction",
        lat: 12.9176,
        lng: 77.6233,
        zone: "RED",
        priorityScore: 89,
        predictedViolations: 128,
        observedViolations: isFuture ? null : 101,
        flowImpact: 16.1,
        recommendedPatrolUnits: 3,
        locationContext: "TRANSIT_HUB",
        scoreBreakdown: { junctionType: 18, violationSeverity: 21, vehicleSize: 14, peakHour: 17, hotspotDensity: 14, repeatOffenders: 5 },
      },
      {
        gridCellId: "12.93_77.62",
        junctionName: "Ejipura Main Road",
        lat: 12.9334,
        lng: 77.6218,
        zone: "RED",
        priorityScore: 85,
        predictedViolations: 117,
        observedViolations: isFuture ? null : 98,
        flowImpact: 14.7,
        recommendedPatrolUnits: 3,
        locationContext: "COMMERCIAL",
        scoreBreakdown: { junctionType: 17, violationSeverity: 18, vehicleSize: 16, peakHour: 16, hotspotDensity: 12, repeatOffenders: 6 },
      },
      {
        gridCellId: "12.98_77.60",
        junctionName: "Shivajinagar Bus Terminal",
        lat: 12.985,
        lng: 77.6001,
        zone: "RED",
        priorityScore: 82,
        predictedViolations: 109,
        observedViolations: isFuture ? null : 94,
        flowImpact: 13.9,
        recommendedPatrolUnits: 3,
        locationContext: "TRANSIT_HUB",
        scoreBreakdown: { junctionType: 20, violationSeverity: 16, vehicleSize: 13, peakHour: 15, hotspotDensity: 11, repeatOffenders: 7 },
      },
      {
        gridCellId: "12.97_77.60",
        junctionName: "MG Road / Brigade Rd Junction",
        lat: 12.9716,
        lng: 77.6059,
        zone: "YELLOW",
        priorityScore: 74,
        predictedViolations: 94,
        observedViolations: isFuture ? null : 88,
        flowImpact: 11.2,
        recommendedPatrolUnits: 2,
        locationContext: "COMMERCIAL",
        scoreBreakdown: { junctionType: 16, violationSeverity: 15, vehicleSize: 12, peakHour: 14, hotspotDensity: 11, repeatOffenders: 6 },
      },
      {
        gridCellId: "13.03_77.59",
        junctionName: "Hebbal Flyover Junction",
        lat: 13.0358,
        lng: 77.597,
        zone: "YELLOW",
        priorityScore: 69,
        predictedViolations: 87,
        observedViolations: isFuture ? null : 72,
        flowImpact: 10.1,
        recommendedPatrolUnits: 2,
        locationContext: "TRANSIT_HUB",
        scoreBreakdown: { junctionType: 14, violationSeverity: 14, vehicleSize: 11, peakHour: 13, hotspotDensity: 10, repeatOffenders: 7 },
      },
      {
        gridCellId: "12.93_77.62",
        junctionName: "Koramangala 5th Block Jn",
        lat: 12.9352,
        lng: 77.6245,
        zone: "YELLOW",
        priorityScore: 65,
        predictedViolations: 82,
        observedViolations: isFuture ? null : 79,
        flowImpact: 9.4,
        recommendedPatrolUnits: 2,
        locationContext: "COMMERCIAL",
        scoreBreakdown: { junctionType: 13, violationSeverity: 13, vehicleSize: 12, peakHour: 12, hotspotDensity: 9, repeatOffenders: 6 },
      },
      {
        gridCellId: "13.10_77.59",
        junctionName: "Yelahanka Cross",
        lat: 13.1004,
        lng: 77.5963,
        zone: "YELLOW",
        priorityScore: 58,
        predictedViolations: 71,
        observedViolations: isFuture ? null : 65,
        flowImpact: 7.8,
        recommendedPatrolUnits: 2,
        locationContext: "TRANSIT_HUB",
        scoreBreakdown: { junctionType: 12, violationSeverity: 11, vehicleSize: 10, peakHour: 11, hotspotDensity: 8, repeatOffenders: 6 },
      },
      {
        gridCellId: "12.89_77.59",
        junctionName: "Bannerghatta Road Jn",
        lat: 12.8933,
        lng: 77.5955,
        zone: "GREEN",
        priorityScore: 47,
        predictedViolations: 58,
        observedViolations: isFuture ? null : 60,
        flowImpact: 5.9,
        recommendedPatrolUnits: 1,
        locationContext: "OTHER_JUNCTION",
        scoreBreakdown: { junctionType: 10, violationSeverity: 9, vehicleSize: 8, peakHour: 9, hotspotDensity: 7, repeatOffenders: 4 },
      },
      {
        gridCellId: "13.02_77.55",
        junctionName: "Yeshwanthpur Circle",
        lat: 13.0214,
        lng: 77.556,
        zone: "GREEN",
        priorityScore: 39,
        predictedViolations: 49,
        observedViolations: isFuture ? null : 51,
        flowImpact: 4.7,
        recommendedPatrolUnits: 1,
        locationContext: "TRANSIT_HUB",
        scoreBreakdown: { junctionType: 8, violationSeverity: 8, vehicleSize: 7, peakHour: 7, hotspotDensity: 5, repeatOffenders: 4 },
      },
    ],
  };
}

function mockAllocations(mode: DeployMode): AllocationResponse {
  return {
    mode,
    summary: {
      availableOfficers: 50,
      locationsCovered: 42,
      totalEligibleLocations: 90,
      totalPriorityScore: 3840,
    },
    byZone: [
      { zone: "RED", officers: 26 },
      { zone: "YELLOW", officers: 16 },
      { zone: "GREEN", officers: 8 },
    ],
    allocations: [
      { gridCellId: "12.96_77.58", junctionName: "KR Market Junction",          zone: "RED",    dateHour: "2024-04-28T08:00", officersAllocated: 4, expectedPriorityScore: 376 },
      { gridCellId: "12.91_77.62", junctionName: "Silk Board Junction",          zone: "RED",    dateHour: "2024-04-28T08:00", officersAllocated: 3, expectedPriorityScore: 267 },
      { gridCellId: "12.93_77.62", junctionName: "Ejipura Main Road",            zone: "RED",    dateHour: "2024-04-28T09:00", officersAllocated: 3, expectedPriorityScore: 255 },
      { gridCellId: "12.98_77.60", junctionName: "Shivajinagar Bus Terminal",    zone: "RED",    dateHour: "2024-04-28T10:00", officersAllocated: 3, expectedPriorityScore: 246 },
      { gridCellId: "12.97_77.60", junctionName: "MG Road / Brigade Rd Junction",zone: "YELLOW", dateHour: "2024-04-28T08:00", officersAllocated: 2, expectedPriorityScore: 148 },
      { gridCellId: "13.03_77.59", junctionName: "Hebbal Flyover Junction",      zone: "YELLOW", dateHour: "2024-04-28T08:00", officersAllocated: 2, expectedPriorityScore: 138 },
      { gridCellId: "12.93_77.62", junctionName: "Koramangala 5th Block Jn",    zone: "YELLOW", dateHour: "2024-04-28T17:00", officersAllocated: 2, expectedPriorityScore: 130 },
      { gridCellId: "13.10_77.59", junctionName: "Yelahanka Cross",              zone: "YELLOW", dateHour: "2024-04-28T08:00", officersAllocated: 2, expectedPriorityScore: 116 },
      { gridCellId: "12.89_77.59", junctionName: "Bannerghatta Road Jn",         zone: "GREEN",  dateHour: "2024-04-28T16:00", officersAllocated: 1, expectedPriorityScore: 47 },
      { gridCellId: "13.02_77.55", junctionName: "Yeshwanthpur Circle",          zone: "GREEN",  dateHour: "2024-04-28T08:00", officersAllocated: 1, expectedPriorityScore: 39 },
      { gridCellId: "12.98_77.56", junctionName: "Indiranagar 100ft Road",       zone: "GREEN",  dateHour: "2024-04-28T09:00", officersAllocated: 1, expectedPriorityScore: 35 },
      { gridCellId: "12.92_77.58", junctionName: "Jayanagar 4th Block",          zone: "GREEN",  dateHour: "2024-04-28T10:00", officersAllocated: 1, expectedPriorityScore: 31 },
    ],
  };
}

function mockRepeatOffenders(): RepeatOffendersResponse {
  return {
    tierSummary: [
      { tier: "CRITICAL", count: 142,  percentage: 8  },
      { tier: "HIGH",     count: 338,  percentage: 19 },
      { tier: "MEDIUM",   count: 551,  percentage: 31 },
      { tier: "LOW",      count: 746,  percentage: 42 },
    ],
    criticalVehicles: [
      { vehicleNumber: "KA 01 AB 1234", totalViolations: 18, criticalRatio: 0.61, topGridConcentration: 0.82, meanImpactScore: 84.2, riskLevel: "CRITICAL", recommendedAction: "IMMEDIATE_TOW",  riskExplanation: "Violations=18; CriticalRatio=0.61; Recurrence=0.82; MeanImpact=84.2" },
      { vehicleNumber: "KA 05 MN 7722", totalViolations: 16, criticalRatio: 0.56, topGridConcentration: 0.75, meanImpactScore: 79.5, riskLevel: "CRITICAL", recommendedAction: "IMMEDIATE_TOW",  riskExplanation: "Violations=16; CriticalRatio=0.56; Recurrence=0.75; MeanImpact=79.5" },
      { vehicleNumber: "KA 03 CX 4401", totalViolations: 14, criticalRatio: 0.45, topGridConcentration: 0.60, meanImpactScore: 72.3, riskLevel: "CRITICAL", recommendedAction: "PRIORITY_TICKET",riskExplanation: "Violations=14; CriticalRatio=0.45; Recurrence=0.60; MeanImpact=72.3" },
      { vehicleNumber: "KA 51 ZZ 9981", totalViolations: 13, criticalRatio: 0.54, topGridConcentration: 0.70, meanImpactScore: 68.8, riskLevel: "CRITICAL", recommendedAction: "PRIORITY_TICKET",riskExplanation: "Violations=13; CriticalRatio=0.54; Recurrence=0.70; MeanImpact=68.8" },
      { vehicleNumber: "KA 04 PQ 3315", totalViolations: 12, criticalRatio: 0.42, topGridConcentration: 0.55, meanImpactScore: 63.1, riskLevel: "CRITICAL", recommendedAction: "MONITOR",        riskExplanation: "Violations=12; CriticalRatio=0.42; Recurrence=0.55; MeanImpact=63.1" },
      { vehicleNumber: "KA 09 RS 0041", totalViolations: 12, criticalRatio: 0.50, topGridConcentration: 0.65, meanImpactScore: 61.7, riskLevel: "CRITICAL", recommendedAction: "PRIORITY_TICKET",riskExplanation: "Violations=12; CriticalRatio=0.50; Recurrence=0.65; MeanImpact=61.7" },
    ],
  };
}

function mockSimulatorDefaults(): SimulatorDefaults {
  return {
    officerCount: 50,
    demandWeight: 0.25,
    flowWeight: 0.15,
    maxOfficersPerLocation: 2,
    minimumPriorityThreshold: 40,
    baseResult: {
      locationsCovered: 42,
      totalPriorityScore: 3840,
      coverageUtilizationPct: 47,
    },
  };
}

function mockLimitations(): LimitationsResponse {
  return {
    featureImportance: [
      { feature: "violation_count_lag_1",    importance: 0.24, model: "violation_count" },
      { feature: "violation_count_rolling_24",importance: 0.18, model: "violation_count" },
      { feature: "avg_impact_score_lag_1",   importance: 0.15, model: "violation_count" },
      { feature: "critical_ratio",           importance: 0.13, model: "violation_count" },
      { feature: "repeat_offender_ratio",    importance: 0.10, model: "violation_count" },
      { feature: "hour_sin",                 importance: 0.09, model: "violation_count" },
      { feature: "dow_sin",                  importance: 0.07, model: "violation_count" },
      { feature: "grid_mean_count",          importance: 0.04, model: "violation_count" },
    ],
    dataQuality: [
      { label: "Mean priority score",          value: "58.4" },
      { label: "Median priority score",        value: "53.1" },
      { label: "Max priority score",           value: "94.0" },
      { label: "Grid cells with named junction", value: "~40%" },
      { label: "Dataset coverage",             value: "Nov 2023 – Apr 2024" },
      { label: "Unique grid cells",            value: "217" },
    ],
  };
}
