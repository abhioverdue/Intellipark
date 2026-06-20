"""
Pydantic response models.

Every field name and shape here is copied 1:1 from
dashboard/src/app/api/intellipark.ts so the frontend never has to change —
this file IS the contract, just expressed in Python.
"""

from typing import Literal, Optional
from pydantic import BaseModel

ZoneColor = Literal["RED", "YELLOW", "GREEN"]
RiskLevel = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
DeployMode = Literal["historical_gap", "future_forecast"]
RecommendedAction = Literal["IMMEDIATE_TOW", "PRIORITY_TICKET", "MONITOR", "IGNORE"]
ModelName = Literal["violation_count", "impact_score"]


# --- Section 1: Hotspot map -------------------------------------------------

class ScoreBreakdown(BaseModel):
    junctionType: float
    violationSeverity: float
    vehicleSize: float
    peakHour: float
    hotspotDensity: float
    repeatOffenders: float


class HotspotLocation(BaseModel):
    gridCellId: str
    junctionName: str
    lat: float
    lng: float
    zone: ZoneColor
    priorityScore: float
    predictedViolations: float
    observedViolations: Optional[float] = None
    flowImpact: float
    recommendedPatrolUnits: int
    locationContext: str
    scoreBreakdown: ScoreBreakdown


class HotspotResponse(BaseModel):
    mode: DeployMode
    dataAsOf: str
    locations: list[HotspotLocation]


# --- Section 2: Officer allocation ------------------------------------------

class AllocationSummary(BaseModel):
    availableOfficers: int
    locationsCovered: int
    totalEligibleLocations: int
    totalPriorityScore: float


class AllocationRow(BaseModel):
    gridCellId: str
    junctionName: str
    zone: ZoneColor
    dateHour: str
    officersAllocated: int
    expectedPriorityScore: float


class ZoneOfficers(BaseModel):
    zone: ZoneColor
    officers: int


class AllocationResponse(BaseModel):
    mode: DeployMode
    summary: AllocationSummary
    byZone: list[ZoneOfficers]
    allocations: list[AllocationRow]


# --- Section 3: Repeat offenders --------------------------------------------

class RiskTierSummary(BaseModel):
    tier: RiskLevel
    count: int
    percentage: float


class VehicleRisk(BaseModel):
    vehicleNumber: str
    totalViolations: int
    criticalRatio: float
    topGridConcentration: float
    meanImpactScore: float
    riskLevel: RiskLevel
    recommendedAction: RecommendedAction
    riskExplanation: str


class RepeatOffendersResponse(BaseModel):
    tierSummary: list[RiskTierSummary]
    criticalVehicles: list[VehicleRisk]


# --- Section 4: Policy simulator defaults -----------------------------------

class SimulatorBaseResult(BaseModel):
    locationsCovered: int
    totalPriorityScore: float
    coverageUtilizationPct: float


class SimulatorDefaults(BaseModel):
    officerCount: int
    demandWeight: float
    flowWeight: float
    maxOfficersPerLocation: int
    minimumPriorityThreshold: float
    baseResult: SimulatorBaseResult


# --- Section 5: Known limitations -------------------------------------------

class FeatureImportanceRow(BaseModel):
    feature: str
    importance: float
    model: ModelName


class DataQualityMetric(BaseModel):
    label: str
    value: str


class LimitationsResponse(BaseModel):
    featureImportance: list[FeatureImportanceRow]
    dataQuality: list[DataQualityMetric]
