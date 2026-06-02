from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TimelinePoint(BaseModel):
    second: float = Field(..., ge=0)
    intensity: float = Field(..., ge=0, le=1)
    focus_drift: float = Field(..., ge=0, le=1)


class RegionHeatmapRow(BaseModel):
    region: str
    values: list[float]


class ScoreBreakdown(BaseModel):
    hook: float = Field(..., ge=0, le=100)
    hold: float = Field(..., ge=0, le=100)
    consistency: float = Field(..., ge=0, le=100)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    engine: str
    duration_seconds: float = Field(..., ge=0)
    virality_score: float = Field(..., ge=0, le=100)
    hook_score: float = Field(..., ge=0, le=100)
    hold_rate: float = Field(..., ge=0, le=100)
    peak_hook_second: float = Field(..., ge=0)
    confidence: float = Field(..., ge=0, le=1)
    score_breakdown: ScoreBreakdown
    timeline: list[TimelinePoint]
    brain_heatmap: list[RegionHeatmapRow]
    highlights: list[str]
    notes: list[str]

