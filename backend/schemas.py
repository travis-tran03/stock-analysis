"""Pydantic models for analyze request and per-stock trade analysis responses."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TradeDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class EntryRange(BaseModel):
    low: Optional[float] = None
    high: Optional[float] = None


class PremarketAnalysis(BaseModel):
    premarket_price: Optional[float] = None
    premarket_change_percent: Optional[float] = None
    premarket_high: Optional[float] = None
    premarket_low: Optional[float] = None
    premarket_volume: Optional[float] = None
    previous_close: Optional[float] = None
    adjusted_entry_range: list[Optional[float]] = Field(default_factory=list)
    premarket_signal: str = "NEUTRAL"
    note: str = ""


class AfterhoursAnalysis(BaseModel):
    afterhours_price: Optional[float] = None
    afterhours_change_percent: Optional[float] = None
    afterhours_high: Optional[float] = None
    afterhours_low: Optional[float] = None
    afterhours_volume: Optional[float] = None
    regular_close: Optional[float] = None
    adjusted_entry_range: list[Optional[float]] = Field(default_factory=list)
    afterhours_signal: str = "NEUTRAL"
    note: str = ""


class RiskReward(BaseModel):
    ratio: Optional[float] = None
    label: str = ""


class AnalyzeRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1, description="Stock symbols, e.g. TSLA, AAPL")


class StockAnalysis(BaseModel):
    ticker: str
    direction: TradeDirection
    confidence: float = Field(ge=0.0, le=1.0)
    entry_range: Optional[EntryRange] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profits: list[float] = Field(default_factory=list)
    risk_reward: RiskReward
    premarket_analysis: Optional[PremarketAnalysis] = None
    afterhours_analysis: Optional[AfterhoursAnalysis] = None
    rationale: str = ""
    summary_points: list[str] = Field(default_factory=list)
    monitoring_triggers: list[str] = Field(default_factory=list)
    plan_steps: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    results: list[StockAnalysis]
    errors: list[str] = Field(default_factory=list)
