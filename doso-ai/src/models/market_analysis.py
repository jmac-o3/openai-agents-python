from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel


class MarketTrend(BaseModel):
    """Market trend analysis data"""

    segment: str
    trend_direction: str  # "up", "down", or "stable"
    strength: float  # 0.0 to 1.0 indicating trend strength
    key_drivers: List[str]
    confidence_score: float
    last_updated: datetime


class FastTurnData(BaseModel):
    """Fast Turn report data structure"""

    model_code: str
    model_year: int
    region: str
    days_supply: float
    turn_rate: float
    market_demand_score: float
    allocation_units: int
    dealer_orders: Optional[int]
    timestamp: datetime


class MarketAnalysis(BaseModel):
    """Complete market analysis output"""

    analyzed_segments: List[str]
    segment_trends: Dict[str, MarketTrend]
    demand_forecast: Dict[str, float]  # Segment to demand score mapping
    allocation_recommendations: Dict[str, int]  # Segment to recommended units
    risk_factors: List[str]
    opportunities: List[str]
    timestamp: datetime
    confidence_score: float
    data_freshness_score: float  # 0.0 to 1.0 indicating how recent the data is


class MarketInsight(BaseModel):
    """Individual market insight with supporting data"""

    insight_type: str  # "demand_shift", "competitive_threat", "opportunity", etc.
    segment: str
    description: str
    impact_score: float  # 0.0 to 1.0 indicating the magnitude of impact
    supporting_metrics: Dict[str, float]
    action_items: List[str]
    priority: str  # "high", "medium", "low"
    timestamp: datetime
    expiration: Optional[datetime]
    confidence_score: float
    source_data: Dict[str, Any]  # Raw data supporting this insight
