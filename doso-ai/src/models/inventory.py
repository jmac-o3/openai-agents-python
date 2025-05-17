from typing import Dict, List

from pydantic import BaseModel


class InventoryMetrics(BaseModel):
    """
    Key metrics for inventory analysis
    """

    total_units: int
    total_value: float
    average_days_supply: float
    turnover_rate: float
    aging_distribution: Dict[str, int]


class InventoryAnalysis(BaseModel):
    """
    Comprehensive inventory analysis results
    """

    metrics: InventoryMetrics
    insights: List[str]
    risk_factors: List[str]
    opportunities: List[str]
