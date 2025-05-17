from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel


class InventoryGap(BaseModel):
    """
    Represents a specific gap in inventory
    """

    category: str
    gap_type: str  # "mix" or "competitive"
    current_value: int
    target_value: int
    priority: float


class GapAnalysis(BaseModel):
    """
    Complete gap analysis results
    """

    analysis_date: datetime
    gaps: List[InventoryGap]
    recommendations: List[str]
    estimated_impact: Dict
