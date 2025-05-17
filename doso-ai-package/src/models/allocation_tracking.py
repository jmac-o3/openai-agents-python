from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from pydantic import BaseModel


class AllocationStatus(str, Enum):
    """Status of an allocation"""
    
    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    MODIFIED = "modified"
    EXPIRED = "expired"


class AllocationLineItem(BaseModel):
    """Individual line item in an allocation"""
    
    model_code: str
    model_year: int
    trim_level: str
    quantity: int
    status: AllocationStatus
    accept_deadline: datetime
    constraints: Optional[Dict[str, Any]] = None


class DealerAllocation(BaseModel):
    """Complete allocation package for a dealer"""
    
    allocation_id: str
    dealer_id: str
    allocation_date: datetime
    effective_date: datetime
    expiry_date: datetime
    line_items: List[AllocationLineItem]
    total_units: int
    status_summary: Dict[str, int]  # Count of each status
    notes: Optional[str] = None


class AllocationHistory(BaseModel):
    """Historical record of allocations"""
    
    dealer_id: str
    time_period: str  # e.g., "Q1-2023", "Jan-2023"
    total_allocated: int
    total_accepted: int
    total_declined: int
    allocation_by_model: Dict[str, int]
    performance_metrics: Dict[str, float]
    

class AllocationTracking(BaseModel):
    """Comprehensive allocation tracking data"""
    
    current_allocation: Optional[DealerAllocation] = None
    historical_allocations: List[AllocationHistory]
    allocation_trends: Dict[str, Any]
    upcoming_allocation_estimate: Optional[Dict[str, Any]] = None
    acceptance_rate: float
    modification_rate: float
    expiration_rate: float
    last_updated: datetime


class AllocationPerformance(BaseModel):
    """Performance metrics for allocations"""
    
    dealer_id: str
    time_period: str
    allocation_effectiveness: float  # 0.0 to 1.0 score
    missed_opportunities: List[Dict[str, Any]]
    strategic_insights: List[str]
    benchmark_comparison: Dict[str, float]
    improvement_areas: List[str]
    last_updated: datetime
