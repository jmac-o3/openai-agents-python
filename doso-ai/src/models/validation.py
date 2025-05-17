from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel

class SeasonalRequirements(BaseModel):
    """Seasonal model mix requirements"""
    model_type: str
    required_percentage: float
    applicable_months: List[int]

class OEMPolicyConstraints(BaseModel):
    """OEM policy and allocation constraints"""
    model_allocations: Dict[str, int]
    min_days_supply: int
    max_days_supply: int
    seasonal_requirements: Dict[int, Dict[str, float]]  # month -> {model_type: required_pct}
    order_bank_limit: int
    scheduling_horizon_days: int

class InventoryConstraints(BaseModel):
    """Defines constraints for inventory management"""
    min_units: int
    max_units: int
    max_inventory_value: float
    space_per_unit: float
    available_space: float
    seasonal_adjustment_factor: Optional[float] = 1.0
    oem_allocation_limit: Optional[int] = None
    oem_policies: Optional[OEMPolicyConstraints] = None

class ValidationResult(BaseModel):
    """Result of inventory constraint validation"""
    is_valid: bool
    violations: List[str]
    warnings: Optional[List[str]] = None
    last_checked: datetime = datetime.now()
