"""
Common request/response schemas
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field


class MarketConditions(BaseModel):
    """Market conditions input schema"""

    economic_indicators: Dict[str, float]
    regional_factors: Dict[str, Dict[str, float]]
    seasonal_factors: Dict[str, float]
    competitor_actions: List[Dict[str, str]]
    timestamp: datetime = datetime.now()


class PaginationParams(BaseModel):
    """Common pagination parameters"""

    page: int = 1
    page_size: int = 50

    @property
    def skip(self) -> int:
        return (self.page - 1) * self.page_size


class ErrorResponse(BaseModel):
    """Standard error response"""

    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = datetime.now()


class WorkflowType(str, Enum):
    """Types of processing workflows"""
    
    INVENTORY_OPTIMIZATION = "inventory_optimization"
    ORDER_PLANNING = "order_planning"
    MARKET_ANALYSIS = "market_analysis"
    ALLOCATION_TRACKING = "allocation_tracking"
    GAP_ANALYSIS = "gap_analysis"
    CONSTRAINT_CHECK = "constraint_check"
    DATA_COLLECTION = "data_collection"
    GENERAL_ANALYSIS = "general_analysis"
    CUSTOM_ANALYSIS = "custom_analysis"
    EXCEPTION_HANDLING = "exception_handling"


class DealerRequest(BaseModel):
    """Dealer request model"""
    
    request_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))
    dealer_id: str
    request_type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    priority: int = 0


class TriageResponse(BaseModel):
    """Response from triage agent"""
    
    is_valid: bool
    workflow_type: WorkflowType
    agent_sequence: List[str]
    missing_fields: List[str] = Field(default_factory=list)
    additional_info: str = ""
    needs_human_review: bool = False


class OrchestrationResult(BaseModel):
    """Result from workflow orchestration"""
    
    request_id: str
    dealer_id: str
    workflow_type: WorkflowType
    is_complete: bool
    results: Dict[str, Any]
    recommendations: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    execution_time: float = 0.0


class AgentExecutionMetrics(BaseModel):
    """Metrics for agent execution"""
    
    agent_name: str
    operation: str
    start_time: datetime
    end_time: datetime
    execution_time: float
    token_usage: Optional[Dict[str, int]] = None
    status: str = "success"
    error_message: Optional[str] = None


class WorkflowExecutionSummary(BaseModel):
    """Summary of workflow execution"""
    
    workflow_id: str
    workflow_type: WorkflowType
    request_id: str
    dealer_id: str
    start_time: datetime
    end_time: datetime
    total_execution_time: float
    agent_metrics: List[AgentExecutionMetrics]
    total_token_usage: Dict[str, int]
    status: str
    error: Optional[str] = None
