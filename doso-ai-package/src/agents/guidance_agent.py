from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel

from agents import Agent, function_tool

class ExplanationRequest(BaseModel):
    """Request for explanation of recommendations or metrics"""
    topic: str
    context: Dict[str, any]
    detail_level: str = "standard"  # standard, detailed, executive
    include_historical: bool = False

class AlertConfig(BaseModel):
    """Configuration for system alerts"""
    severity: str  # info, warning, critical
    threshold: float
    message_template: str

class GuidanceResponse(BaseModel):
    """Structured response with explanations and guidance"""
    explanation: str
    action_items: List[str]
    references: Optional[List[str]] = None
    next_steps: Optional[List[str]] = None

class AlertNotification(BaseModel):
    """Structured alert notification"""
    severity: str
    message: str
    timestamp: datetime
    action_required: bool
    suggested_actions: List[str]

class GuidanceAgent(Agent):
    """
    Specialized agent for providing explanations, generating alerts,
    and guiding users through the system's recommendations.
    """
    
    def __init__(self):
        super().__init__(
            name="Guidance Agent",
            instructions="""
            You are an expert guide for the DOSO AI system, responsible for:
            1. Explaining complex metrics and recommendations in clear terms
            2. Providing actionable insights from system outputs
            3. Generating timely alerts based on configured thresholds
            4. Guiding users through implementing recommendations
            5. Offering historical context and trend analysis
            6. Suggesting next steps based on current state
            
            Always provide explanations that are:
            - Clear and concise
            - Actionable and practical
            - Backed by data
            - Appropriate for the user's context
            """,
        )

    @function_tool
    async def explain_recommendation(
        self,
        topic: str,
        context: Dict[str, any],
        detail_level: str = "standard"
    ) -> GuidanceResponse:
        """Generate detailed explanation for a specific recommendation or metric"""
        
        if detail_level == "executive":
            return self._generate_executive_response(topic, context)
        elif detail_level == "detailed":
            return self._generate_detailed_response(topic, context)
        else:
            return self._generate_standard_response(topic, context)

    @function_tool
    async def generate_alert(
        self,
        metric_name: str,
        current_value: float,
        alert_config: AlertConfig
    ) -> Optional[AlertNotification]:
        """Generate alert notification if threshold is exceeded"""
        
        if current_value > alert_config.threshold:
            return AlertNotification(
                severity=alert_config.severity,
                message=alert_config.message_template.format(
                    metric=metric_name,
                    value=current_value,
                    threshold=alert_config.threshold
                ),
                timestamp=datetime.now(),
                action_required=True,
                suggested_actions=self._generate_alert_actions(
                    metric_name,
                    current_value,
                    alert_config
                )
            )
        return None

    def _generate_executive_response(
        self,
        topic: str,
        context: Dict[str, any]
    ) -> GuidanceResponse:
        """Generate executive-level response with high-level insights"""
        return GuidanceResponse(
            explanation=f"Executive summary for {topic}...",
            action_items=[
                "Review key metrics dashboard",
                "Approve recommended actions",
                "Schedule follow-up review"
            ],
            next_steps=[
                "Share insights with leadership team",
                "Monitor implementation progress",
                "Schedule monthly review"
            ]
        )

    def _generate_detailed_response(
        self,
        topic: str,
        context: Dict[str, any]
    ) -> GuidanceResponse:
        """Generate detailed technical response with comprehensive analysis"""
        return GuidanceResponse(
            explanation=f"Detailed analysis for {topic}...",
            action_items=[
                "Review complete metrics breakdown",
                "Implement specific recommendations",
                "Document implementation steps"
            ],
            references=[
                "Historical performance data",
                "Industry benchmarks",
                "OEM guidelines"
            ],
            next_steps=[
                "Schedule technical review",
                "Update tracking systems",
                "Prepare progress report"
            ]
        )

    def _generate_standard_response(
        self,
        topic: str,
        context: Dict[str, any]
    ) -> GuidanceResponse:
        """Generate standard response with balanced detail level"""
        return GuidanceResponse(
            explanation=f"Standard guidance for {topic}...",
            action_items=[
                "Review recommended actions",
                "Implement priority items",
                "Monitor results"
            ],
            next_steps=[
                "Schedule follow-up check",
                "Update action items",
                "Document outcomes"
            ]
        )

    def _generate_alert_actions(
        self,
        metric_name: str,
        current_value: float,
        alert_config: AlertConfig
    ) -> List[str]:
        """Generate suggested actions based on alert type and severity"""
        if alert_config.severity == "critical":
            return [
                "Immediate review required",
                "Schedule emergency planning session",
                "Prepare mitigation strategy"
            ]
        elif alert_config.severity == "warning":
            return [
                "Review within 24 hours",
                "Prepare action plan",
                "Monitor closely"
            ]
        else:
            return [
                "Review during next session",
                "Update monitoring parameters",
                "Document in system"
            ]
