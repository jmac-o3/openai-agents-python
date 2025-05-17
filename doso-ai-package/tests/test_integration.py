import pytest
from datetime import datetime, timedelta

from doso_ai.src.agents.order_bank_agent import (
    OrderBankAnalysisRequest,
    OrderBankAnalysisResult,
    OrderMetrics,
    OptimizationRecommendation
)
from doso_ai.src.agents.sva_agent import (
    SVAAgent,
    SVACalculationRequest,
    SVAMetrics
)
from doso_ai.src.agents.constraint_check import (
    ConstraintCheckAgent,
    InventoryConstraints,
    ValidationResult
)
from doso_ai.src.agents.guidance_agent import (
    GuidanceAgent,
    ExplanationRequest,
    GuidanceResponse,
    AlertConfig,
    AlertNotification
)

@pytest.fixture
def sample_order_data():
    return {
        "dealer_id": "TEST001",
        "orders": [
            {
                "config_id": "F150-XLT-401A",
                "quantity": 5,
                "order_date": datetime.now() - timedelta(days=30)
            },
            {
                "config_id": "MACHE-GT-400",
                "quantity": 3,
                "order_date": datetime.now() - timedelta(days=15)
            }
        ]
    }

@pytest.fixture
def sample_inventory_constraints():
    return InventoryConstraints(
        min_units=10,
        max_units=100,
        max_inventory_value=5000000.0,
        space_per_unit=200.0,
        available_space=25000.0
    )

@pytest.mark.asyncio
async def test_end_to_end_analysis_flow(
    sample_order_data,
    sample_inventory_constraints
):
    """Test the complete analysis workflow with all agents"""
    
    # 1. Order Bank Analysis
    order_request = OrderBankAnalysisRequest(
        dealer_id=sample_order_data["dealer_id"],
        date_range_start=datetime.now() - timedelta(days=90),
        date_range_end=datetime.now(),
        include_historical=True
    )
    
    # 2. SVA Analysis
    sva_agent = SVAAgent()
    sva_request = SVACalculationRequest(
        dealer_id=sample_order_data["dealer_id"],
        inventory_data={"F150-XLT-401A": 8, "MACHE-GT-400": 4},
        sales_data={"F150-XLT-401A": 6, "MACHE-GT-400": 3}
    )
    sva_result = await sva_agent.calculate_sva(sva_request)
    assert isinstance(sva_result.metrics, SVAMetrics)
    assert sva_result.metrics.total_sva > 0

    # 3. Constraint Check
    constraint_agent = ConstraintCheckAgent()
    validation_result = await constraint_agent.validate_constraints(
        inventory_data={
            "total_units": 12,
            "total_value": 1000000.0
        },
        constraints=sample_inventory_constraints
    )
    assert isinstance(validation_result, ValidationResult)
    assert validation_result.is_valid

    # 4. Guidance Generation
    guidance_agent = GuidanceAgent()
    explanation_request = ExplanationRequest(
        topic="order_bank_optimization",
        context={
            "sva_metrics": sva_result.metrics.dict(),
            "validation": validation_result.dict()
        }
    )
    guidance_result = await guidance_agent.explain_recommendation(
        topic=explanation_request.topic,
        context=explanation_request.context
    )
    assert isinstance(guidance_result, GuidanceResponse)
    assert len(guidance_result.action_items) > 0

    # Test alert generation
    alert_config = AlertConfig(
        severity="warning",
        threshold=90.0,
        message_template="SVA for {metric} exceeds {threshold}%"
    )
    alert = await guidance_agent.generate_alert(
        metric_name="total_sva",
        current_value=95.0,
        alert_config=alert_config
    )
    assert isinstance(alert, AlertNotification)
    assert alert.severity == "warning"

@pytest.mark.asyncio
async def test_constraint_validation_flow(sample_inventory_constraints):
    """Test the constraint validation workflow"""
    
    agent = ConstraintCheckAgent()
    
    # Test valid case
    valid_result = await agent.validate_constraints(
        inventory_data={
            "total_units": 50,
            "total_value": 2500000.0
        },
        constraints=sample_inventory_constraints
    )
    assert valid_result.is_valid
    assert len(valid_result.violations) == 0

    # Test invalid case
    invalid_result = await agent.validate_constraints(
        inventory_data={
            "total_units": 150,  # Exceeds max_units
            "total_value": 6000000.0  # Exceeds max_inventory_value
        },
        constraints=sample_inventory_constraints
    )
    assert not invalid_result.is_valid
    assert len(invalid_result.violations) > 0

@pytest.mark.asyncio
async def test_guidance_explanation_levels():
    """Test different detail levels of guidance explanations"""
    
    agent = GuidanceAgent()
    context = {"metric": "sva", "value": 85.0}

    # Test executive level
    exec_response = await agent.explain_recommendation(
        topic="sva_performance",
        context=context,
        detail_level="executive"
    )
    assert isinstance(exec_response, GuidanceResponse)
    assert len(exec_response.action_items) > 0

    # Test detailed level
    detailed_response = await agent.explain_recommendation(
        topic="sva_performance",
        context=context,
        detail_level="detailed"
    )
    assert isinstance(detailed_response, GuidanceResponse)
    assert detailed_response.references is not None

    # Test standard level
    standard_response = await agent.explain_recommendation(
        topic="sva_performance",
        context=context,
        detail_level="standard"
    )
    assert isinstance(standard_response, GuidanceResponse)
    assert standard_response.next_steps is not None
