from datetime import datetime, timedelta

import pytest
from doso_ai.src.agents.inventory_analysis import InventoryAnalysisAgent
from doso_ai.src.models.inventory import InventoryAnalysis, InventoryMetrics


@pytest.fixture
def inventory_agent():
    return InventoryAnalysisAgent()


@pytest.fixture
def sample_inventory():
    return {
        "total_units": 100,
        "total_value": 500000.0,
        "categories": {"SUV": 40, "Sedan": 35, "Truck": 25},
    }


@pytest.fixture
def sample_historical_data():
    today = datetime.now()
    return [
        {"date": today - timedelta(days=i), "total_units": 100 - i, "sales": 2} for i in range(90)
    ]


@pytest.fixture
def sample_market_data():
    return {"market_growth": 0.05, "competitor_inventory": {"SUV": 45, "Sedan": 30, "Truck": 25}}


async def test_inventory_metrics_calculation(
    inventory_agent,
    sample_inventory,
    sample_historical_data,
):
    metrics = await inventory_agent.analyze_inventory_metrics(
        sample_inventory,
        sample_historical_data,
    )

    assert isinstance(metrics, InventoryMetrics)
    assert metrics.total_units == sample_inventory["total_units"]
    assert metrics.total_value == sample_inventory["total_value"]
    assert metrics.average_days_supply >= 0
    assert metrics.turnover_rate >= 0
    assert all(v >= 0 for v in metrics.aging_distribution.values())


async def test_inventory_analysis_generation(
    inventory_agent,
    sample_inventory,
    sample_historical_data,
    sample_market_data,
):
    metrics = await inventory_agent.analyze_inventory_metrics(
        sample_inventory,
        sample_historical_data,
    )

    analysis = await inventory_agent.generate_inventory_analysis(metrics, sample_market_data)

    assert isinstance(analysis, InventoryAnalysis)
    assert analysis.metrics == metrics
    assert isinstance(analysis.insights, list)
    assert isinstance(analysis.risk_factors, list)
    assert isinstance(analysis.opportunities, list)
