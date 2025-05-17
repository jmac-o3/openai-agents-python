"""
Tests for the Market Analysis components of DOSO AI
"""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from ..src.agents.market_analysis import MarketAnalysisAgent
from ..src.api.routers.market_analysis import router
from ..src.models.market_analysis import FastTurnData, MarketAnalysis, MarketInsight, MarketTrend


@pytest.fixture
def sample_fast_turn_data():
    """Generate sample Fast Turn data for testing"""
    now = datetime.utcnow()
    return [
        FastTurnData(
            model_code="F150",
            model_year=2024,
            region="West",
            days_supply=45.0,
            turn_rate=0.8,
            market_demand_score=0.85,
            allocation_units=100,
            dealer_orders=90,
            timestamp=now - timedelta(hours=1)
        ),
        FastTurnData(
            model_code="F150",
            model_year=2024,
            region="East",
            days_supply=35.0,
            turn_rate=0.9,
            market_demand_score=0.9,
            allocation_units=120,
            dealer_orders=110,
            timestamp=now
        )
    ]


@pytest.fixture
def sample_market_trend(sample_fast_turn_data):
    """Generate a sample market trend for testing"""
    agent = MarketAnalysisAgent()
    return agent._calculate_segment_trend(sample_fast_turn_data)


def test_market_analysis_agent(sample_fast_turn_data):
    """Test MarketAnalysisAgent's core functionality"""
    agent = MarketAnalysisAgent()
    
    # Test segment trend calculation
    trend = agent._calculate_segment_trend(sample_fast_turn_data)
    assert trend.segment == "Full-Size Pickup"
    assert trend.trend_direction == "up"
    assert trend.strength > 0.7
    assert len(trend.key_drivers) > 0
    assert trend.confidence_score > 0

    # Test demand score calculation
    demand_score = agent._calculate_demand_score(sample_fast_turn_data, trend)
    assert 0 <= demand_score <= 1
    assert demand_score > 0.7  # Should be high given sample data

    # Test allocation recommendation
    alloc = agent._calculate_allocation_recommendation(sample_fast_turn_data, demand_score)
    assert alloc > 0
    assert isinstance(alloc, int)

    # Test risk identification
    risks = agent._identify_risk_factors(sample_fast_turn_data, trend)
    assert isinstance(risks, list)

    # Test opportunity identification
    opps = agent._identify_opportunities(sample_fast_turn_data, trend, demand_score)
    assert isinstance(opps, list)
    assert len(opps) > 0  # Should identify opportunities given strong sample data


@pytest.mark.asyncio
async def test_analyze_fast_turn_data(sample_fast_turn_data):
    """Test full Fast Turn data analysis"""
    agent = MarketAnalysisAgent()
    analysis = await agent.analyze_fast_turn_data(sample_fast_turn_data)
    
    assert isinstance(analysis, MarketAnalysis)
    assert "Full-Size Pickup" in analysis.analyzed_segments
    assert len(analysis.segment_trends) > 0
    assert len(analysis.demand_forecast) > 0
    assert len(analysis.allocation_recommendations) > 0
    assert analysis.confidence_score > 0
    assert analysis.data_freshness_score > 0


@pytest.mark.asyncio
async def test_generate_market_insight(sample_fast_turn_data, sample_market_trend):
    """Test market insight generation"""
    agent = MarketAnalysisAgent()
    insight = await agent.generate_market_insight(sample_market_trend, sample_fast_turn_data)
    
    assert isinstance(insight, MarketInsight)
    assert insight.segment == "Full-Size Pickup"
    assert insight.impact_score > 0
    assert len(insight.action_items) > 0
    assert insight.confidence_score > 0


def test_api_analyze_fast_turn(sample_fast_turn_data):
    """Test the Fast Turn analysis API endpoint"""
    client = TestClient(router)
    response = client.post("/analyze-fast-turn", json=[d.dict() for d in sample_fast_turn_data])
    
    assert response.status_code == 200
    data = response.json()
    assert "analyzed_segments" in data
    assert "segment_trends" in data
    assert "demand_forecast" in data
    assert "allocation_recommendations" in data


def test_api_market_insight(sample_fast_turn_data):
    """Test the market insight API endpoint"""
    client = TestClient(router)
    response = client.post(
        "/market-insight/Full-Size Pickup",
        json=[d.dict() for d in sample_fast_turn_data]
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["segment"] == "Full-Size Pickup"
    assert data["impact_score"] > 0
    assert len(data["action_items"]) > 0


def test_api_market_trends():
    """Test the market trends API endpoint"""
    client = TestClient(router)
    response = client.get("/market-trends")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert len(data) > 0
    assert all(isinstance(trend["segment"], str) for trend in data.values())
