from datetime import datetime

import pytest
from doso_ai.src.agents.gap_analysis import GapAnalysisAgent
from doso_ai.src.models.gap_analysis import GapAnalysis


@pytest.fixture
def gap_agent():
    return GapAnalysisAgent()


@pytest.fixture
def sample_current_inventory():
    return {"SUV": 40, "Sedan": 35, "Truck": 25}


@pytest.fixture
def sample_target_mix():
    return {
        "SUV": {"min_units": 50, "importance": 0.8},
        "Sedan": {"min_units": 40, "importance": 0.6},
        "Truck": {"min_units": 30, "importance": 0.7},
    }


@pytest.fixture
def sample_market_data():
    return {
        "market_share": {"SUV": 0.45, "Sedan": 0.35, "Truck": 0.20},
        "competitor_inventory": {"SUV": 55, "Sedan": 40, "Truck": 35},
    }


async def test_gap_analysis(
    gap_agent,
    sample_current_inventory,
    sample_target_mix,
    sample_market_data,
):
    analysis = await gap_agent.analyze_inventory_gaps(
        sample_current_inventory,
        sample_target_mix,
        sample_market_data,
    )

    assert isinstance(analysis, GapAnalysis)
    assert isinstance(analysis.analysis_date, datetime)
    assert len(analysis.gaps) > 0

    # Check that gaps are properly identified
    suv_gap = next(g for g in analysis.gaps if g.category == "SUV")
    assert suv_gap.current_value == 40
    assert suv_gap.target_value == 50

    # Verify recommendations
    assert len(analysis.recommendations) > 0
    assert any("SUV" in r for r in analysis.recommendations)

    # Check impact estimation
    assert "revenue_impact" in analysis.estimated_impact
    assert "priority_score" in analysis.estimated_impact
    assert analysis.estimated_impact["priority_score"] > 0


async def test_gap_prioritization(
    gap_agent,
    sample_current_inventory,
    sample_target_mix,
    sample_market_data,
):
    analysis = await gap_agent.analyze_inventory_gaps(
        sample_current_inventory,
        sample_target_mix,
        sample_market_data,
    )

    # Verify gaps are properly prioritized
    gaps = sorted(analysis.gaps, key=lambda x: x.priority, reverse=True)
    assert len(gaps) > 1
    assert gaps[0].priority >= gaps[1].priority

    # Check that high importance categories have higher priority
    suv_gap = next(g for g in gaps if g.category == "SUV")
    sedan_gap = next(g for g in gaps if g.category == "Sedan")
    assert suv_gap.priority > sedan_gap.priority
