"""
RecommendationAgent - Provides optimized vehicle configuration recommendations

This agent uses learned weights from the LearningAgent to generate vehicle
configuration recommendations that optimize for profit, turnover, forecast
accuracy, and market factors.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from agents import Agent, RunContextWrapper, function_tool

# Ensure the data directory exists
DATA_DIR = Path("doso-ai/data")
DOSO_CONFIG = DATA_DIR / "doso_config.json"
RECOMMENDATIONS_LOG = DATA_DIR / "recommendations_log.jsonl"

os.makedirs(DATA_DIR, exist_ok=True)


class DosoWeights(BaseModel):
    """Weights configuration for DOSO scoring algorithm"""
    profit_weight: float = 0.25
    ddt_weight: float = 0.25
    market_weight: float = 0.25
    forecast_weight: float = 0.25
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    model_type: str = "default"


class RecommendationRequest(BaseModel):
    """Request parameters for generating a recommendation"""
    config_id: str
    model: Optional[str] = "F-150"
    trim: Optional[str] = None
    options: Optional[List[str]] = None
    profit_score: Optional[float] = None
    ddt_score: Optional[float] = None
    market_score: Optional[float] = None
    current_inventory: Optional[int] = None


class RecommendationResult(BaseModel):
    """Result of a recommendation calculation"""
    config_id: str
    recommended_qty: int
    confidence: float
    model: Optional[str] = None
    trim: Optional[str] = None
    options: Optional[List[str]] = None
    profit_score: Optional[float] = None
    ddt_score: Optional[float] = None
    market_score: Optional[float] = None
    forecast_score: Optional[float] = None
    recommendation_id: str = Field(default_factory=lambda: f"rec_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    weights_used: DosoWeights = Field(default_factory=DosoWeights)
    reasoning: Optional[str] = None


def load_weights() -> DosoWeights:
    """Load weights from the configuration file"""
    if not DOSO_CONFIG.exists():
        # Return default weights if the file doesn't exist
        return DosoWeights()
    
    try:
        with open(DOSO_CONFIG, "r") as f:
            config_data = json.load(f)
            weights_data = config_data.get("weights", {})
            
            return DosoWeights(
                profit_weight=weights_data.get("profit_weight", 0.25),
                ddt_weight=weights_data.get("ddt_weight", 0.25),
                market_weight=weights_data.get("market_weight", 0.25),
                forecast_weight=weights_data.get("forecast_weight", 0.25),
                created_at=weights_data.get("created_at", datetime.now().isoformat()),
                model_type=weights_data.get("model_type", "default")
            )
    except Exception as e:
        print(f"Error loading weights: {e}")
        return DosoWeights()


def save_recommendation(recommendation: RecommendationResult) -> bool:
    """Save recommendation to log file"""
    try:
        with open(RECOMMENDATIONS_LOG, "a") as f:
            f.write(json.dumps(recommendation.model_dump()) + "\n")
        return True
    except Exception as e:
        print(f"Error saving recommendation: {e}")
        return False


@function_tool
async def get_recommendation(ctx: RunContextWrapper[Any], config_id: str, current_inventory: int = 0) -> Dict[str, Any]:
    """
    Generate an optimized quantity recommendation for a vehicle configuration.
    
    Args:
        config_id: The configuration ID to recommend for
        current_inventory: Current on-hand inventory (default: 0)
    """
    try:
        # Load the learned weights
        weights = load_weights()
        
        # We would normally fetch these scores from databases or APIs
        # For now, we'll use simulated values
        profit_score = 0.8  # High profit vehicle
        ddt_score = 0.7    # Reasonably fast turnover
        market_score = 0.6  # Decent market demand
        forecast_score = 0.9  # Strong forecast
        
        # Calculate weighted score (0-1)
        weighted_score = (
            weights.profit_weight * profit_score +
            weights.ddt_weight * ddt_score +
            weights.market_weight * market_score +
            weights.forecast_weight * forecast_score
        )
        
        # Convert to recommended quantity (1-20 scale)
        raw_quantity = max(1, min(20, round(weighted_score * 20)))
        
        # Adjust for current inventory
        recommended_qty = max(0, raw_quantity - current_inventory)
        
        # Calculate confidence level based on score consistency
        scores = [profit_score, ddt_score, market_score, forecast_score]
        score_variance = sum((s - weighted_score) ** 2 for s in scores) / len(scores)
        confidence = max(0.5, min(1.0, 1.0 - score_variance))
        
        # Create recommendation result
        recommendation = RecommendationResult(
            config_id=config_id,
            recommended_qty=recommended_qty,
            confidence=confidence,
            profit_score=profit_score,
            ddt_score=ddt_score,
            market_score=market_score,
            forecast_score=forecast_score,
            weights_used=weights,
            reasoning=f"Recommendation based on weighted score of {weighted_score:.2f} "
                    f"(profit: {weights.profit_weight:.2f}*{profit_score:.2f}, "
                    f"ddt: {weights.ddt_weight:.2f}*{ddt_score:.2f}, "
                    f"market: {weights.market_weight:.2f}*{market_score:.2f}, "
                    f"forecast: {weights.forecast_weight:.2f}*{forecast_score:.2f})"
        )
        
        # Save recommendation to log
        save_recommendation(recommendation)
        
        return {
            "status": "success",
            "recommendation": recommendation.model_dump(),
            "weights_used": weights.model_dump()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error generating recommendation: {str(e)}"
        }


@function_tool
async def batch_recommend(ctx: RunContextWrapper[Any], config_ids: List[str]) -> Dict[str, Any]:
    """
    Generate recommendations for multiple configurations in batch.
    
    Args:
        config_ids: List of configuration IDs to recommend for
    """
    try:
        results = []
        for config_id in config_ids:
            result = await get_recommendation(ctx, config_id)
            if result.get("status") == "success":
                results.append(result.get("recommendation"))
        
        return {
            "status": "success",
            "count": len(results),
            "recommendations": results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error in batch recommendations: {str(e)}"
        }


@function_tool
async def get_current_weights(ctx: RunContextWrapper[Any]) -> Dict[str, Any]:
    """
    Get the current recommendation weights being used.
    """
    try:
        weights = load_weights()
        
        return {
            "status": "success",
            "weights": weights.model_dump()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving weights: {str(e)}"
        }


# Define the recommendation agent
recommendation_agent = Agent(
    name="Recommendation Agent",
    description="Provides optimized vehicle configuration recommendations",
    instructions="""
    You are an agent that provides optimized vehicle configuration recommendations.
    Your responsibilities include:
    
    1. Using learned weights to calculate optimal inventory recommendations
    2. Considering profit, days-to-turn, market factors, and forecasts
    3. Providing recommendations with confidence scores and explanations
    
    When generating recommendations, explain your reasoning and include the weights used.
    """,
    tools=[
        get_recommendation,
        batch_recommend,
        get_current_weights
    ],
    model="gpt-4o",
)
