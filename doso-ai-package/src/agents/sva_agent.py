from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel

from agents import Agent, function_tool

class SVAMetrics(BaseModel):
    """SVA (Stock vs Actual) performance metrics"""
    total_sva: float
    model_sva: Dict[str, float]
    dealer_rank: int
    regional_avg: float
    trend: List[float]
    ws2_impact: Optional[float] = None

class SVACalculationRequest(BaseModel):
    """Input for SVA calculation"""
    dealer_id: str
    inventory_data: Dict[str, int]  # Model code -> quantity
    sales_data: Dict[str, int]  # Model code -> last month sales
    regional_data: Optional[Dict[str, float]] = None  # Dealer ID -> SVA score
    calculation_date: datetime

class SVAAnalysisResult(BaseModel):
    """Complete SVA analysis results"""
    metrics: SVAMetrics
    recommendations: List[str]
    impact_analysis: str

class SVAAgent(Agent):
    """Agent for calculating and analyzing Stock vs Actual (SVA) metrics"""
    
    def __init__(self):
        super().__init__(
            name="SVA Analysis Agent",
            instructions="""
            You are an expert in analyzing Stock vs Actual (SVA) metrics for automotive dealerships.
            Your responsibilities include:
            1. Calculate accurate SVA scores following the formula: SVA = (Last Month Sales ÷ Month-End Stock) × 100
            2. Analyze SVA performance by model
            3. Compare dealer performance against regional averages
            4. Project WS2 (Wholesale to Salesperson) impact
            5. Provide actionable recommendations based on SVA analysis
            6. Identify trends and patterns in SVA performance
            """,
        )

    @function_tool
    async def calculate_sva(self, request: SVACalculationRequest) -> SVAAnalysisResult:
        """
        Calculate SVA metrics and provide analysis
        
        Args:
            request: SVA calculation request with inventory and sales data
            
        Returns:
            Comprehensive SVA analysis including metrics and recommendations
        """
        # Calculate total SVA
        total_stock = sum(request.inventory_data.values())
        total_sales = sum(request.sales_data.values())
        total_sva = (total_sales / total_stock * 100) if total_stock > 0 else 0

        # Calculate model-specific SVA
        model_sva = {}
        for model, stock in request.inventory_data.items():
            sales = request.sales_data.get(model, 0)
            sva = (sales / stock * 100) if stock > 0 else 0
            model_sva[model] = sva

        # Calculate dealer rank if regional data available
        dealer_rank = 1
        if request.regional_data:
            dealer_score = total_sva
            better_dealers = sum(1 for score in request.regional_data.values() if score > dealer_score)
            dealer_rank = better_dealers + 1
            regional_avg = sum(request.regional_data.values()) / len(request.regional_data)
        else:
            regional_avg = 0

        # Generate metrics
        metrics = SVAMetrics(
            total_sva=total_sva,
            model_sva=model_sva,
            dealer_rank=dealer_rank,
            regional_avg=regional_avg,
            trend=[],  # Would be populated with historical data
            ws2_impact=self._calculate_ws2_impact(total_sva, model_sva)
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, request)

        return SVAAnalysisResult(
            metrics=metrics,
            recommendations=recommendations,
            impact_analysis=self._generate_impact_analysis(metrics)
        )

    def _calculate_ws2_impact(self, total_sva: float, model_sva: Dict[str, float]) -> float:
        """Calculate the impact on Wholesale to Salesperson (WS2) scores"""
        # Implement WS2 impact calculation logic
        # This would typically factor in OEM-specific formulas
        return total_sva * 0.15  # Simplified example

    def _generate_recommendations(
        self,
        metrics: SVAMetrics,
        request: SVACalculationRequest
    ) -> List[str]:
        """Generate actionable recommendations based on SVA analysis"""
        recommendations = []
        
        # Check overall SVA performance
        if metrics.total_sva < 100:
            recommendations.append(
                "Overall SVA below target. Consider reducing stock levels or implementing "
                "sales acceleration strategies."
            )

        # Check model-specific performance
        for model, sva in metrics.model_sva.items():
            if sva < 80:
                recommendations.append(
                    f"Model {model} showing low SVA ({sva:.1f}). Review stocking strategy "
                    "and consider reallocation."
                )
            elif sva > 120:
                recommendations.append(
                    f"Model {model} showing high SVA ({sva:.1f}). Potential opportunity "
                    "to increase inventory if allocation available."
                )

        return recommendations

    def _generate_impact_analysis(self, metrics: SVAMetrics) -> str:
        """Generate detailed impact analysis based on SVA metrics"""
        regional_comparison = (
            f"Performance is {metrics.regional_avg - metrics.total_sva:+.1f} points "
            f"versus regional average"
        )
        
        ws2_impact = (
            f"Current SVA performance projects to {metrics.ws2_impact:+.1f} point "
            "impact on WS2 score"
        )
        
        return f"{regional_comparison}. {ws2_impact}."
