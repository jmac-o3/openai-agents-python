from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai_agents import Agent, function_tool
from openai_agents.exceptions import AgentsException
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from ..models.market_analysis import FastTurnData, MarketAnalysis, MarketInsight, MarketTrend
from ..utils.validation import validate_data_freshness

# Configure logging
logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class MarketAnalysisAgent(Agent):
    """
    Market Analysis Agent for DOSO AI system.

    Responsible for analyzing market trends, Fast Turn reports, and market data
    to provide actionable insights for inventory optimization.

    Capabilities:
    - Fast Turn report analysis
    - Market trend detection
    - Regional performance comparison
    - Configuration-specific analysis
    - Real-time market monitoring

    Error Handling:
    - Validates input data freshness
    - Implements retry mechanisms for API calls
    - Provides detailed error messages
    - Logs errors for monitoring

    Performance:
    - OpenTelemetry integration for tracing
    - Performance monitoring hooks
    - Caching for frequently accessed data
    """

    def __init__(self):
        super().__init__(
            name="Market Analysis Agent",
            instructions="""You are an expert market analysis agent for Ford's DOSO AI system responsible for:
            1. Analyzing Fast Turn reports and market data to identify trends
            2. Calculating demand forecasts for different vehicle segments
            3. Identifying market opportunities and risks
            4. Making allocation recommendations based on market analysis
            5. Monitoring competitive landscape and market dynamics
            
            Use data from Fast Turn reports, market research, and historical trends to generate
            actionable insights. Focus on identifying both immediate opportunities and potential
            risks that could impact inventory optimization.
            """,
        )
        self.retry_attempts = 3
        self.cache = {}

    @function_tool
    async def analyze_fast_turn_data(self, data: List[FastTurnData]) -> MarketAnalysis:
        """Analyze Fast Turn report data to identify market trends and insights"""
        with tracer.start_as_current_span("analyze_fast_turn_data") as span:
            try:
                # Validate data freshness
                if not validate_data_freshness([d.timestamp for d in data], max_age_hours=24):
                    raise AgentsException("Fast Turn data is too old for accurate analysis")

                # Group data by segment
                segment_data = {}
                for item in data:
                    segment = self._get_segment_from_model(item.model_code)
                    if segment not in segment_data:
                        segment_data[segment] = []
                    segment_data[segment].append(item)

                # Analyze trends by segment
                segment_trends = {}
                demand_forecast = {}
                allocation_recommendations = {}
                risk_factors = []
                opportunities = []

                for segment, items in segment_data.items():
                    # Calculate segment trend
                    trend = self._calculate_segment_trend(items)
                    segment_trends[segment] = trend

                    # Generate demand forecast
                    demand_score = self._calculate_demand_score(items, trend)
                    demand_forecast[segment] = demand_score

                    # Calculate allocation recommendations
                    alloc = self._calculate_allocation_recommendation(items, demand_score)
                    allocation_recommendations[segment] = alloc

                    # Identify risks and opportunities
                    segment_risks = self._identify_risk_factors(items, trend)
                    risk_factors.extend(segment_risks)

                    segment_opps = self._identify_opportunities(items, trend, demand_score)
                    opportunities.extend(segment_opps)

                return MarketAnalysis(
                    analyzed_segments=list(segment_data.keys()),
                    segment_trends=segment_trends,
                    demand_forecast=demand_forecast,
                    allocation_recommendations=allocation_recommendations,
                    risk_factors=risk_factors,
                    opportunities=opportunities,
                    timestamp=datetime.utcnow(),
                    confidence_score=self._calculate_confidence_score(data),
                    data_freshness_score=self._calculate_freshness_score(data)
                )

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR))
                logger.error(f"Error analyzing Fast Turn data: {str(e)}")
                raise

    @function_tool
    async def generate_market_insight(
        self, 
        trend: MarketTrend,
        fast_turn_data: List[FastTurnData]
    ) -> MarketInsight:
        """Generate a specific market insight from trend and Fast Turn data"""
        with tracer.start_as_current_span("generate_market_insight") as span:
            try:
                # Calculate impact metrics
                impact_metrics = self._calculate_impact_metrics(trend, fast_turn_data)
                impact_score = self._calculate_impact_score(impact_metrics)

                # Determine insight type and priority
                insight_type = self._determine_insight_type(trend, impact_score)
                priority = self._calculate_priority(impact_score, trend.confidence_score)

                # Generate action items
                action_items = self._generate_action_items(trend, impact_metrics)

                return MarketInsight(
                    insight_type=insight_type,
                    segment=trend.segment,
                    description=self._generate_insight_description(trend, impact_metrics),
                    impact_score=impact_score,
                    supporting_metrics=impact_metrics,
                    action_items=action_items,
                    priority=priority,
                    timestamp=datetime.utcnow(),
                    expiration=self._calculate_insight_expiration(trend),
                    confidence_score=trend.confidence_score,
                    source_data={"trend": trend.dict(), "metrics": impact_metrics}
                )

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR))
                logger.error(f"Error generating market insight: {str(e)}")
                raise

    async def _analyze_segment_trends_with_retry(
        self,
        fast_turn_data: List[FastTurnData],
        historical_context: Optional[Dict],
    ) -> Dict[str, MarketTrend]:
        """
        Analyze segment trends with retry mechanism.

        Args:
            fast_turn_data: List of FastTurnData objects
            historical_context: Optional historical data

        Returns:
            Dict mapping segments to their MarketTrend analysis

        Raises:
            AgentsException: If all retry attempts fail

        """
        for attempt in range(self.retry_attempts):
            try:
                return self._analyze_segment_trends(fast_turn_data, historical_context)
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    logger.error(f"All retry attempts failed in segment trend analysis: {e!s}")
                    raise AgentsException(
                        "Failed to analyze segment trends after multiple attempts",
                    )
                logger.warning(f"Retry attempt {attempt + 1} for segment trend analysis")
                continue

    @function_tool
    async def monitor_real_time_trends(
        self,
        segment: str,
        window_hours: int = 24,
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Monitor real-time market trends for rapid changes in demand or market conditions

        Args:
            segment: Vehicle segment to monitor
            window_hours: Time window for trend analysis in hours
            threshold: Minimum change threshold to trigger an alert

        Returns:
            Dict containing trend alerts and metrics

        """
        current_time = datetime.now()

        # Implementation would include:
        # 1. Real-time data collection
        # 2. Trend calculation over specified window
        # 3. Change detection and alert generation
        # 4. Confidence scoring for detected changes

        return {
            "segment": segment,
            "timestamp": current_time,
            "trend_alerts": [],  # List of detected trend changes
            "metrics": {},  # Current metric values
            "change_confidence": 0.0,
        }

    @function_tool
    async def analyze_regional_performance(
        self,
        fast_turn_data: List[FastTurnData],
        region: str,
        compare_regions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze regional performance metrics compared to other regions

        Args:
            fast_turn_data: List of FastTurnData objects
            region: Primary region to analyze
            compare_regions: Optional list of regions to compare against

        Returns:
            Dict containing regional performance analysis

        """
        # Filter data for target region
        region_data = [d for d in fast_turn_data if d.region == region]

        # Calculate regional metrics
        region_metrics = {
            "avg_turn_rate": sum(d.turn_rate for d in region_data) / len(region_data),
            "total_allocation": sum(d.allocation_units for d in region_data),
            "avg_days_supply": sum(d.days_supply for d in region_data) / len(region_data),
        }

        # Compare with other regions if specified
        regional_comparison = {}
        if compare_regions:
            for comp_region in compare_regions:
                comp_data = [d for d in fast_turn_data if d.region == comp_region]
                if comp_data:
                    regional_comparison[comp_region] = {
                        "avg_turn_rate": sum(d.turn_rate for d in comp_data) / len(comp_data),
                        "total_allocation": sum(d.allocation_units for d in comp_data),
                        "avg_days_supply": sum(d.days_supply for d in comp_data) / len(comp_data),
                    }

        return {
            "region": region,
            "metrics": region_metrics,
            "comparisons": regional_comparison,
            "timestamp": datetime.now(),
        }

    @function_tool
    async def analyze_configuration(
        self,
        model_code: str,
        fast_turn_data: List[FastTurnData],
        historical_data: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Perform detailed analysis of a specific vehicle configuration

        Args:
            model_code: Vehicle model code to analyze
            fast_turn_data: Current Fast Turn report data
            historical_data: Optional historical data for trend analysis

        Returns:
            Dict containing configuration-specific analysis

        """
        # Filter data for specific configuration
        config_data = [d for d in fast_turn_data if d.model_code == model_code]

        if not config_data:
            return {
                "model_code": model_code,
                "error": "No data available for configuration",
                "timestamp": datetime.now(),
            }

        # Calculate configuration metrics
        current_metrics = {
            "avg_turn_rate": sum(d.turn_rate for d in config_data) / len(config_data),
            "total_allocation": sum(d.allocation_units for d in config_data),
            "avg_days_supply": sum(d.days_supply for d in config_data) / len(config_data),
            "market_demand_score": sum(d.market_demand_score for d in config_data)
            / len(config_data),
        }

        # Generate trend analysis if historical data available
        trend_analysis = None
        if historical_data:
            trend_analysis = self._analyze_configuration_trends(model_code, historical_data)

        return {
            "model_code": model_code,
            "current_metrics": current_metrics,
            "trend_analysis": trend_analysis,
            "timestamp": datetime.now(),
            "data_freshness": self._calculate_data_freshness(config_data),
        }

    def _analyze_configuration_trends(
        self,
        model_code: str,
        historical_data: List[Dict],
    ) -> Dict[str, Any]:
        """
        Analyze historical trends for a specific configuration

        Args:
            model_code: Vehicle model code
            historical_data: Historical configuration data

        Returns:
            Dict containing trend analysis results

        """
        # Implementation would include:
        # 1. Time series analysis of key metrics
        # 2. Trend direction and strength calculation
        # 3. Seasonality detection
        # 4. Anomaly detection

        return {
            "trend_direction": "stable",
            "trend_strength": 0.0,
            "seasonal_factors": {},
            "anomalies": [],
        }

    def _calculate_data_freshness(self, fast_turn_data: List[FastTurnData]) -> float:
        """
        Calculate data freshness score based on timestamps.

        Args:
            fast_turn_data: List of FastTurnData objects

        Returns:
            float: Freshness score between 0.0 and 1.0

        """
        if not fast_turn_data:
            return 0.0

        current_time = datetime.now()
        max_age_hours = 24  # Consider data older than 24 hours as completely stale

        timestamps = [d.timestamp for d in fast_turn_data]
        newest_data = max(timestamps)
        age_hours = (current_time - newest_data).total_seconds() / 3600

        # Calculate freshness score (1.0 = completely fresh, 0.0 = completely stale)
        freshness = max(0.0, 1.0 - (age_hours / max_age_hours))
        return round(freshness, 2)

    def _analyze_segment_trends(
        self,
        fast_turn_data: List[FastTurnData],
        historical_context: Optional[Dict],
    ) -> Dict[str, MarketTrend]:
        """Analyze trends for each vehicle segment"""
        trends = {}
        # Group data by segment
        segment_data = {}
        for d in fast_turn_data:
            segment = self._get_segment(d.model_code)
            if segment not in segment_data:
                segment_data[segment] = []
            segment_data[segment].append(d)

        # Calculate trends for each segment
        for segment, data in segment_data.items():
            trends[segment] = self._calculate_segment_trend(
                data,
                historical_context.get(segment) if historical_context else None,
            )

        return trends

    def _calculate_segment_trend(
        self,
        segment_data: List[FastTurnData],
        historical_data: Optional[Dict],
    ) -> MarketTrend:
        """
        Calculate market trend for a specific vehicle segment using current and historical data.

        Args:
            segment_data: Current Fast Turn data for the segment
            historical_data: Optional historical metrics for trend comparison

        Returns:
            MarketTrend containing trend analysis

        """
        if not segment_data:
            raise ValueError("No segment data provided for trend analysis")

        # Sort data by timestamp to analyze progression
        sorted_data = sorted(segment_data, key=lambda x: x.timestamp)

        # Calculate key metrics
        current_demand = sum(d.market_demand_score for d in sorted_data) / len(sorted_data)
        current_turn = sum(d.turn_rate for d in sorted_data) / len(sorted_data)
        current_supply = sum(d.days_supply for d in sorted_data) / len(sorted_data)

        # Determine trend direction
        trend_direction = "stable"
        trend_strength = 0.5
        key_drivers = []

        if historical_data:
            hist_demand = historical_data.get("avg_demand", current_demand)
            hist_turn = historical_data.get("avg_turn_rate", current_turn)
            hist_supply = historical_data.get("avg_supply", current_supply)

            # Calculate demand change
            demand_change = ((current_demand - hist_demand) / hist_demand) * 100
            turn_change = ((current_turn - hist_turn) / hist_turn) * 100
            supply_change = ((current_supply - hist_supply) / hist_supply) * 100

            # Determine trend direction and strength
            if demand_change > 5 and turn_change > 0:
                trend_direction = "up"
                trend_strength = min(1.0, abs(demand_change) / 20)
                key_drivers.append("Increasing market demand")
            elif demand_change < -5 or turn_change < 0:
                trend_direction = "down"
                trend_strength = min(1.0, abs(demand_change) / 20)
                key_drivers.append("Decreasing market demand")

            # Analyze supply situation
            if supply_change > 10:
                key_drivers.append("Rising inventory levels")
                if trend_direction == "down":
                    trend_strength = min(1.0, trend_strength + 0.2)
            elif supply_change < -10:
                key_drivers.append("Decreasing inventory levels")
                if trend_direction == "up":
                    trend_strength = min(1.0, trend_strength + 0.2)

        # Analyze dealer orders vs allocation
        order_ratio = sum(
            1 for d in sorted_data if d.dealer_orders and d.dealer_orders > d.allocation_units
        )
        if order_ratio > len(sorted_data) * 0.7:
            key_drivers.append("Strong dealer demand")
            if trend_direction == "up":
                trend_strength = min(1.0, trend_strength + 0.1)

        # Calculate confidence based on data quality and quantity
        confidence_score = min(1.0, len(sorted_data) / 10) * 0.7  # Data quantity factor
        if historical_data:
            confidence_score += 0.3  # Bonus for historical context

        return MarketTrend(
            segment=self._get_segment(sorted_data[0].model_code),
            trend_direction=trend_direction,
            strength=trend_strength,
            key_drivers=key_drivers,
            confidence_score=confidence_score,
            last_updated=datetime.now(),
        )

    def _get_segment(self, model_code: str) -> str:
        """
        Extract vehicle segment from model code.
        Ford model codes typically follow a pattern where specific positions indicate segment.

        Args:
            model_code: The Ford model code to analyze

        Returns:
            Identified vehicle segment (e.g., 'SUV-C', 'TRUCK-F', etc.)

        """
        # First character typically indicates vehicle type
        vehicle_type = model_code[0].upper()

        # Map vehicle type to segment
        segment_map = {
            "U": "SUV",  # Utility vehicles
            "T": "TRUCK",  # Trucks
            "P": "PICKUP",  # Pickup trucks
            "C": "CAR",  # Cars
            "V": "VAN",  # Vans
            "S": "SPORT",  # Sports cars
        }

        # Get base segment
        base_segment = segment_map.get(vehicle_type, "OTHER")

        # Second character often indicates size/class
        size_class = model_code[1].upper()
        size_map = {
            "S": "S",  # Small
            "M": "M",  # Medium
            "L": "L",  # Large
            "C": "C",  # Compact
            "F": "F",  # Full-size
            "X": "X",  # Extra large/Heavy duty
        }

        # Combine base segment with size class if available
        if size_class in size_map:
            return f"{base_segment}-{size_map[size_class]}"

        return base_segment

    def _generate_demand_forecast(
        self,
        fast_turn_data: List[FastTurnData],
        segment_trends: Dict[str, MarketTrend],
        market_conditions: Optional[Dict],
    ) -> Dict[str, float]:
        """
        Generate demand forecast by segment using trend analysis and market conditions.
        Returns a score between 0-1 indicating expected relative demand.

        Args:
            fast_turn_data: Current Fast Turn data
            segment_trends: Analyzed market trends by segment
            market_conditions: Optional market context

        Returns:
            Dictionary mapping segments to demand forecast scores (0-1)

        """
        forecasts = {}

        # Group data by segment
        segment_data = {}
        for d in fast_turn_data:
            segment = self._get_segment(d.model_code)
            if segment not in segment_data:
                segment_data[segment] = []
            segment_data[segment].append(d)

        # Generate forecast for each segment
        for segment, data in segment_data.items():
            # Start with current demand score
            base_demand = sum(d.market_demand_score for d in data) / len(data)

            # Factor in trend direction and strength
            trend = segment_trends.get(segment)
            if trend:
                trend_factor = 0.0
                if trend.trend_direction == "up":
                    trend_factor = trend.strength * 0.2  # Up to 20% boost
                elif trend.trend_direction == "down":
                    trend_factor = -trend.strength * 0.2  # Up to 20% reduction
                base_demand = min(1.0, max(0.0, base_demand * (1 + trend_factor)))

            # Consider market conditions if available
            if market_conditions:
                # Economic factors
                if "economic_outlook" in market_conditions:
                    economic_factor = market_conditions["economic_outlook"]  # -1 to 1 scale
                    base_demand *= 1 + (economic_factor * 0.1)  # ±10% adjustment

                # Seasonality
                if "seasonality" in market_conditions:
                    seasonality = market_conditions["seasonality"].get(
                        segment,
                        1.0,
                    )  # Default to no effect
                    base_demand *= seasonality

                # Competition
                if "competitive_pressure" in market_conditions:
                    comp_pressure = market_conditions["competitive_pressure"].get(
                        segment,
                        0.0,
                    )  # 0-1 scale
                    base_demand *= 1 - (comp_pressure * 0.15)  # Up to 15% reduction

            # Factor in dealer order ratio as indicator of real demand
            order_ratio = sum(
                1 for d in data if d.dealer_orders and d.dealer_orders > d.allocation_units
            ) / len(data)
            if order_ratio > 0.7:  # Strong dealer demand
                base_demand = min(1.0, base_demand * 1.1)  # 10% boost

            # Consider days supply as demand signal
            avg_days_supply = sum(d.days_supply for d in data) / len(data)
            if avg_days_supply < 30:  # Low supply could indicate strong demand
                base_demand = min(1.0, base_demand * 1.05)  # 5% boost
            elif avg_days_supply > 90:  # High supply could indicate weak demand
                base_demand = max(0.0, base_demand * 0.95)  # 5% reduction

            # Ensure forecast is in valid range
            forecasts[segment] = min(1.0, max(0.0, base_demand))

        return forecasts

    def _calculate_allocation_recommendations(
        self,
        demand_forecast: Dict[str, float],
        segment_trends: Dict[str, MarketTrend],
        market_conditions: Optional[Dict],
    ) -> Dict[str, int]:
        """Calculate recommended allocation units by segment"""
        # Implementation pending allocation calculation logic

    def _identify_risk_factors(
        self,
        segment_trends: Dict[str, MarketTrend],
        demand_forecast: Dict[str, float],
        market_conditions: Optional[Dict],
    ) -> List[str]:
        """Identify potential risk factors"""
        # Implementation pending risk analysis logic

    def _identify_opportunities(
        self,
        segment_trends: Dict[str, MarketTrend],
        demand_forecast: Dict[str, float],
        market_conditions: Optional[Dict],
    ) -> List[str]:
        """Identify market opportunities"""
        # Implementation pending opportunity analysis logic

    def _calculate_confidence_score(
        self,
        data_freshness: float,
        data_points: int,
        has_historical: bool,
        has_market_conditions: bool,
    ) -> float:
        """Calculate overall confidence score for the analysis"""
        # Base confidence from data freshness
        confidence = data_freshness * 0.4

        # Add confidence based on amount of data
        data_score = min(1.0, data_points / 100) * 0.3
        confidence += data_score

        # Add confidence for available context
        if has_historical:
            confidence += 0.2
        if has_market_conditions:
            confidence += 0.1

        return min(1.0, confidence)

    def _calculate_segment_metrics(self, segment_data: List[FastTurnData]) -> Dict[str, float]:
        """Calculate key metrics for a segment"""
        # Implementation pending metric calculation logic

    def _determine_insight_type(
        self,
        metrics: Dict[str, float],
        market_conditions: Optional[Dict],
    ) -> str:
        """Determine the type of market insight"""
        # Implementation pending insight classification logic

    def _calculate_impact_score(
        self,
        metrics: Dict[str, float],
        market_conditions: Optional[Dict],
    ) -> float:
        """Calculate impact score for an insight"""
        # Implementation pending impact scoring logic

    def _generate_recommendations(
        self,
        insight_type: str,
        metrics: Dict[str, float],
        market_conditions: Optional[Dict],
    ) -> List[str]:
        """Generate recommended actions"""
        # Implementation pending recommendation logic

    def _generate_insight_description(self, insight_type: str, metrics: Dict[str, float]) -> str:
        """Generate natural language description of the insight"""
        # Implementation pending description generation logic

    def _get_segment_from_model(self, model_code: str) -> str:
        """Extract segment information from model code"""
        # TODO: Implement proper segment mapping logic based on Ford's model codes
        segment_mappings = {
            'F': 'Full-Size Pickup',
            'S': 'SUV',
            'C': 'Crossover',
            'M': 'Midsize',
            'E': 'Electric'
        }
        return segment_mappings.get(model_code[0], 'Unknown')

    def _calculate_segment_trend(self, items: List[FastTurnData]) -> MarketTrend:
        """Calculate market trend for a segment based on Fast Turn data"""
        # Calculate average turn rate and trend direction
        avg_turn_rate = sum(item.turn_rate for item in items) / len(items)
        demand_scores = [item.market_demand_score for item in items]
        avg_demand = sum(demand_scores) / len(demand_scores)

        # Determine trend direction and strength
        if avg_demand > 0.7:
            direction = "up"
            strength = min(avg_demand, 1.0)
        elif avg_demand < 0.3:
            direction = "down"
            strength = 1.0 - avg_demand
        else:
            direction = "stable"
            strength = 0.5

        return MarketTrend(
            segment=self._get_segment_from_model(items[0].model_code),
            trend_direction=direction,
            strength=strength,
            key_drivers=self._identify_key_drivers(items),
            confidence_score=self._calculate_confidence_score(items),
            last_updated=max(item.timestamp for item in items)
        )

    def _calculate_demand_score(self, items: List[FastTurnData], trend: MarketTrend) -> float:
        """Calculate demand forecast score for a segment"""
        base_score = sum(item.market_demand_score for item in items) / len(items)
        trend_adjustment = 0.1 if trend.trend_direction == "up" else -0.1 if trend.trend_direction == "down" else 0
        return min(max(base_score + trend_adjustment, 0.0), 1.0)

    def _calculate_allocation_recommendation(
        self, 
        items: List[FastTurnData], 
        demand_score: float
    ) -> int:
        """Calculate recommended allocation units based on demand and current data"""
        current_allocation = sum(item.allocation_units for item in items)
        demand_factor = 1.0 + (demand_score - 0.5)  # Adjust by up to ±50% based on demand
        return int(current_allocation * demand_factor)

    def _identify_risk_factors(self, items: List[FastTurnData], trend: MarketTrend) -> List[str]:
        """Identify risk factors from market data"""
        risks = []
        avg_days_supply = sum(item.days_supply for item in items) / len(items)
        
        if avg_days_supply > 90:
            risks.append(f"High days supply ({int(avg_days_supply)} days) in {trend.segment}")
        if trend.trend_direction == "down" and trend.strength > 0.7:
            risks.append(f"Strong downward trend in {trend.segment} market")
        if any(item.turn_rate < 0.5 for item in items):
            risks.append(f"Low turn rate models detected in {trend.segment}")
            
        return risks

    def _identify_opportunities(
        self, 
        items: List[FastTurnData], 
        trend: MarketTrend, 
        demand_score: float
    ) -> List[str]:
        """Identify market opportunities from data"""
        opportunities = []
        
        if trend.trend_direction == "up" and demand_score > 0.7:
            opportunities.append(f"Strong growth potential in {trend.segment}")
        if any(item.market_demand_score > 0.8 for item in items):
            opportunities.append(f"High demand configurations identified in {trend.segment}")
            
        return opportunities

    def _calculate_confidence_score(self, data: List[FastTurnData]) -> float:
        """Calculate confidence score based on data quality and quantity"""
        if not data:
            return 0.0
            
        # Factors affecting confidence:
        # 1. Data freshness
        freshness_score = self._calculate_freshness_score(data)
        
        # 2. Data volume
        volume_score = min(len(data) / 10, 1.0)  # Normalize up to 10 data points
        
        # 3. Data consistency
        consistency_score = self._calculate_consistency_score(data)
        
        return (freshness_score + volume_score + consistency_score) / 3

    def _calculate_freshness_score(self, data: List[FastTurnData]) -> float:
        """Calculate data freshness score"""
        if not data:
            return 0.0
            
        now = datetime.utcnow()
        ages = [(now - item.timestamp).total_seconds() / 3600 for item in data]  # Ages in hours
        max_age = 72  # Consider data older than 72 hours as completely stale
        
        freshness_scores = [1.0 - min(age / max_age, 1.0) for age in ages]
        return sum(freshness_scores) / len(freshness_scores)

    def _calculate_consistency_score(self, data: List[FastTurnData]) -> float:
        """Calculate data consistency score"""
        if not data or len(data) < 2:
            return 1.0  # Single data point is considered consistent
            
        # Calculate variance in key metrics
        turn_rates = [item.turn_rate for item in data]
        demand_scores = [item.market_demand_score for item in data]
        
        # Normalize variances to [0,1] range where 0 is high variance
        tr_variance = 1.0 - min(self._calculate_variance(turn_rates), 1.0)
        ds_variance = 1.0 - min(self._calculate_variance(demand_scores), 1.0)
        
        return (tr_variance + ds_variance) / 2

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if not values or len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

    def _identify_key_drivers(self, items: List[FastTurnData]) -> List[str]:
        """Identify key drivers affecting market trends"""
        drivers = []
        
        # Analyze turn rate patterns
        high_turn = any(item.turn_rate > 0.8 for item in items)
        low_turn = any(item.turn_rate < 0.2 for item in items)
        
        if high_turn:
            drivers.append("High turn rate performance")
        if low_turn:
            drivers.append("Turn rate challenges")
            
        # Analyze demand patterns
        if any(item.market_demand_score > 0.7 for item in items):
            drivers.append("Strong market demand")
        if any(item.market_demand_score < 0.3 for item in items):
            drivers.append("Weak market demand")
            
        # Analyze supply patterns
        avg_days_supply = sum(item.days_supply for item in items) / len(items)
        if avg_days_supply > 75:
            drivers.append("High inventory levels")
        elif avg_days_supply < 30:
            drivers.append("Low inventory levels")
            
        return drivers

    def _calculate_impact_metrics(
        self, 
        trend: MarketTrend, 
        fast_turn_data: List[FastTurnData]
    ) -> Dict[str, float]:
        """Calculate impact metrics for market insights"""
        metrics = {
            "demand_strength": sum(d.market_demand_score for d in fast_turn_data) / len(fast_turn_data),
            "turn_rate_avg": sum(d.turn_rate for d in fast_turn_data) / len(fast_turn_data),
            "days_supply_avg": sum(d.days_supply for d in fast_turn_data) / len(fast_turn_data),
            "trend_strength": trend.strength,
            "data_confidence": trend.confidence_score
        }
        return metrics

    def _calculate_impact_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall impact score from metrics"""
        weights = {
            "demand_strength": 0.3,
            "turn_rate_avg": 0.2,
            "trend_strength": 0.3,
            "data_confidence": 0.2
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
        return min(max(score, 0.0), 1.0)

    def _determine_insight_type(self, trend: MarketTrend, impact_score: float) -> str:
        """Determine the type of market insight"""
        if trend.trend_direction == "up" and impact_score > 0.7:
            return "opportunity"
        elif trend.trend_direction == "down" and impact_score > 0.7:
            return "risk"
        elif impact_score > 0.5:
            return "trend_shift"
        else:
            return "market_update"

    def _calculate_priority(self, impact_score: float, confidence_score: float) -> str:
        """Calculate priority level of an insight"""
        combined_score = impact_score * confidence_score
        if combined_score > 0.7:
            return "high"
        elif combined_score > 0.4:
            return "medium"
        else:
            return "low"

    def _generate_action_items(
        self, 
        trend: MarketTrend, 
        impact_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate action items based on market trend and impact metrics"""
        actions = []
        
        if trend.trend_direction == "up":
            if impact_metrics["days_supply_avg"] < 45:
                actions.append(f"Increase allocation for {trend.segment} segment")
            if impact_metrics["turn_rate_avg"] > 0.7:
                actions.append(f"Review pricing strategy for {trend.segment}")
                
        elif trend.trend_direction == "down":
            if impact_metrics["days_supply_avg"] > 75:
                actions.append(f"Reduce allocation for {trend.segment} segment")
            if impact_metrics["turn_rate_avg"] < 0.3:
                actions.append(f"Evaluate incentive programs for {trend.segment}")
                
        if impact_metrics["data_confidence"] < 0.5:
            actions.append("Gather additional market data to improve confidence")
            
        return actions

    def _generate_insight_description(
        self, 
        trend: MarketTrend, 
        impact_metrics: Dict[str, float]
    ) -> str:
        """Generate a natural language description of the market insight"""
        direction = trend.trend_direction.title()
        strength = "Strong" if trend.strength > 0.7 else "Moderate" if trend.strength > 0.3 else "Weak"
        
        description = f"{strength} {direction} trend detected in {trend.segment} segment. "
        
        if impact_metrics["demand_strength"] > 0.7:
            description += "Market showing strong demand signals. "
        elif impact_metrics["demand_strength"] < 0.3:
            description += "Market showing weak demand signals. "
            
        if impact_metrics["days_supply_avg"] > 75:
            description += "Inventory levels are above target. "
        elif impact_metrics["days_supply_avg"] < 30:
            description += "Inventory levels are below target. "
            
        return description.strip()

    def _calculate_insight_expiration(self, trend: MarketTrend) -> datetime:
        """Calculate when this insight should expire"""
        from datetime import timedelta
        
        # Base expiration on trend strength and confidence
        if trend.strength > 0.7 and trend.confidence_score > 0.7:
            days = 7  # Strong, confident trends valid for a week
        elif trend.strength > 0.3 or trend.confidence_score > 0.5:
            days = 3  # Moderate trends or confidence valid for 3 days
        else:
            days = 1  # Weak trends need daily review
            
        return datetime.utcnow() + timedelta(days=days)
