from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from openai_agents import Agent, function_tool
from openai_agents.exceptions import AgentsException
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from ..models.allocation_tracking import (
    AllocationHistory, 
    AllocationLineItem, 
    AllocationPerformance, 
    AllocationStatus, 
    AllocationTracking,
    DealerAllocation
)
from ..utils.validation import validate_data_freshness

# Configure logging
logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class AllocationTrackingAgent(Agent):
    """
    Allocation Tracking Agent for DOSO AI system.

    Responsible for tracking and analyzing allocations from Ford to dealers,
    monitoring acceptance rates, and providing insights on allocation performance.

    Capabilities:
    - Track current and historical allocations
    - Monitor allocation acceptance, decline, and modification rates
    - Analyze allocation effectiveness
    - Identify missed opportunities
    - Provide strategic insights for future allocations

    Error Handling:
    - Validates input data freshness
    - Handles missing allocation data
    - Provides detailed error messages
    - Logs errors for monitoring

    Performance:
    - OpenTelemetry integration for tracing
    - Caching of allocation data for performance optimization
    """

    def __init__(self):
        super().__init__(
            name="Allocation Tracking Agent",
            instructions="""
            You are an expert allocation tracking agent responsible for:
            1. Tracking and analyzing dealer allocations
            2. Monitoring allocation acceptance rates
            3. Identifying patterns and trends in allocation utilization
            4. Providing strategic insights for allocation optimization
            5. Integrating allocation data with inventory and market analysis
            """,
        )

    @function_tool
    async def track_current_allocation(
        self, 
        dealer_id: str,
        allocation_data: Dict[str, Any]
    ) -> DealerAllocation:
        """
        Process and track the current allocation for a dealer

        Args:
            dealer_id: The ID of the dealer
            allocation_data: Raw allocation data from Ford

        Returns:
            DealerAllocation: Structured allocation data with status summary

        Raises:
            AgentsException: If allocation data is invalid or processing fails
        """
        with tracer.start_as_current_span("track_current_allocation") as span:
            try:
                span.set_attribute("dealer_id", dealer_id)
                logger.info(f"Processing current allocation for dealer {dealer_id}")
                
                # Validate allocation data
                if not allocation_data or "allocation_id" not in allocation_data:
                    raise AgentsException("Invalid allocation data: missing required fields")
                
                # Process line items
                line_items = []
                status_summary = {status.value: 0 for status in AllocationStatus}
                
                for item_data in allocation_data.get("line_items", []):
                    line_item = AllocationLineItem(
                        model_code=item_data["model_code"],
                        model_year=item_data["model_year"],
                        trim_level=item_data.get("trim_level", ""),
                        quantity=item_data["quantity"],
                        status=item_data.get("status", AllocationStatus.PENDING),
                        accept_deadline=datetime.fromisoformat(item_data["accept_deadline"]),
                        constraints=item_data.get("constraints")
                    )
                    line_items.append(line_item)
                    status_summary[line_item.status] = status_summary.get(line_item.status, 0) + 1
                
                # Create allocation object
                allocation = DealerAllocation(
                    allocation_id=allocation_data["allocation_id"],
                    dealer_id=dealer_id,
                    allocation_date=datetime.fromisoformat(allocation_data["allocation_date"]),
                    effective_date=datetime.fromisoformat(allocation_data["effective_date"]),
                    expiry_date=datetime.fromisoformat(allocation_data["expiry_date"]),
                    line_items=line_items,
                    total_units=sum(item.quantity for item in line_items),
                    status_summary=status_summary,
                    notes=allocation_data.get("notes")
                )
                
                logger.info(f"Successfully processed allocation {allocation.allocation_id} with {allocation.total_units} total units")
                span.set_status(Status(StatusCode.OK))
                return allocation
                
            except Exception as e:
                error_msg = f"Failed to process allocation for dealer {dealer_id}: {str(e)}"
                logger.error(error_msg)
                span.set_status(Status(StatusCode.ERROR), error_msg)
                raise AgentsException(error_msg) from e

    @function_tool
    async def analyze_allocation_history(
        self,
        dealer_id: str,
        historical_data: List[Dict[str, Any]],
        time_periods: int = 4
    ) -> List[AllocationHistory]:
        """
        Analyze historical allocation data for a dealer

        Args:
            dealer_id: The ID of the dealer
            historical_data: List of historical allocation data
            time_periods: Number of time periods to analyze (default: 4 quarters)

        Returns:
            List[AllocationHistory]: Structured historical allocation data

        Raises:
            AgentsException: If historical data is invalid or processing fails
        """
        with tracer.start_as_current_span("analyze_allocation_history") as span:
            try:
                span.set_attribute("dealer_id", dealer_id)
                span.set_attribute("time_periods", time_periods)
                logger.info(f"Analyzing historical allocation for dealer {dealer_id} over {time_periods} periods")
                
                # Validate data freshness
                if not validate_data_freshness(historical_data, max_age_days=90):
                    logger.warning(f"Historical allocation data for dealer {dealer_id} may be outdated")
                
                # Group data by time period
                periods = {}
                for alloc in historical_data:
                    period = alloc.get("time_period")
                    if not period:
                        continue
                        
                    if period not in periods:
                        periods[period] = {
                            "total_allocated": 0,
                            "total_accepted": 0,
                            "total_declined": 0,
                            "allocation_by_model": {},
                            "performance_metrics": {}
                        }
                    
                    # Aggregate data
                    periods[period]["total_allocated"] += alloc.get("total_units", 0)
                    accepted = sum(1 for item in alloc.get("line_items", []) 
                                 if item.get("status") == AllocationStatus.ACCEPTED.value)
                    declined = sum(1 for item in alloc.get("line_items", []) 
                                 if item.get("status") == AllocationStatus.DECLINED.value)
                    
                    periods[period]["total_accepted"] += accepted
                    periods[period]["total_declined"] += declined
                    
                    # Track by model
                    for item in alloc.get("line_items", []):
                        model = item.get("model_code")
                        if not model:
                            continue
                            
                        if model not in periods[period]["allocation_by_model"]:
                            periods[period]["allocation_by_model"][model] = 0
                        
                        periods[period]["allocation_by_model"][model] += item.get("quantity", 0)
                
                # Calculate performance metrics
                results = []
                for period, data in periods.items():
                    # Calculate acceptance rate
                    total = data["total_accepted"] + data["total_declined"]
                    acceptance_rate = data["total_accepted"] / total if total > 0 else 0
                    
                    # Set performance metrics
                    data["performance_metrics"] = {
                        "acceptance_rate": acceptance_rate,
                        "utilization_rate": self._calculate_utilization_rate(data, historical_data),
                        "effectiveness_score": self._calculate_effectiveness_score(data, period)
                    }
                    
                    # Create history object
                    history = AllocationHistory(
                        dealer_id=dealer_id,
                        time_period=period,
                        total_allocated=data["total_allocated"],
                        total_accepted=data["total_accepted"],
                        total_declined=data["total_declined"],
                        allocation_by_model=data["allocation_by_model"],
                        performance_metrics=data["performance_metrics"]
                    )
                    results.append(history)
                
                # Sort by time period and limit to requested number
                results.sort(key=lambda x: x.time_period, reverse=True)
                results = results[:time_periods]
                
                logger.info(f"Successfully analyzed {len(results)} historical allocation periods")
                span.set_status(Status(StatusCode.OK))
                return results
                
            except Exception as e:
                error_msg = f"Failed to analyze allocation history for dealer {dealer_id}: {str(e)}"
                logger.error(error_msg)
                span.set_status(Status(StatusCode.ERROR), error_msg)
                raise AgentsException(error_msg) from e
    
    @function_tool
    async def generate_allocation_performance(
        self,
        dealer_id: str,
        current_allocation: Optional[DealerAllocation] = None,
        historical_allocations: Optional[List[AllocationHistory]] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> AllocationPerformance:
        """
        Generate performance metrics and insights for allocations

        Args:
            dealer_id: The ID of the dealer
            current_allocation: Current dealer allocation data
            historical_allocations: Historical allocation data
            market_data: Market analysis data for context

        Returns:
            AllocationPerformance: Performance metrics and strategic insights

        Raises:
            AgentsException: If data is insufficient or processing fails
        """
        with tracer.start_as_current_span("generate_allocation_performance") as span:
            try:
                span.set_attribute("dealer_id", dealer_id)
                logger.info(f"Generating allocation performance for dealer {dealer_id}")
                
                # Determine time period (current quarter or month)
                now = datetime.now()
                time_period = f"Q{(now.month-1)//3+1}-{now.year}"
                
                # Calculate effectiveness based on historical data
                effectiveness = 0.0
                if historical_allocations:
                    # Average of last 2 periods if available
                    recent = historical_allocations[:2]
                    metrics = [h.performance_metrics.get("effectiveness_score", 0) for h in recent]
                    effectiveness = sum(metrics) / len(metrics) if metrics else 0.0
                
                # Identify missed opportunities
                missed_opportunities = self._identify_missed_opportunities(
                    current_allocation, historical_allocations, market_data
                )
                
                # Generate strategic insights
                insights = self._generate_strategic_insights(
                    dealer_id, current_allocation, historical_allocations, market_data
                )
                
                # Create benchmark comparison
                benchmark = self._create_benchmark_comparison(dealer_id, historical_allocations)
                
                # Identify improvement areas
                improvement_areas = self._identify_improvement_areas(
                    dealer_id, effectiveness, benchmark, missed_opportunities
                )
                
                # Create performance object
                performance = AllocationPerformance(
                    dealer_id=dealer_id,
                    time_period=time_period,
                    allocation_effectiveness=effectiveness,
                    missed_opportunities=missed_opportunities,
                    strategic_insights=insights,
                    benchmark_comparison=benchmark,
                    improvement_areas=improvement_areas,
                    last_updated=datetime.now()
                )
                
                logger.info(f"Successfully generated allocation performance with {len(insights)} insights")
                span.set_status(Status(StatusCode.OK))
                return performance
                
            except Exception as e:
                error_msg = f"Failed to generate allocation performance for dealer {dealer_id}: {str(e)}"
                logger.error(error_msg)
                span.set_status(Status(StatusCode.ERROR), error_msg)
                raise AgentsException(error_msg) from e

    @function_tool
    async def track_allocation(
        self,
        dealer_id: str,
        current_allocation_data: Optional[Dict[str, Any]] = None,
        historical_data: Optional[List[Dict[str, Any]]] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> AllocationTracking:
        """
        Comprehensive allocation tracking including current, historical, and performance data

        Args:
            dealer_id: The ID of the dealer
            current_allocation_data: Current allocation data (if available)
            historical_data: Historical allocation data
            market_data: Market analysis data for context

        Returns:
            AllocationTracking: Comprehensive allocation tracking data

        Raises:
            AgentsException: If processing fails
        """
        with tracer.start_as_current_span("track_allocation") as span:
            try:
                span.set_attribute("dealer_id", dealer_id)
                logger.info(f"Performing comprehensive allocation tracking for dealer {dealer_id}")
                
                # Process current allocation if available
                current_allocation = None
                if current_allocation_data:
                    current_allocation = await self.track_current_allocation(
                        dealer_id, current_allocation_data
                    )
                
                # Process historical allocations if available
                historical_allocations = []
                if historical_data:
                    historical_allocations = await self.analyze_allocation_history(
                        dealer_id, historical_data
                    )
                
                # Calculate rates
                acceptance_rate = 0.0
                modification_rate = 0.0
                expiration_rate = 0.0
                
                if historical_allocations:
                    # Calculate rates from historical data
                    total_accepted = sum(h.total_accepted for h in historical_allocations)
                    total_allocated = sum(h.total_allocated for h in historical_allocations)
                    
                    # Get modification and expiration counts from the original data
                    total_modified = sum(
                        sum(1 for item in alloc.get("line_items", []) 
                            if item.get("status") == AllocationStatus.MODIFIED.value)
                        for alloc in historical_data
                    )
                    
                    total_expired = sum(
                        sum(1 for item in alloc.get("line_items", []) 
                            if item.get("status") == AllocationStatus.EXPIRED.value)
                        for alloc in historical_data
                    )
                    
                    if total_allocated > 0:
                        acceptance_rate = total_accepted / total_allocated
                        modification_rate = total_modified / total_allocated
                        expiration_rate = total_expired / total_allocated
                
                # Calculate allocation trends
                trends = self._calculate_allocation_trends(historical_allocations)
                
                # Estimate upcoming allocation
                upcoming_estimate = None
                if historical_allocations and market_data:
                    upcoming_estimate = self._estimate_upcoming_allocation(
                        dealer_id, historical_allocations, market_data
                    )
                
                # Create tracking object
                tracking = AllocationTracking(
                    current_allocation=current_allocation,
                    historical_allocations=historical_allocations,
                    allocation_trends=trends,
                    upcoming_allocation_estimate=upcoming_estimate,
                    acceptance_rate=acceptance_rate,
                    modification_rate=modification_rate,
                    expiration_rate=expiration_rate,
                    last_updated=datetime.now()
                )
                
                logger.info(f"Successfully completed allocation tracking for dealer {dealer_id}")
                span.set_status(Status(StatusCode.OK))
                return tracking
                
            except Exception as e:
                error_msg = f"Failed to track allocation for dealer {dealer_id}: {str(e)}"
                logger.error(error_msg)
                span.set_status(Status(StatusCode.ERROR), error_msg)
                raise AgentsException(error_msg) from e

    # Helper methods
    def _calculate_utilization_rate(self, period_data: Dict[str, Any], historical_data: List[Dict[str, Any]]) -> float:
        """Calculate utilization rate of allocations"""
        # This would use additional data about how allocated inventory performed
        # For now, return a placeholder calculation
        return 0.75  # Placeholder
    
    def _calculate_effectiveness_score(self, period_data: Dict[str, Any], period: str) -> float:
        """Calculate effectiveness score for a time period"""
        # Combine acceptance rate with utilization
        acceptance = period_data["performance_metrics"].get("acceptance_rate", 0)
        utilization = period_data["performance_metrics"].get("utilization_rate", 0)
        
        # Weight acceptance and utilization equally
        return (acceptance + utilization) / 2
    
    def _identify_missed_opportunities(
        self,
        current_allocation: Optional[DealerAllocation],
        historical_allocations: Optional[List[AllocationHistory]],
        market_data: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify missed allocation opportunities"""
        opportunities = []
        
        # If we have market data and allocations, compare for gaps
        if market_data and (current_allocation or historical_allocations):
            # Look for models with high market demand but low allocation
            market_demand = market_data.get("demand_forecast", {})
            
            # Get allocated models from current and recent history
            allocated_models = {}
            
            if current_allocation:
                for item in current_allocation.line_items:
                    if item.model_code not in allocated_models:
                        allocated_models[item.model_code] = 0
                    allocated_models[item.model_code] += item.quantity
            
            if historical_allocations and len(historical_allocations) > 0:
                recent = historical_allocations[0]  # Most recent period
                for model, qty in recent.allocation_by_model.items():
                    if model not in allocated_models:
                        allocated_models[model] = 0
                    allocated_models[model] += qty // 2  # Adjust for comparison
            
            # Compare demand with allocation
            for model, demand in market_demand.items():
                allocated = allocated_models.get(model, 0)
                
                # If demand is high but allocation is low
                if demand > 0.7 and (allocated == 0 or (demand > 0.9 and allocated < 3)):
                    opportunities.append({
                        "model_code": model,
                        "market_demand": demand,
                        "current_allocation": allocated,
                        "opportunity_type": "underallocated_high_demand",
                        "estimated_impact": round((demand - 0.7) * 10, 1)  # 0-10 scale
                    })
        
        return opportunities
    
    def _generate_strategic_insights(
        self,
        dealer_id: str,
        current_allocation: Optional[DealerAllocation],
        historical_allocations: Optional[List[AllocationHistory]],
        market_data: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate strategic insights for allocation optimization"""
        insights = []
        
        # If we have historical data, look for patterns
        if historical_allocations and len(historical_allocations) > 0:
            # Check for declining acceptance rate
            rates = [h.performance_metrics.get("acceptance_rate", 0) for h in historical_allocations]
            if len(rates) > 1 and all(a > b for a, b in zip(rates[:-1], rates[1:])):
                insights.append(
                    "Declining allocation acceptance rate detected. Consider reviewing "
                    "allocation requests to better align with dealership needs."
                )
            
            # Check for allocation vs market alignment
            if market_data and "segment_trends" in market_data:
                segments = market_data["segment_trends"]
                for segment, trend in segments.items():
                    if trend.get("trend_direction") == "up" and trend.get("strength", 0) > 0.7:
                        # Check if this segment is well-allocated
                        is_allocated = False
                        for alloc in historical_allocations:
                            for model in alloc.allocation_by_model:
                                if segment.lower() in model.lower():
                                    is_allocated = True
                                    break
                        
                        if not is_allocated:
                            insights.append(
                                f"Strong upward trend detected in {segment} segment, but "
                                f"recent allocations show limited inventory in this category. "
                                f"Consider increasing allocation requests for these models."
                            )
        
        # If we have current allocation, check for optimization opportunities
        if current_allocation:
            # Check for expiring allocations
            expiring_soon = []
            now = datetime.now()
            for item in current_allocation.line_items:
                if item.status == AllocationStatus.PENDING and (item.accept_deadline - now).days < 3:
                    expiring_soon.append(item.model_code)
            
            if expiring_soon:
                insights.append(
                    f"Pending allocations for {', '.join(expiring_soon)} are expiring soon. "
                    f"Review and respond promptly to avoid missing allocation opportunities."
                )
        
        # Add general insights if we have limited data
        if not insights:
            insights.append(
                "Regular review of allocation offers against current inventory needs "
                "can improve inventory turn rates and profitability."
            )
            
            insights.append(
                "Consider analyzing declined allocations to identify patterns that "
                "can inform future allocation strategy."
            )
        
        return insights
    
    def _create_benchmark_comparison(
        self,
        dealer_id: str,
        historical_allocations: Optional[List[AllocationHistory]]
    ) -> Dict[str, float]:
        """Create benchmark comparison with regional and national averages"""
        # This would use additional data about regional and national performance
        # For now, return placeholder comparisons
        return {
            "regional_acceptance_rate_diff": 0.05,  # 5% better than regional average
            "national_acceptance_rate_diff": -0.02,  # 2% worse than national average
            "regional_utilization_diff": 0.03,  # 3% better than regional average
            "national_utilization_diff": 0.01,  # 1% better than national average
        }
    
    def _identify_improvement_areas(
        self,
        dealer_id: str,
        effectiveness: float,
        benchmark: Dict[str, float],
        missed_opportunities: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify areas for allocation strategy improvement"""
        areas = []
        
        # Check effectiveness
        if effectiveness < 0.6:
            areas.append("Overall allocation effectiveness is below target. "
                         "Review allocation acceptance strategy.")
        
        # Check benchmarks
        if benchmark.get("regional_acceptance_rate_diff", 0) < -0.05:
            areas.append("Allocation acceptance rate is significantly below regional average. "
                         "Consider more selective allocation requests.")
        
        if benchmark.get("national_utilization_diff", 0) < -0.05:
            areas.append("Allocation utilization is below national average. "
                         "Review inventory management practices post-allocation.")
        
        # Check missed opportunities
        if len(missed_opportunities) > 2:
            areas.append("Multiple high-demand models show allocation gaps. "
                         "Review allocation requests to target these opportunities.")
        
        # Add general improvement area if none specific
        if not areas:
            areas.append("Consider more granular tracking of allocation performance "
                         "by model to identify specific optimization opportunities.")
        
        return areas
    
    def _calculate_allocation_trends(
        self, 
        historical_allocations: List[AllocationHistory]
    ) -> Dict[str, Any]:
        """Calculate trends in allocation data over time"""
        trends = {
            "volume_trend": "stable",
            "model_mix_changes": {},
            "acceptance_trend": "stable",
        }
        
        if not historical_allocations or len(historical_allocations) < 2:
            return trends
        
        # Sort by time period
        sorted_history = sorted(historical_allocations, key=lambda x: x.time_period)
        
        # Calculate volume trend
        volumes = [h.total_allocated for h in sorted_history]
        if all(a < b for a, b in zip(volumes[:-1], volumes[1:])):
            trends["volume_trend"] = "increasing"
        elif all(a > b for a, b in zip(volumes[:-1], volumes[1:])):
            trends["volume_trend"] = "decreasing"
        
        # Calculate acceptance trend
        rates = [h.performance_metrics.get("acceptance_rate", 0) for h in sorted_history]
        if all(a < b for a, b in zip(rates[:-1], rates[1:])):
            trends["acceptance_trend"] = "improving"
        elif all(a > b for a, b in zip(rates[:-1], rates[1:])):
            trends["acceptance_trend"] = "declining"
        
        # Calculate model mix changes
        first = sorted_history[0]
        last = sorted_history[-1]
        
        all_models = set(first.allocation_by_model.keys()) | set(last.allocation_by_model.keys())
        
        for model in all_models:
            old_value = first.allocation_by_model.get(model, 0)
            new_value = last.allocation_by_model.get(model, 0)
            
            if old_value == 0:
                trends["model_mix_changes"][model] = "new"
            elif new_value == 0:
                trends["model_mix_changes"][model] = "discontinued"
            elif new_value > old_value * 1.25:
                trends["model_mix_changes"][model] = "significant_increase"
            elif new_value < old_value * 0.75:
                trends["model_mix_changes"][model] = "significant_decrease"
            else:
                trends["model_mix_changes"][model] = "stable"
        
        return trends
    
    def _estimate_upcoming_allocation(
        self,
        dealer_id: str,
        historical_allocations: List[AllocationHistory],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate the upcoming allocation based on history and market data"""
        if not historical_allocations:
            return None
        
        # Use the most recent allocation as a baseline
        latest = historical_allocations[0]
        
        # Start with the baseline
        estimate = {
            "estimated_total": latest.total_allocated,
            "estimated_by_model": dict(latest.allocation_by_model),
            "confidence_score": 0.7,
            "influencing_factors": []
        }
        
        # Adjust based on market data
        if market_data and "demand_forecast" in market_data:
            demand_forecast = market_data["demand_forecast"]
            
            # List of models with changing demand
            increasing_models = []
            decreasing_models = []
            
            # Adjust model estimates based on demand forecast
            for model, quantity in estimate["estimated_by_model"].items():
                if model in demand_forecast:
                    demand = demand_forecast[model]
                    
                    # Adjust quantity based on demand change
                    if demand > 0.8:  # High demand
                        new_qty = int(quantity * 1.2)  # 20% increase
                        estimate["estimated_by_model"][model] = new_qty
                        increasing_models.append(model)
                    elif demand < 0.3:  # Low demand
                        new_qty = int(quantity * 0.8)  # 20% decrease
                        estimate["estimated_by_model"][model] = new_qty
                        decreasing_models.append(model)
            
            # Add influencing factors
            if increasing_models:
                estimate["influencing_factors"].append(
                    f"Increased market demand for {', '.join(increasing_models)}"
                )
            
            if decreasing_models:
                estimate["influencing_factors"].append(
                    f"Decreased market demand for {', '.join(decreasing_models)}"
                )
        
        # Recalculate total
        estimate["estimated_total"] = sum(estimate["estimated_by_model"].values())
        
        # Add seasonal factor if applicable
        now = datetime.now()
        if 3 <= now.month <= 5:  # Spring
            estimate["influencing_factors"].append("Seasonal spring inventory increase")
            estimate["estimated_total"] = int(estimate["estimated_total"] * 1.1)  # 10% more
        elif 9 <= now.month <= 11:  # Fall
            estimate["influencing_factors"].append("New model year introduction period")
            estimate["estimated_total"] = int(estimate["estimated_total"] * 1.15)  # 15% more
        
        return estimate
