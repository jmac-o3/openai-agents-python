from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
import logging

from ...agents.allocation_tracking import AllocationTrackingAgent
from ...models.allocation_tracking import (
    AllocationHistory,
    AllocationPerformance,
    AllocationTracking,
    DealerAllocation
)
from ...utils.caching import cache_response, get_cached_response
from ...utils.auth import get_current_user

# Configure logging and tracing
logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Create router
router = APIRouter(
    prefix="/allocation",
    tags=["allocation"],
    responses={404: {"description": "Not found"}},
)

# Create agent instance
allocation_agent = AllocationTrackingAgent()


@router.get("/current/{dealer_id}", response_model=DealerAllocation)
async def get_current_allocation(
    dealer_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get the current allocation for a dealer
    """
    with tracer.start_as_current_span("get_current_allocation_endpoint") as span:
        try:
            span.set_attribute("dealer_id", dealer_id)
            logger.info(f"API request for current allocation: dealer_id={dealer_id}")
            
            # Check cache first
            cache_key = f"allocation:current:{dealer_id}"
            cached_data = get_cached_response(cache_key)
            if cached_data:
                logger.info(f"Returning cached allocation for dealer {dealer_id}")
                return cached_data
            
            # Fetch allocation data (in production, this would get data from a database)
            # For now, we'll use a placeholder response for demo purposes
            allocation_data = {
                "allocation_id": f"A{dealer_id}-{datetime.now().strftime('%Y%m')}",
                "allocation_date": datetime.now().isoformat(),
                "effective_date": datetime.now().isoformat(),
                "expiry_date": (datetime.now() + timedelta(days=14)).isoformat(),
                "line_items": [
                    {
                        "model_code": "F150",
                        "model_year": 2023,
                        "trim_level": "XLT",
                        "quantity": 5,
                        "status": "pending",
                        "accept_deadline": (datetime.now() + timedelta(days=7)).isoformat()
                    },
                    {
                        "model_code": "BRONCO",
                        "model_year": 2023,
                        "trim_level": "OUTER BANKS",
                        "quantity": 2,
                        "status": "pending",
                        "accept_deadline": (datetime.now() + timedelta(days=7)).isoformat()
                    }
                ]
            }
            
            # Process allocation data
            result = await allocation_agent.track_current_allocation(
                dealer_id=dealer_id,
                allocation_data=allocation_data
            )
            
            # Cache result
            cache_response(cache_key, result.dict(), expiry_seconds=3600)  # 1 hour cache
            
            return result
            
        except Exception as e:
            error_msg = f"Error retrieving current allocation for dealer {dealer_id}: {str(e)}"
            logger.error(error_msg)
            span.set_status(Status(StatusCode.ERROR), error_msg)
            raise HTTPException(status_code=500, detail=error_msg)


@router.get("/history/{dealer_id}", response_model=List[AllocationHistory])
async def get_allocation_history(
    dealer_id: str,
    periods: int = Query(4, ge=1, le=12),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get historical allocation data for a dealer
    """
    with tracer.start_as_current_span("get_allocation_history_endpoint") as span:
        try:
            span.set_attribute("dealer_id", dealer_id)
            span.set_attribute("periods", periods)
            logger.info(f"API request for allocation history: dealer_id={dealer_id}, periods={periods}")
            
            # Check cache first
            cache_key = f"allocation:history:{dealer_id}:{periods}"
            cached_data = get_cached_response(cache_key)
            if cached_data:
                logger.info(f"Returning cached allocation history for dealer {dealer_id}")
                return cached_data
            
            # Fetch historical data (placeholder for demo)
            historical_data = []
            now = datetime.now()
            
            # Generate sample data for last 4 quarters
            for i in range(periods):
                quarter = (now.month - 1) // 3 + 1 - i
                year = now.year
                if quarter < 1:
                    quarter += 4
                    year -= 1
                
                period = f"Q{quarter}-{year}"
                
                data = {
                    "time_period": period,
                    "total_units": 20 - i,  # Declining units for demo
                    "line_items": [
                        {
                            "model_code": "F150",
                            "model_year": year,
                            "quantity": 10 - i,
                            "status": "accepted" if i % 2 == 0 else "modified"
                        },
                        {
                            "model_code": "EXPLORER",
                            "model_year": year,
                            "quantity": 5,
                            "status": "accepted"
                        },
                        {
                            "model_code": "BRONCO",
                            "model_year": year,
                            "quantity": 5 - (i // 2),
                            "status": "declined" if i == 1 else "accepted"
                        }
                    ]
                }
                historical_data.append(data)
            
            # Process historical data
            result = await allocation_agent.analyze_allocation_history(
                dealer_id=dealer_id,
                historical_data=historical_data,
                time_periods=periods
            )
            
            # Cache result
            cache_response(cache_key, [r.dict() for r in result], expiry_seconds=86400)  # 1 day cache
            
            return result
            
        except Exception as e:
            error_msg = f"Error retrieving allocation history for dealer {dealer_id}: {str(e)}"
            logger.error(error_msg)
            span.set_status(Status(StatusCode.ERROR), error_msg)
            raise HTTPException(status_code=500, detail=error_msg)


@router.get("/performance/{dealer_id}", response_model=AllocationPerformance)
async def get_allocation_performance(
    dealer_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get allocation performance metrics for a dealer
    """
    with tracer.start_as_current_span("get_allocation_performance_endpoint") as span:
        try:
            span.set_attribute("dealer_id", dealer_id)
            logger.info(f"API request for allocation performance: dealer_id={dealer_id}")
            
            # Check cache first
            cache_key = f"allocation:performance:{dealer_id}"
            cached_data = get_cached_response(cache_key)
            if cached_data:
                logger.info(f"Returning cached allocation performance for dealer {dealer_id}")
                return cached_data
            
            # Get current allocation and history
            try:
                current_allocation = await get_current_allocation(dealer_id, current_user)
            except Exception:
                current_allocation = None
                
            try:
                historical_allocations = await get_allocation_history(dealer_id, 4, current_user)
            except Exception:
                historical_allocations = []
            
            # Get market data (placeholder for demo)
            market_data = {
                "demand_forecast": {
                    "F150": 0.85,
                    "EXPLORER": 0.65,
                    "BRONCO": 0.92,
                    "MUSTANG": 0.78,
                    "ESCAPE": 0.45,
                }
            }
            
            # Generate performance data
            result = await allocation_agent.generate_allocation_performance(
                dealer_id=dealer_id,
                current_allocation=current_allocation,
                historical_allocations=historical_allocations,
                market_data=market_data
            )
            
            # Cache result
            cache_response(cache_key, result.dict(), expiry_seconds=43200)  # 12 hour cache
            
            return result
            
        except Exception as e:
            error_msg = f"Error retrieving allocation performance for dealer {dealer_id}: {str(e)}"
            logger.error(error_msg)
            span.set_status(Status(StatusCode.ERROR), error_msg)
            raise HTTPException(status_code=500, detail=error_msg)


@router.get("/tracking/{dealer_id}", response_model=AllocationTracking)
async def get_allocation_tracking(
    dealer_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get comprehensive allocation tracking data for a dealer
    """
    with tracer.start_as_current_span("get_allocation_tracking_endpoint") as span:
        try:
            span.set_attribute("dealer_id", dealer_id)
            logger.info(f"API request for allocation tracking: dealer_id={dealer_id}")
            
            # Check cache first
            cache_key = f"allocation:tracking:{dealer_id}"
            cached_data = get_cached_response(cache_key)
            if cached_data:
                logger.info(f"Returning cached allocation tracking for dealer {dealer_id}")
                return cached_data
            
            # Get current allocation data
            try:
                # For current allocation we need the raw data
                now = datetime.now()
                current_allocation_data = {
                    "allocation_id": f"A{dealer_id}-{now.strftime('%Y%m')}",
                    "allocation_date": now.isoformat(),
                    "effective_date": now.isoformat(),
                    "expiry_date": (now + timedelta(days=14)).isoformat(),
                    "line_items": [
                        {
                            "model_code": "F150",
                            "model_year": 2023,
                            "trim_level": "XLT",
                            "quantity": 5,
                            "status": "pending",
                            "accept_deadline": (now + timedelta(days=7)).isoformat()
                        },
                        {
                            "model_code": "BRONCO",
                            "model_year": 2023,
                            "trim_level": "OUTER BANKS",
                            "quantity": 2,
                            "status": "pending",
                            "accept_deadline": (now + timedelta(days=7)).isoformat()
                        }
                    ]
                }
            except Exception:
                current_allocation_data = None
                
            # Get historical data (same sample as in history endpoint)
            historical_data = []
            now = datetime.now()
            
            # Generate sample data for last 4 quarters
            for i in range(4):
                quarter = (now.month - 1) // 3 + 1 - i
                year = now.year
                if quarter < 1:
                    quarter += 4
                    year -= 1
                
                period = f"Q{quarter}-{year}"
                
                data = {
                    "time_period": period,
                    "total_units": 20 - i,
                    "line_items": [
                        {
                            "model_code": "F150",
                            "model_year": year,
                            "quantity": 10 - i,
                            "status": "accepted" if i % 2 == 0 else "modified"
                        },
                        {
                            "model_code": "EXPLORER",
                            "model_year": year,
                            "quantity": 5,
                            "status": "accepted"
                        },
                        {
                            "model_code": "BRONCO",
                            "model_year": year,
                            "quantity": 5 - (i // 2),
                            "status": "declined" if i == 1 else "accepted"
                        }
                    ]
                }
                historical_data.append(data)
                
            # Get market data (placeholder)
            market_data = {
                "demand_forecast": {
                    "F150": 0.85,
                    "EXPLORER": 0.65,
                    "BRONCO": 0.92,
                    "MUSTANG": 0.78,
                    "ESCAPE": 0.45,
                },
                "segment_trends": {
                    "trucks": {
                        "trend_direction": "up",
                        "strength": 0.8
                    },
                    "suvs": {
                        "trend_direction": "stable",
                        "strength": 0.6
                    }
                }
            }
            
            # Generate tracking data
            result = await allocation_agent.track_allocation(
                dealer_id=dealer_id,
                current_allocation_data=current_allocation_data,
                historical_data=historical_data,
                market_data=market_data
            )
            
            # Cache result
            cache_response(cache_key, result.dict(), expiry_seconds=3600)  # 1 hour cache
            
            return result
            
        except Exception as e:
            error_msg = f"Error retrieving allocation tracking for dealer {dealer_id}: {str(e)}"
            logger.error(error_msg)
            span.set_status(Status(StatusCode.ERROR), error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
