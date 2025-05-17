from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

from rich.console import Console

from agents import Runner, custom_span, gen_trace_id, trace
from .order_bank_agent import (
    OrderBankAnalysisRequest,
    OrderBankAnalysisResult,
    order_bank_agent
)

class OrderBankManager:
    """
    Orchestrates the order bank analysis workflow:
    1. Validates and preprocesses input data
    2. Runs analysis on order patterns
    3. Generates optimization recommendations
    4. Produces final report with insights
    """

    def __init__(self) -> None:
        self.console = Console()

    async def run_analysis(
        self, 
        dealer_id: str,
        start_date: datetime,
        end_date: datetime,
        include_historical: bool = False
    ) -> OrderBankAnalysisResult:
        """Run a complete order bank analysis for a dealer"""
        
        trace_id = gen_trace_id()
        with trace("Order Bank Analysis", trace_id=trace_id):
            # Create analysis request
            request = OrderBankAnalysisRequest(
                dealer_id=dealer_id,
                date_range_start=start_date,
                date_range_end=end_date,
                include_historical=include_historical
            )

            # Run the analysis using the agent
            with custom_span("Running order bank analysis"):
                result = await Runner.run(
                    order_bank_agent,
                    input=request.json()
                )
                return result.final_output_as(OrderBankAnalysisResult)

async def analyze_dealer_orders(
    dealer_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    include_historical: bool = False
) -> OrderBankAnalysisResult:
    """
    Convenience function to run order bank analysis for a dealer.
    If dates not specified, uses last 30 days by default.
    """
    if not start_date:
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = start_date.replace(day=start_date.day - 30)
    
    if not end_date:
        end_date = datetime.now()

    manager = OrderBankManager()
    return await manager.run_analysis(
        dealer_id=dealer_id,
        start_date=start_date,
        end_date=end_date,
        include_historical=include_historical
    )
