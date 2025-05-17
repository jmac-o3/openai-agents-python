"""
Workflow Orchestration Manager for DOSO AI

This module provides workflow orchestration for multi-agent workflows,
managing the execution sequence and data flow between agents.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from openai_agents import Agent, RunConfig, run_agent

from ..agents.allocation_tracking import AllocationTrackingAgent
from ..agents.constraint_check import ConstraintCheckAgent
from ..agents.gap_analysis import GapAnalysisAgent
from ..agents.guidance_agent import GuidanceAgent
from ..agents.inventory_analysis import InventoryAnalysisAgent
from ..agents.market_analysis import MarketAnalysisAgent
from ..agents.order_bank_agent import OrderBankAgent
from ..agents.sva_agent import SVAAgent
from ..agents.triage_agent import TriageAgent
from ..db.redis import cache_key, get_cached_data, set_cached_data
from ..db.vector_store import VectorStore
from ..monitoring.tracing import create_trace, trace_operation
from ..schemas import DealerRequest, OrchestrationResult, TriageResponse, WorkflowType
from ..utils.telemetry import log_info, log_warning

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """
    Orchestrates the execution of multi-agent workflows for DOSO AI requests
    """

    def __init__(self):
        """Initialize orchestrator with agent registry and vector store"""
        # Register all available agents
        self.agent_registry = {
            "triage": TriageAgent,
            "inventory_analysis": InventoryAnalysisAgent,
            "market_analysis": MarketAnalysisAgent,
            "gap_analysis": GapAnalysisAgent,
            "allocation_tracking": AllocationTrackingAgent,
            "constraint_check": ConstraintCheckAgent,
            "order_bank_agent": OrderBankAgent,
            "guidance_agent": GuidanceAgent,
            "sva_agent": SVAAgent,
        }
        
        self.instance_cache = {}
        self.vector_store = VectorStore()
    
    async def process_request(
        self, 
        request: DealerRequest,
        trace_id: Optional[str] = None,
        run_config: Optional[RunConfig] = None,
    ) -> OrchestrationResult:
        """
        Process a dealer request through the appropriate workflow
        
        Args:
            request: Dealer request to process
            trace_id: Optional trace ID for request tracking
            run_config: Optional run configuration 
            
        Returns:
            OrchestrationResult with combined agent results
        """
        # Create tracing context
        trace_context = create_trace(
            operation="workflow_orchestration",
            trace_id=trace_id,
            dealer_id=request.dealer_id,
            request_type=request.request_type
        )
        
        # Check cache first
        cache_result = await self._check_cache(request)
        if cache_result:
            log_info("Retrieved result from cache", dealer_id=request.dealer_id)
            return cache_result
        
        # Start with triage agent to analyze request
        with trace_operation("triage_analysis", trace_context):
            triage_agent = self._get_agent_instance("triage")
            triage_response = await run_agent(
                agent=triage_agent, 
                messages=[("process_request", request)],
                run_config=run_config
            )
        
        # Convert triage response to TriageResponse if needed
        if isinstance(triage_response.result, dict):
            triage_result = TriageResponse(**triage_response.result)
        else:
            triage_result = triage_response.result
            
        # If request needs human review, return early
        if triage_result.needs_human_review:
            return OrchestrationResult(
                request_id=request.request_id,
                dealer_id=request.dealer_id,
                workflow_type=triage_result.workflow_type,
                is_complete=False,
                results={
                    "triage": triage_result.dict(),
                },
                recommendations=[
                    "This request requires human review before processing.",
                    f"Missing fields: {', '.join(triage_result.missing_fields)}"
                ],
                execution_time=triage_response.execution_time,
                next_steps=["Submit for human review"],
            )
        
        # Execute agent sequence
        agent_results = {"triage": triage_result}
        execution_times = [triage_response.execution_time]
        
        with trace_operation("agent_sequence_execution", trace_context):
            for agent_name in triage_result.agent_sequence:
                result, exec_time = await self._execute_agent(
                    agent_name=agent_name,
                    request=request,
                    prior_results=agent_results,
                    trace_context=trace_context,
                    run_config=run_config
                )
                
                agent_results[agent_name] = result
                execution_times.append(exec_time)
        
        # Generate response based on collected results
        with trace_operation("response_generation", trace_context):
            recommendations, next_steps = self._generate_recommendations(
                triage_result.workflow_type,
                agent_results
            )
        
        # Prepare orchestration result
        orchestration_result = OrchestrationResult(
            request_id=request.request_id,
            dealer_id=request.dealer_id,
            workflow_type=triage_result.workflow_type,
            is_complete=True,
            results=agent_results,
            recommendations=recommendations,
            execution_time=sum(execution_times),
            next_steps=next_steps,
        )
        
        # Cache result for future use
        await self._cache_result(request, orchestration_result)
        
        return orchestration_result

    async def _execute_agent(
        self,
        agent_name: str,
        request: DealerRequest,
        prior_results: Dict[str, Any],
        trace_context: Dict[str, Any],
        run_config: Optional[RunConfig] = None,
    ) -> Tuple[Any, float]:
        """
        Execute a single agent in the workflow
        
        Args:
            agent_name: Name of the agent to execute
            request: Original dealer request
            prior_results: Results from previously executed agents
            trace_context: Tracing context
            run_config: Optional run configuration
            
        Returns:
            Tuple of (agent_result, execution_time)
        """
        with trace_operation(f"execute_agent_{agent_name}", trace_context):
            try:
                # Get the appropriate agent instance
                agent = self._get_agent_instance(agent_name)
                
                # Determine which method to call based on agent type
                method_mapping = {
                    "inventory_analysis": "generate_inventory_analysis",
                    "market_analysis": "analyze_market_trends",
                    "gap_analysis": "perform_gap_analysis",
                    "allocation_tracking": "track_allocation",
                    "constraint_check": "validate_constraints",
                    "order_bank_agent": "process_order_bank",
                    "guidance_agent": "generate_guidance",
                    "sva_agent": "analyze_sales_velocity",
                }
                
                method_name = method_mapping.get(agent_name, "process")
                
                # Get relevant data from request and prior results
                agent_input = self._prepare_agent_input(
                    agent_name, 
                    request, 
                    prior_results
                )
                
                # Run the agent
                agent_response = await run_agent(
                    agent=agent,
                    messages=[(method_name, agent_input)],
                    run_config=run_config
                )
                
                return agent_response.result, agent_response.execution_time
                
            except Exception as e:
                log_warning(
                    f"Error executing agent {agent_name}", 
                    error=str(e), 
                    dealer_id=request.dealer_id
                )
                # Return error result
                return {
                    "status": "error",
                    "message": f"Agent execution failed: {str(e)}",
                }, 0.0

    def _get_agent_instance(self, agent_name: str) -> Agent:
        """
        Get or create an agent instance from the registry
        
        Args:
            agent_name: Name of the agent to retrieve
            
        Returns:
            Agent instance
        """
        # Check if we already have an instance
        if agent_name in self.instance_cache:
            return self.instance_cache[agent_name]
        
        # Get agent class from registry
        if agent_name not in self.agent_registry:
            raise ValueError(f"Unknown agent: {agent_name}")
            
        agent_class = self.agent_registry[agent_name]
        
        # Create new instance
        instance = agent_class()
        
        # Cache for future use
        self.instance_cache[agent_name] = instance
        
        return instance

    def _prepare_agent_input(
        self, 
        agent_name: str, 
        request: DealerRequest, 
        prior_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare input for an agent based on request and prior results
        
        Args:
            agent_name: Name of the agent
            request: Original dealer request
            prior_results: Results from previously executed agents
            
        Returns:
            Agent-specific input dictionary
        """
        # Start with the original request data
        agent_input = {
            "dealer_id": request.dealer_id,
            "request_type": request.request_type,
            **request.data,
        }
        
        # Add relevant data from prior results
        if agent_name == "gap_analysis" and "inventory_analysis" in prior_results:
            agent_input["inventory_metrics"] = prior_results["inventory_analysis"].get("metrics", {})
            
        if agent_name == "guidance_agent":
            # Add all prior results for guidance
            agent_input["prior_analysis"] = {k: v for k, v in prior_results.items() if k != "triage"}
            
        if agent_name == "order_bank_agent" and "constraint_check" in prior_results:
            agent_input["validated_constraints"] = prior_results["constraint_check"]
            
        return agent_input

    def _generate_recommendations(
        self,
        workflow_type: WorkflowType,
        agent_results: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """
        Generate recommendations and next steps based on agent results
        
        Args:
            workflow_type: Type of workflow executed
            agent_results: Results from all executed agents
            
        Returns:
            Tuple of (recommendations, next_steps)
        """
        # Default recommendations and next steps
        recommendations = []
        next_steps = []
        
        # Add results from guidance agent if available
        if "guidance_agent" in agent_results:
            guidance = agent_results["guidance_agent"]
            if isinstance(guidance, dict):
                recommendations.extend(guidance.get("recommendations", []))
                next_steps.extend(guidance.get("suggested_actions", []))
        
        # Add results from other agents based on workflow type
        if workflow_type == WorkflowType.INVENTORY_OPTIMIZATION:
            if "inventory_analysis" in agent_results:
                inv_analysis = agent_results["inventory_analysis"]
                if isinstance(inv_analysis, dict) and "insights" in inv_analysis:
                    recommendations.extend(inv_analysis["insights"])
            
            if "gap_analysis" in agent_results:
                gap_analysis = agent_results["gap_analysis"]
                if isinstance(gap_analysis, dict) and "opportunities" in gap_analysis:
                    next_steps.extend([
                        f"Address gap in {opp}" for opp in gap_analysis["opportunities"][:3]
                    ])
        
        # Ensure we have at least some recommendations and next steps
        if not recommendations:
            recommendations = ["No specific recommendations generated from analysis."]
            
        if not next_steps:
            next_steps = ["Consult with Ford representative for detailed action plan."]
            
        return recommendations, next_steps

    async def _check_cache(self, request: DealerRequest) -> Optional[OrchestrationResult]:
        """
        Check if result for this request is available in cache
        
        Args:
            request: Dealer request to check
            
        Returns:
            Cached result if available, None otherwise
        """
        cache_result = await get_cached_data(
            cache_key("orchestration", request.dealer_id, request.request_id)
        )
        
        if cache_result:
            return OrchestrationResult(**cache_result)
        return None

    async def _cache_result(self, request: DealerRequest, result: OrchestrationResult) -> None:
        """
        Cache orchestration result for future use
        
        Args:
            request: Original dealer request
            result: Orchestration result to cache
        """
        await set_cached_data(
            key=cache_key("orchestration", request.dealer_id, request.request_id),
            data=result.dict(),
            expire_seconds=3600 * 24  # Cache for 24 hours
        )
        
        # Also update vector store for semantic search
        await self._update_vector_store(request, result)

    async def _update_vector_store(
        self, 
        request: DealerRequest, 
        result: OrchestrationResult
    ) -> None:
        """
        Update vector store with request and result data for semantic search
        
        Args:
            request: Original dealer request
            result: Orchestration result 
        """
        # Extract key content for embedding
        content = {
            "request_id": request.request_id,
            "dealer_id": request.dealer_id,
            "request_type": request.request_type,
            "workflow_type": result.workflow_type.value,
            "recommendations": result.recommendations,
            "request_data": request.data,
        }
        
        # Store in vector DB for future semantic search
        await self.vector_store.add_document(
            document_id=request.request_id,
            content=content,
            metadata={
                "dealer_id": request.dealer_id,
                "request_type": request.request_type,
                "timestamp": str(request.timestamp),
            }
        )


# Create singleton instance
orchestrator = WorkflowOrchestrator()
