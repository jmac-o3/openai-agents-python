"""
Workflow Orchestration Manager for DOSO AI using OpenAI Assistants API

This module provides workflow orchestration for multi-agent workflows using
OpenAI's Assistants API for file storage, semantic search, and agent execution
instead of PostgreSQL and Redis.
"""

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime
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
from ..monitoring.tracing import create_trace, trace_operation
from ..schemas import DealerRequest, OrchestrationResult, TriageResponse, WorkflowType
from ..utils.agent_query import run_agent_query, run_file_analysis
from ..utils.upload_to_vector_store import file_store
from ..utils.telemetry import log_info, log_warning

logger = logging.getLogger(__name__)


class OpenAIWorkflowOrchestrator:
    """
    Orchestrates the execution of multi-agent workflows for DOSO AI requests
    using OpenAI Assistants API instead of SQL databases
    """

    def __init__(self):
        """Initialize orchestrator with in-memory cache"""
        # Register all available agent types
        self.agent_types = {
            "triage": "Triage Agent",
            "inventory_analysis": "Inventory Analysis Agent",
            "market_analysis": "Market Analysis Agent",
            "gap_analysis": "Gap Analysis Agent",
            "allocation_tracking": "Allocation Tracking Agent",
            "constraint_check": "Constraint Check Agent",
            "order_bank_agent": "Order Bank Agent",
            "guidance_agent": "Guidance Agent",
            "sva_agent": "Sales Velocity Agent",
        }
        
        # In-memory cache for recent results
        self.result_cache = {}
    
    async def process_request(
        self, 
        request: DealerRequest,
        trace_id: Optional[str] = None,
        run_config: Optional[RunConfig] = None,
    ) -> OrchestrationResult:
        """
        Process a dealer request through the appropriate workflow using OpenAI Assistants
        
        Args:
            request: Dealer request to process
            trace_id: Optional trace ID for request tracking
            run_config: Optional run configuration 
            
        Returns:
            OrchestrationResult with combined agent results
        """
        # Create tracing context
        trace_context = create_trace(
            operation="workflow_orchestration_openai",
            trace_id=trace_id,
            dealer_id=request.dealer_id,
            request_type=request.request_type
        )
        
        # Check cache first
        cache_key = f"{request.dealer_id}:{request.request_id}"
        if cache_key in self.result_cache:
            log_info("Retrieved result from in-memory cache", dealer_id=request.dealer_id)
            return self.result_cache[cache_key]
        
        # Store uploaded files for later reference
        uploaded_files = {}
        
        # Start with triage agent to analyze request
        with trace_operation("triage_analysis", trace_context):
            # Run triage agent
            triage_result = await run_agent_query(
                agent_type="triage",
                query=request.dict(),
                trace_id=trace_id
            )
            
            # If not a dict, ensure it's properly formatted
            if not isinstance(triage_result, dict):
                triage_result = {
                    "workflow_type": "MANUAL_REVIEW",
                    "agent_sequence": [],
                    "needs_human_review": True,
                    "missing_fields": ["Invalid triage response format"]
                }
        
        # If request needs human review, return early
        if triage_result.get("needs_human_review", False):
            result = OrchestrationResult(
                request_id=request.request_id,
                dealer_id=request.dealer_id,
                workflow_type=triage_result.get("workflow_type", "MANUAL_REVIEW"),
                is_complete=False,
                results={
                    "triage": triage_result,
                },
                recommendations=[
                    "This request requires human review before processing.",
                    f"Missing fields: {', '.join(triage_result.get('missing_fields', ['Unknown']))}"
                ],
                execution_time=0,
                next_steps=["Submit for human review"],
            )
            
            # Store in cache
            self.result_cache[cache_key] = result
            return result
        
        # Upload any data files if included in the request
        if "files" in request.data and request.data["files"]:
            with trace_operation("upload_data_files", trace_context):
                for file_info in request.data["files"]:
                    if "path" in file_info and os.path.exists(file_info["path"]):
                        file_id = file_store.upload_and_attach_file(file_info["path"])
                        uploaded_files[file_info["path"]] = file_id
        
        # Execute agent sequence
        agent_results = {"triage": triage_result}
        execution_times = []
        
        with trace_operation("agent_sequence_execution", trace_context):
            workflow_type = triage_result.get("workflow_type", "GENERAL")
            agent_sequence = triage_result.get("agent_sequence", [])
            
            for agent_name in agent_sequence:
                # Check if we have any agent-specific files to process
                agent_files = []
                if agent_name == "inventory_analysis" and "inventory" in request.data:
                    agent_files = request.data["inventory"].get("files", [])
                elif agent_name == "market_analysis" and "market" in request.data:
                    agent_files = request.data["market"].get("files", [])
                elif agent_name == "sva_agent" and "sales" in request.data:
                    agent_files = request.data["sales"].get("files", [])
                
                # Prepare file paths for the agent
                file_paths = []
                for file_info in agent_files:
                    if "path" in file_info and os.path.exists(file_info["path"]):
                        file_paths.append(file_info["path"])
                
                # Prepare agent input
                agent_input = self._prepare_agent_input(
                    agent_name, 
                    request, 
                    agent_results
                )
                
                # Execute the agent
                start_time = datetime.now()
                
                try:
                    if agent_name == "inventory_analysis" and file_paths:
                        # Use specialized file analysis for inventory
                        result = await run_file_analysis(
                            file_paths=file_paths,
                            analysis_type="inventory",
                            parameters=agent_input
                        )
                    elif agent_name == "market_analysis" and file_paths:
                        # Use specialized file analysis for market data
                        result = await run_file_analysis(
                            file_paths=file_paths,
                            analysis_type="market",
                            parameters=agent_input
                        )
                    elif agent_name == "sva_agent" and file_paths:
                        # Use specialized file analysis for sales data
                        result = await run_file_analysis(
                            file_paths=file_paths,
                            analysis_type="sales",
                            parameters=agent_input
                        )
                    else:
                        # Use standard agent query execution
                        result = await run_agent_query(
                            agent_type=agent_name,
                            query=agent_input,
                            file_paths=file_paths,
                            trace_id=trace_id
                        )
                except Exception as e:
                    log_warning(
                        f"Error executing agent {agent_name}", 
                        error=str(e), 
                        dealer_id=request.dealer_id
                    )
                    # Return error result
                    result = {
                        "status": "error",
                        "message": f"Agent execution failed: {str(e)}",
                    }
                
                # Store the result
                agent_results[agent_name] = result
                
                # Calculate execution time
                end_time = datetime.now()
                exec_time = (end_time - start_time).total_seconds()
                execution_times.append(exec_time)
        
        # Generate response based on collected results
        with trace_operation("response_generation", trace_context):
            recommendations, next_steps = self._generate_recommendations(
                workflow_type,
                agent_results
            )
        
        # Prepare orchestration result
        orchestration_result = OrchestrationResult(
            request_id=request.request_id,
            dealer_id=request.dealer_id,
            workflow_type=workflow_type,
            is_complete=True,
            results=agent_results,
            recommendations=recommendations,
            execution_time=sum(execution_times),
            next_steps=next_steps,
        )
        
        # Cache result for future use
        self.result_cache[cache_key] = orchestration_result
        
        # Store in Assistants for future reference
        await self._store_result_for_reference(request, orchestration_result)
        
        return orchestration_result

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
        workflow_type: str,
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
        if workflow_type == "INVENTORY_OPTIMIZATION" or workflow_type == WorkflowType.INVENTORY_OPTIMIZATION:
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

    async def _store_result_for_reference(
        self, 
        request: DealerRequest, 
        result: OrchestrationResult
    ) -> None:
        """
        Store orchestration result for future reference in OpenAI Assistants
        
        Args:
            request: Original dealer request
            result: Orchestration result to store
        """
        # Create a temporary JSON file with the result
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp:
            json.dump(
                {
                    "request": request.dict(),
                    "result": result.dict()
                },
                temp,
                default=str,
                indent=2
            )
            temp_path = temp.name
        
        try:
            # Upload to OpenAI with metadata
            file_id = file_store.upload_and_attach_file(temp_path)
            logger.info(f"Stored result in OpenAI with file ID: {file_id}")
        except Exception as e:
            logger.error(f"Error storing result in OpenAI: {str(e)}")
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_path)
            except:
                pass


# Create singleton instance
orchestrator = OpenAIWorkflowOrchestrator()
