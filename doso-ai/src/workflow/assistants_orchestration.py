"""
Workflow Orchestration using openai-agents Assistants SDK

This module provides workflow orchestration for multi-agent workflows in DOSO AI
using native openai-agents SDK for storage and agent execution.
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
from ..agents.assistants_storage import assistants_storage
from ..monitoring.tracing import create_trace, trace_operation
from ..schemas import DealerRequest, OrchestrationResult, TriageResponse, WorkflowType
from ..utils.telemetry import log_info, log_warning

logger = logging.getLogger(__name__)


class AssistantsWorkflowOrchestrator:
    """
    Orchestrates the execution of multi-agent workflows using openai-agents SDK
    for Assistants-based storage and execution
    """

    def __init__(self):
        """Initialize orchestrator"""
        # Register all available agent types and their classes
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
        
        # In-memory cache for recent results
        self.result_cache = {}
    
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
            operation="workflow_orchestration_assistants",
            trace_id=trace_id,
            dealer_id=request.dealer_id,
            request_type=request.request_type
        )
        
        # Check cache first
        cache_key = f"{request.dealer_id}:{request.request_id}"
        if cache_key in self.result_cache:
            log_info("Retrieved result from in-memory cache", dealer_id=request.dealer_id)
            return self.result_cache[cache_key]
        
        # Handle any file uploads first
        uploaded_file_paths = []
        if "files" in request.data and isinstance(request.data["files"], list):
            # Process each file by creating a temporary file that can be uploaded
            for file_info in request.data["files"]:
                if isinstance(file_info, dict) and "path" in file_info and os.path.exists(file_info["path"]):
                    uploaded_file_paths.append(file_info["path"])
                    
                    # Create a structured JSON file about this upload for later search
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as meta_file:
                        meta_content = {
                            "original_file": os.path.basename(file_info["path"]),
                            "dealer_id": request.dealer_id,
                            "request_id": request.request_id,
                            "upload_time": datetime.now().isoformat()
                        }
                        json.dump(meta_content, meta_file, indent=2)
                        meta_path = meta_file.name
                    
                    # Store the metadata file
                    await assistants_storage.store_analysis_result(
                        dealer_id=request.dealer_id,
                        analysis_type="file_metadata",
                        result_data=meta_content
                    )
                    
                    # Clean up temp file
                    try:
                        os.unlink(meta_path)
                    except:
                        pass
        
        # Start with triage agent to analyze request
        with trace_operation("triage_analysis", trace_context):
            # Get the triage agent from the registry
            triage_agent_class = self.agent_registry["triage"]
            triage_agent = triage_agent_class()
            
            # Run triage analysis
            triage_response = await triage_agent.analyze_request(request.dict())
            
            # Convert to expected format
            if isinstance(triage_response, dict):
                triage_result = triage_response
            else:
                # If response is not directly a dict (e.g., it's a model class)
                triage_result = {
                    "workflow_type": getattr(triage_response, "workflow_type", "MANUAL_REVIEW"),
                    "agent_sequence": getattr(triage_response, "agent_sequence", []),
                    "needs_human_review": getattr(triage_response, "needs_human_review", True),
                    "missing_fields": getattr(triage_response, "missing_fields", ["Error in triage response"])
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
        
        # Execute agent sequence
        agent_results = {"triage": triage_result}
        execution_times = []
        
        with trace_operation("agent_sequence_execution", trace_context):
            workflow_type = triage_result.get("workflow_type", "GENERAL")
            agent_sequence = triage_result.get("agent_sequence", [])
            
            for agent_name in agent_sequence:
                if agent_name not in self.agent_registry:
                    log_warning(f"Unknown agent type: {agent_name}", dealer_id=request.dealer_id)
                    continue
                
                # Get agent class and create instance
                agent_class = self.agent_registry[agent_name]
                agent = agent_class()
                
                # Check if we have any agent-specific files to process
                agent_files = []
                if agent_name == "inventory_analysis" and "inventory" in request.data:
                    if "files" in request.data["inventory"] and isinstance(request.data["inventory"]["files"], list):
                        agent_files = [f["path"] for f in request.data["inventory"]["files"] 
                                      if isinstance(f, dict) and "path" in f and os.path.exists(f["path"])]
                elif agent_name == "market_analysis" and "market" in request.data:
                    if "files" in request.data["market"] and isinstance(request.data["market"]["files"], list):
                        agent_files = [f["path"] for f in request.data["market"]["files"] 
                                      if isinstance(f, dict) and "path" in f and os.path.exists(f["path"])]
                elif agent_name == "sva_agent" and "sales" in request.data:
                    if "files" in request.data["sales"] and isinstance(request.data["sales"]["files"], list):
                        agent_files = [f["path"] for f in request.data["sales"]["files"] 
                                      if isinstance(f, dict) and "path" in f and os.path.exists(f["path"])]
                
                # Prepare agent input with prior results
                agent_input = await self._prepare_agent_input(
                    agent_name, 
                    request, 
                    agent_results
                )
                
                # Execute the agent
                start_time = datetime.now()
                
                try:
                    # Find the appropriate method to call based on agent type
                    if agent_name == "inventory_analysis" and hasattr(agent, "analyze_inventory_metrics"):
                        # Use specialized method for inventory analysis
                        metrics = await agent.analyze_inventory_metrics(
                            current_inventory=agent_input.get("current_inventory", {}),
                            historical_data=agent_input.get("historical_data", []),
                            time_period_days=agent_input.get("time_period_days", 90)
                        )
                        
                        # Now generate the full analysis with insights
                        result = await agent.generate_inventory_analysis(
                            metrics=metrics,
                            market_data=agent_input.get("market_data", {})
                        )
                        
                        # Convert Pydantic model to dict if needed
                        if hasattr(result, "dict"):
                            result = result.dict()
                        
                    elif agent_name == "market_analysis" and hasattr(agent, "analyze_market_trends"):
                        # Use specialized method for market analysis
                        result = await agent.analyze_market_trends(
                            market_data=agent_input.get("market_data", {}),
                            timeframe=agent_input.get("timeframe", "90 days")
                        )
                        
                        # Convert Pydantic model to dict if needed
                        if hasattr(result, "dict"):
                            result = result.dict()
                            
                    elif agent_name == "gap_analysis" and hasattr(agent, "identify_inventory_gaps"):
                        # Use specialized method for gap analysis
                        result = await agent.identify_inventory_gaps(
                            current_inventory=agent_input.get("current_inventory", {}),
                            ideal_mix=agent_input.get("ideal_mix", {}),
                            market_data=agent_input.get("market_data", {})
                        )
                        
                        # Convert Pydantic model to dict if needed
                        if hasattr(result, "dict"):
                            result = result.dict()
                            
                    else:
                        # Generic approach: run the agent with the context
                        raw_result = await run_agent(
                            agent=agent,
                            messages=[{"role": "user", "content": json.dumps(agent_input, default=str)}],
                            run_config=run_config
                        )
                        
                        # Extract the content from the result
                        result_content = raw_result.content
                        
                        # Try to parse as JSON if possible
                        try:
                            # If wrapped in markdown code blocks, extract the JSON
                            if "```json" in result_content:
                                import re
                                json_match = re.search(r'```json\n(.*?)\n```', result_content, re.DOTALL)
                                if json_match:
                                    result_content = json_match.group(1)
                                    
                            # Parse the JSON
                            result = json.loads(result_content)
                            
                        except (json.JSONDecodeError, TypeError):
                            # If not JSON, use the raw content
                            result = {"raw_result": result_content}
                    
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
                
                # Store the analysis result for future reference
                try:
                    await assistants_storage.store_analysis_result(
                        dealer_id=request.dealer_id,
                        analysis_type=agent_name,
                        result_data={
                            "request_id": request.request_id,
                            "timestamp": datetime.now().isoformat(),
                            "result": result
                        }
                    )
                except Exception as e:
                    log_warning(f"Error storing analysis result: {str(e)}", dealer_id=request.dealer_id)
                
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
        
        return orchestration_result

    async def _prepare_agent_input(
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
        
        # Check for previous analyses for this dealer
        try:
            previous_analyses = await assistants_storage.retrieve_previous_analysis(
                dealer_id=request.dealer_id,
                analysis_type=agent_name
            )
            
            if previous_analyses and len(previous_analyses) > 0:
                agent_input["previous_analyses"] = previous_analyses
        except Exception as e:
            log_warning(f"Error retrieving previous analyses: {str(e)}", dealer_id=request.dealer_id)
            
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


# Create singleton instance
orchestrator = AssistantsWorkflowOrchestrator()
