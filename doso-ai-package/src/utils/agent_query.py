"""
Agent Query Execution using OpenAI Assistants

This module provides functions to run agent queries using OpenAI Assistants API
instead of the direct SQL-based data access, providing a transition from
PostgreSQL/Redis to OpenAI's file storage and vector search.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from openai import OpenAI
from openai.types.beta import Assistant
from openai.types.beta.threads import ThreadMessage, Run

from ..config import settings
from ..monitoring.tracing import trace_async_method, trace_method
from .upload_to_vector_store import file_store

logger = logging.getLogger(__name__)


@trace_async_method("agent_query.run_agent_query")
async def run_agent_query(
    agent_type: str,
    query: Dict[str, Any],
    file_paths: Optional[List[str]] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute an agent query using OpenAI Assistants API
    
    Args:
        agent_type: Type of agent to run (e.g., inventory_analysis)
        query: Query parameters and context
        file_paths: Optional paths to files to include in the query
        trace_id: Optional trace ID for tracking
        
    Returns:
        Agent response as a dictionary
    """
    # Upload files if provided
    file_ids = []
    if file_paths:
        for file_path in file_paths:
            file_id = file_store.upload_and_attach_file(file_path)
            file_ids.append(file_id)
    
    # Build agent-specific prompt
    prompt = _build_agent_prompt(agent_type, query)
    
    # Add metadata
    metadata = {
        "agent_type": agent_type,
        "trace_id": trace_id or "none",
        "dealer_id": query.get("dealer_id", "unknown")
    }
    
    # Create thread and run query
    result = await file_store.create_thread_and_run(
        query=prompt,
        file_ids=file_ids,
        metadata=metadata
    )
    
    # Parse the response based on expected format
    return await _parse_agent_response(agent_type, result["response"])


def _build_agent_prompt(agent_type: str, query: Dict[str, Any]) -> str:
    """
    Build a specialized prompt for the agent type
    
    Args:
        agent_type: Type of agent
        query: Query parameters
        
    Returns:
        Specialized prompt
    """
    # Base agent information
    agent_info = {
        "inventory_analysis": {
            "task": "Analyze inventory metrics and provide insights",
            "output_format": "JSON with metrics, insights, risk_factors, and opportunities keys",
            "relevant_data": "inventory data, vehicle information, and sales history"
        },
        "market_analysis": {
            "task": "Analyze market trends and competitive positioning",
            "output_format": "JSON with market_share, trends, competitors, and recommendations keys",
            "relevant_data": "market data, competitive positioning, and regional trends"
        },
        "gap_analysis": {
            "task": "Identify gaps in inventory mix compared to ideal allocation",
            "output_format": "JSON with gaps, opportunities, and recommendations keys",
            "relevant_data": "inventory metrics, market data, and sales velocity"
        },
        "allocation_tracking": {
            "task": "Track current allocations and future orders",
            "output_format": "JSON with current_allocation, pending_orders, and status keys",
            "relevant_data": "allocation records, dealer constraints, and order status"
        },
        "constraint_check": {
            "task": "Validate order constraints against rules",
            "output_format": "JSON with constraints, validation_results, and is_valid keys",
            "relevant_data": "constraint rules, order specifications, and allocation limits"
        },
        "order_bank_agent": {
            "task": "Process and optimize order bank requests",
            "output_format": "JSON with order_recommendations, prioritized_models, and timing keys",
            "relevant_data": "validated constraints, inventory gaps, and market trends"
        },
        "guidance_agent": {
            "task": "Generate guidance based on all analyses",
            "output_format": "JSON with recommendations, suggested_actions, and priorities keys",
            "relevant_data": "results from all previous analyses"
        },
        "sva_agent": {
            "task": "Analyze sales velocity and aging metrics",
            "output_format": "JSON with velocity_metrics, aging_analysis, and action_items keys",
            "relevant_data": "sales history, inventory aging, and regional trends"
        },
        "triage": {
            "task": "Analyze the dealer request and determine processing sequence",
            "output_format": "JSON with workflow_type, agent_sequence, and needs_human_review keys",
            "relevant_data": "dealer request details and associated metadata"
        }
    }
    
    info = agent_info.get(agent_type, {
        "task": "Process the provided data",
        "output_format": "JSON with relevant results",
        "relevant_data": "all available data"
    })
    
    # Prepare query for prompt
    query_json = json.dumps(query, indent=2)
    
    # Build specialized prompt
    prompt = f"""
    As a DOSO AI {agent_type.replace('_', ' ')} agent, your task is to {info['task']}.
    
    Analyze the following data:
    ```json
    {query_json}
    ```

    Search through any attached files for relevant information about {info['relevant_data']}.
    
    Instructions:
    1. Use your file_search tool to find relevant information in the attached files
    2. Analyze the data thoroughly and draw meaningful insights
    3. Format your response as {info['output_format']}
    4. Ensure your response is well-structured and maintains proper JSON formatting

    Your output MUST be valid JSON that can be parsed programmatically. Use this structure:
    ```json
    {{
      "key1": value1,
      "key2": value2,
      ...
    }}
    ```
    
    Remember to escape any quotes or special characters in your JSON values.
    """
    
    return prompt


async def _parse_agent_response(agent_type: str, response: str) -> Dict[str, Any]:
    """
    Parse the agent response based on expected format
    
    Args:
        agent_type: Type of agent
        response: Raw response text
        
    Returns:
        Parsed response as a dictionary
    """
    # Extract JSON from response if needed (handle markdown code blocks)
    if "```json" in response:
        import re
        json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
    elif "```" in response:
        import re
        json_match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
    
    # Try to parse as JSON
    try:
        parsed = json.loads(response)
        return parsed
    except json.JSONDecodeError:
        logger.error(f"Failed to parse response as JSON for agent {agent_type}")
        
        # Return a fallback response based on agent type
        if agent_type == "inventory_analysis":
            return {
                "metrics": {},
                "insights": ["Error parsing response"],
                "risk_factors": [],
                "opportunities": []
            }
        elif agent_type == "market_analysis":
            return {
                "market_share": {},
                "trends": [],
                "competitors": [],
                "recommendations": ["Error parsing response"]
            }
        elif agent_type == "triage":
            return {
                "workflow_type": "MANUAL_REVIEW",
                "agent_sequence": [],
                "needs_human_review": True,
                "missing_fields": ["Error parsing response"]
            }
        else:
            return {
                "status": "error",
                "message": "Failed to parse response",
                "raw_response": response
            }


@trace_async_method("agent_query.run_file_analysis")
async def run_file_analysis(
    file_paths: List[str],
    analysis_type: str,
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run analysis on specific files
    
    Args:
        file_paths: Paths to files to analyze
        analysis_type: Type of analysis to perform
        parameters: Optional parameters for analysis
        
    Returns:
        Analysis results
    """
    # Upload files to OpenAI
    file_ids = []
    for file_path in file_paths:
        file_id = file_store.upload_and_attach_file(file_path)
        file_ids.append(file_id)
    
    # Build analysis prompt
    if analysis_type == "inventory":
        prompt = """
        Analyze the attached inventory data files and extract the following metrics:
        1. Total units by model
        2. Total value by model
        3. Average days on lot
        4. Turnover rate
        5. Aging distribution
        
        Return your analysis as structured JSON with these keys:
        - metrics: containing numerical metrics
        - insights: list of key observations
        - risk_factors: potential issues identified
        - opportunities: potential opportunities identified
        
        Format your response as proper JSON.
        """
    elif analysis_type == "sales":
        prompt = """
        Analyze the attached sales data files and extract the following metrics:
        1. Sales by model
        2. Average sale price by model
        3. Sales velocity (units per day)
        4. Regional distribution based on zip codes
        5. Trends over time
        
        Return your analysis as structured JSON with these keys:
        - metrics: containing numerical metrics
        - customer_insights: patterns in customer behavior
        - regional_patterns: geographical observations
        - recommendations: actionable suggestions
        
        Format your response as proper JSON.
        """
    elif analysis_type == "market":
        prompt = """
        Analyze the attached market data files and extract the following:
        1. Market share percentages
        2. Competitive positioning
        3. Trend direction and magnitude
        4. Key competitors by model
        
        Return your analysis as structured JSON with these keys:
        - market_share: percentage by model
        - trends: list of trend observations
        - competitors: key competitor analysis
        - recommendations: strategic suggestions
        
        Format your response as proper JSON.
        """
    else:
        prompt = f"""
        Analyze the attached files for the purpose of {analysis_type} analysis.
        
        Extract relevant metrics and insights based on the file contents.
        
        Return your analysis as structured JSON with appropriate keys for {analysis_type} analysis.
        
        Format your response as proper JSON.
        """
    
    # Add any custom parameters
    if parameters:
        param_json = json.dumps(parameters, indent=2)
        prompt += f"\n\nConsider these parameters in your analysis:\n```json\n{param_json}\n```"
    
    # Create thread and run query
    result = await file_store.create_thread_and_run(
        query=prompt,
        file_ids=file_ids,
        metadata={"analysis_type": analysis_type}
    )
    
    # Parse response
    return await _parse_agent_response(analysis_type, result["response"])
