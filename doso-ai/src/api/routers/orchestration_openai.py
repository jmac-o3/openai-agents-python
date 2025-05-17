"""
Workflow Orchestration API routes using OpenAI Assistants API

This module provides API endpoints for workflow orchestration using 
OpenAI's Assistants API instead of PostgreSQL/Redis.
"""

import asyncio
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from ...schemas import DealerRequest, OrchestrationResult, WorkflowExecutionSummary
from ...workflow.orchestration_openai import orchestrator
from ...monitoring.tracing import create_trace
from ...utils.telemetry import log_info, log_error
from ...utils.upload_to_vector_store import file_store

router = APIRouter()


@router.post("/process", response_model=OrchestrationResult)
async def process_dealer_request(
    request: DealerRequest,
    background_tasks: BackgroundTasks,
):
    """
    Process a dealer request through the OpenAI Assistants-powered workflow

    This endpoint handles dealer requests by:
    1. Creating a trace context for monitoring
    2. Passing the request to the OpenAI workflow orchestrator
    3. Returning the orchestration result

    Args:
        request: Dealer request to process
        background_tasks: FastAPI background tasks

    Returns:
        OrchestrationResult containing the processed results
    """
    # Create trace ID for request tracking
    trace_id = str(uuid.uuid4())
    trace_context = create_trace(
        operation="process_dealer_request_openai",
        trace_id=trace_id,
        dealer_id=request.dealer_id,
        request_type=request.request_type,
    )

    try:
        # Process the request through the orchestrator
        result = await orchestrator.process_request(
            request=request,
            trace_id=trace_id,
        )

        # Log processing in background
        background_tasks.add_task(
            _log_request_processing,
            request.dealer_id,
            request.request_id,
            result.execution_time,
        )

        return result

    except Exception as e:
        log_error(e, dealer_id=request.dealer_id, request_id=request.request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}",
        )


@router.get("/status/{request_id}", response_model=Dict)
async def get_workflow_status(
    request_id: str,
):
    """
    Get the status of a workflow execution

    Args:
        request_id: ID of the request to check

    Returns:
        Status information about the workflow
    """
    # Check in-memory cache first
    if hasattr(orchestrator, "result_cache"):
        for cache_key, result in orchestrator.result_cache.items():
            if request_id in cache_key:
                return {
                    "request_id": request_id,
                    "status": "completed",
                    "is_complete": result.is_complete,
                    "workflow_type": result.workflow_type,
                    "dealer_id": result.dealer_id,
                }
    
    # If not in cache, return unknown status
    return {
        "request_id": request_id,
        "status": "unknown",
        "message": "Request ID not found in active workflows",
    }


@router.get("/summary/{workflow_id}", response_model=WorkflowExecutionSummary)
async def get_workflow_summary(
    workflow_id: str,
):
    """
    Get detailed summary of workflow execution

    Args:
        workflow_id: ID of the workflow execution

    Returns:
        WorkflowExecutionSummary with detailed metrics
    """
    # Query OpenAI for records matching this workflow ID
    try:
        assistant_id = file_store.ensure_assistant_exists()
        
        # For now, we don't have a good way to search by ID, so return not implemented
        # In a full implementation, this would use file_store to retrieve the summary
        raise HTTPException(status_code=501, detail="Not yet implemented")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving workflow summary: {str(e)}",
        )


@router.get("/files", response_model=List[Dict])
async def list_assistant_files():
    """
    List all files attached to the DOSO AI assistant

    Returns:
        List of file information dictionaries
    """
    try:
        files = file_store.list_assistant_files()
        return files
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing assistant files: {str(e)}",
        )


@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """
    Delete a file from the OpenAI assistant

    Args:
        file_id: ID of the file to delete

    Returns:
        Success or error message
    """
    try:
        success = file_store.delete_file(file_id)
        if success:
            return {"status": "success", "message": f"File {file_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"File {file_id} not found or could not be deleted",
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting file: {str(e)}",
        )


async def _log_request_processing(
    dealer_id: str,
    request_id: str,
    execution_time: float,
):
    """
    Log request processing metrics (runs in background)

    Args:
        dealer_id: Dealer ID
        request_id: Request ID
        execution_time: Total execution time
    """
    log_info(
        f"Completed request processing for dealer {dealer_id} using OpenAI Assistants",
        dealer_id=dealer_id,
        request_id=request_id,
        execution_time=execution_time,
    )
