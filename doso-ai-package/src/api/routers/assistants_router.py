"""
Workflow Orchestration API using the openai-agents SDK with Assistants

This module provides API endpoints for workflow orchestration using
the openai-agents SDK with Assistants API for storage and processing.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile, Form
from pydantic import BaseModel

from ...agents.assistants_storage import assistants_storage
from ...schemas import DealerRequest, OrchestrationResult, WorkflowExecutionSummary
from ...workflow.assistants_orchestration import orchestrator
from ...monitoring.tracing import create_trace
from ...utils.telemetry import log_info, log_error

router = APIRouter()


class AssistantInfo(BaseModel):
    """Information about the connected OpenAI Assistant"""
    assistant_id: str
    name: str
    status: str
    file_count: int


@router.post("/process", response_model=OrchestrationResult)
async def process_dealer_request(
    request: DealerRequest,
    background_tasks: BackgroundTasks,
):
    """
    Process a dealer request through the Assistants-powered workflow
    
    This endpoint handles dealer requests by:
    1. Creating a trace context for monitoring
    2. Passing the request to the Assistants workflow orchestrator
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
        operation="process_dealer_request_assistants",
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


@router.get("/status/{request_id}", response_model=Dict[str, Any])
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
    
    # If not in cache, try to find in stored results
    try:
        # Search for this request ID in stored results
        results = await assistants_storage.search_documents(f"request_id: {request_id}")
        
        if results and len(results) > 0:
            # Found some results for this request
            return {
                "request_id": request_id,
                "status": "found",
                "stored_results": len(results),
                "message": "Request found in storage"
            }
        else:
            # No results found
            return {
                "request_id": request_id,
                "status": "unknown",
                "message": "Request ID not found in active workflows or storage"
            }
    except Exception as e:
        # Error during search
        log_error(e, operation="get_workflow_status", request_id=request_id)
        return {
            "request_id": request_id,
            "status": "error",
            "message": f"Error retrieving status: {str(e)}",
        }


@router.post("/upload", response_model=Dict[str, Any])
async def upload_file(
    dealer_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Upload a file for processing by the Assistants API
    
    Args:
        dealer_id: Dealer ID for association
        file: File to upload
        
    Returns:
        Status and information about the uploaded file
    """
    try:
        # Create a temporary file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + file.filename.split(".")[-1]) as temp:
            # Write uploaded file to temp file
            content = await file.read()
            temp.write(content)
            temp_path = temp.name
        
        # Store file metadata
        meta_content = {
            "original_file": file.filename,
            "dealer_id": dealer_id,
            "content_type": file.content_type,
            "upload_time": str(uuid.uuid4())  # Use UUID as timestamp equivalent
        }
        
        # Store in the Assistant
        result = await assistants_storage.store_analysis_result(
            dealer_id=dealer_id,
            analysis_type="file_upload",
            result_data=meta_content
        )
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
        
        return {
            "status": "success",
            "message": f"File {file.filename} uploaded successfully",
            "dealer_id": dealer_id,
            "file_name": file.filename
        }
        
    except Exception as e:
        log_error(e, operation="upload_file", dealer_id=dealer_id)
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}",
        )


@router.get("/search", response_model=List[Dict[str, Any]])
async def search_documents(
    query: str,
    dealer_id: Optional[str] = None,
):
    """
    Search documents stored in the Assistants API
    
    Args:
        query: Search query
        dealer_id: Optional dealer ID to filter results
        
    Returns:
        List of search results
    """
    try:
        # Construct search query
        search_query = query
        if dealer_id:
            search_query = f"dealer_id: {dealer_id} {query}"
            
        # Run search
        results = await assistants_storage.search_documents(search_query)
        return results
        
    except Exception as e:
        log_error(e, operation="search_documents", query=query)
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}",
        )


@router.get("/info", response_model=Dict[str, Any])
async def get_assistant_info():
    """
    Get information about the OpenAI Assistant used
    
    Returns:
        Information about the Assistant
    """
    try:
        # Just use a basic query to get some metadata
        results = await assistants_storage.search_documents("metadata")
        
        return {
            "status": "active",
            "connected": True,
            "file_count": len(results),
            "agent_name": "Assistants Storage Agent"
        }
        
    except Exception as e:
        log_error(e, operation="get_assistant_info")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving assistant info: {str(e)}",
        )


@router.get("/previous-analyses/{dealer_id}", response_model=List[Dict[str, Any]])
async def get_dealer_analyses(
    dealer_id: str,
    analysis_type: Optional[str] = None,
):
    """
    Get previous analyses for a dealer
    
    Args:
        dealer_id: Dealer ID
        analysis_type: Optional type of analysis to filter by
        
    Returns:
        List of previous analysis results
    """
    try:
        # Retrieve previous analyses
        results = await assistants_storage.retrieve_previous_analysis(
            dealer_id=dealer_id,
            analysis_type=analysis_type
        )
        return results
        
    except Exception as e:
        log_error(e, operation="get_dealer_analyses", dealer_id=dealer_id)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving dealer analyses: {str(e)}",
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
        f"Completed request processing for dealer {dealer_id} using Assistants",
        dealer_id=dealer_id,
        request_id=request_id,
        execution_time=execution_time,
    )
