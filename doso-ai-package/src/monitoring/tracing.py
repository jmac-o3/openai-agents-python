"""
Monitoring and tracing utilities for DOSO AI

Implements OpenTelemetry integration for comprehensive tracing and monitoring
"""

import contextlib
import logging
import time
import uuid
from functools import wraps
from typing import Any, Callable, Dict, Generator, List, Optional, Type, TypeVar, cast

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from ..config import settings

# Get tracer for this module
tracer = trace.get_tracer("doso-ai.monitoring")
logger = logging.getLogger(__name__)

# Type variables for generic function wrapping
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


def create_trace(
    operation: str, 
    trace_id: Optional[str] = None, 
    **attributes: Any
) -> Dict[str, Any]:
    """
    Create a new trace context with unique trace ID
    
    Args:
        operation: Name of the operation being traced
        trace_id: Optional trace ID to use (generate if not provided)
        **attributes: Additional attributes to add to trace
        
    Returns:
        Trace context dictionary
    """
    # Generate trace ID if not provided
    if trace_id is None:
        trace_id = str(uuid.uuid4())
        
    # Create trace context
    with tracer.start_as_current_span(
        operation, 
        kind=SpanKind.INTERNAL,
        attributes={"trace.id": trace_id, **attributes}
    ) as span:
        current_span_context = span.get_span_context()
        
    # Return trace context for child spans
    return {
        "trace_id": trace_id, 
        "parent_context": current_span_context,
        "attributes": attributes
    }


@contextlib.contextmanager
def trace_operation(
    operation_name: str, 
    trace_context: Dict[str, Any],
    **attributes: Any
) -> Generator[Span, None, None]:
    """
    Context manager for tracing an operation
    
    Args:
        operation_name: Name of the operation being traced
        trace_context: Trace context from create_trace
        **attributes: Additional attributes to add to span
        
    Yields:
        Active span for the operation
    """
    # Combine trace attributes with operation-specific attributes
    all_attributes = {
        **trace_context.get("attributes", {}),
        **attributes,
        "trace.id": trace_context.get("trace_id")
    }
    
    # Start span with parent context if available
    with tracer.start_as_current_span(
        operation_name,
        context=trace_context.get("parent_context"),
        attributes=all_attributes,
        kind=SpanKind.INTERNAL
    ) as span:
        try:
            yield span
        except Exception as e:
            # Record exception in span
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(e)
            raise


def trace_method(operation_name: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator for tracing class methods
    
    Args:
        operation_name: Name of the operation (default to method name)
        
    Returns:
        Decorated method with tracing
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Get operation name from param or function name
            op_name = operation_name or f"{self.__class__.__name__}.{func.__name__}"
            
            # Check if trace context is passed as kwarg
            trace_context = kwargs.pop("trace_context", None) or {}
            
            # Get attributes from self if available
            attributes = {}
            if hasattr(self, "get_trace_attributes"):
                attributes = self.get_trace_attributes()
                
            # Start span
            with trace_operation(op_name, trace_context, **attributes) as span:
                start_time = time.time()
                result = func(self, *args, **kwargs)
                duration = time.time() - start_time
                
                # Add result status and timing
                span.set_attribute("execution.time_ms", duration * 1000)
                
                return result
                
        return cast(F, wrapper)
    return decorator


def trace_async_method(operation_name: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator for tracing async class methods
    
    Args:
        operation_name: Name of the operation (default to method name)
        
    Returns:
        Decorated async method with tracing
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Get operation name from param or function name
            op_name = operation_name or f"{self.__class__.__name__}.{func.__name__}"
            
            # Check if trace context is passed as kwarg
            trace_context = kwargs.pop("trace_context", None) or {}
            
            # Get attributes from self if available
            attributes = {}
            if hasattr(self, "get_trace_attributes"):
                attributes = self.get_trace_attributes()
                
            # Start span
            with trace_operation(op_name, trace_context, **attributes) as span:
                start_time = time.time()
                result = await func(self, *args, **kwargs)
                duration = time.time() - start_time
                
                # Add result status and timing
                span.set_attribute("execution.time_ms", duration * 1000)
                
                return result
                
        return cast(F, wrapper)
    return decorator


def log_agent_execution(
    agent_name: str,
    operation: str,
    execution_time: float,
    token_usage: Optional[Dict[str, int]] = None,
    status: str = "success",
    error: Optional[str] = None,
) -> None:
    """
    Log agent execution with telemetry
    
    Args:
        agent_name: Name of the agent
        operation: Operation being performed
        execution_time: Execution time in seconds
        token_usage: Optional token usage statistics
        status: Execution status (success/error)
        error: Optional error message
    """
    # Extract current span context
    current_span = trace.get_current_span()
    
    # Add attributes to span
    current_span.set_attribute("agent.name", agent_name)
    current_span.set_attribute("agent.operation", operation)
    current_span.set_attribute("agent.execution_time", execution_time)
    current_span.set_attribute("agent.status", status)
    
    if token_usage:
        for key, value in token_usage.items():
            current_span.set_attribute(f"agent.tokens.{key}", value)
            
    if error:
        current_span.set_attribute("agent.error", error)
        current_span.set_status(Status(StatusCode.ERROR))
        
    # Log details
    log_data = {
        "agent_name": agent_name,
        "operation": operation,
        "execution_time": execution_time,
        "status": status,
    }
    
    if token_usage:
        log_data["token_usage"] = token_usage
        
    if error:
        log_data["error"] = error
        logger.error(f"Agent execution error: {agent_name}", extra=log_data)
    else:
        logger.info(f"Agent execution: {agent_name}.{operation}", extra=log_data)
