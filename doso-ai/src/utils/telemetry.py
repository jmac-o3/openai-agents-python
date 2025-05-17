"""
Logging and telemetry configuration for DOSO AI
"""

import logging
import logging.handlers
import sys
from functools import lru_cache
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from ..config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            "doso-ai.log",
            maxBytes=10485760,  # 10MB
            backupCount=5,
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger("doso-ai")


@lru_cache
def setup_telemetry() -> None:
    """Initialize OpenTelemetry tracing"""
    # Configure trace provider
    tracer_provider = TracerProvider(
        resource=Resource.create({"service.name": settings.PROJECT_NAME}),
    )

    # Set up OTLP exporter for traces
    otlp_exporter = OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT)
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)

    # Set global trace provider
    trace.set_tracer_provider(tracer_provider)


def log_error(e: Exception, **kwargs: Any) -> None:
    """
    Log an error with additional context

    Args:
        e: Exception to log
        **kwargs: Additional context to include in log

    """
    error_context = {
        "error_type": type(e).__name__,
        "error_message": str(e),
        **kwargs,
    }
    logger.error("Error occurred", extra=error_context, exc_info=True)


def log_warning(message: str, **kwargs: Any) -> None:
    """
    Log a warning with additional context

    Args:
        message: Warning message
        **kwargs: Additional context to include in log

    """
    logger.warning(message, extra=kwargs)


def log_info(message: str, **kwargs: Any) -> None:
    """
    Log an info message with additional context

    Args:
        message: Info message
        **kwargs: Additional context to include in log

    """
    logger.info(message, extra=kwargs)
