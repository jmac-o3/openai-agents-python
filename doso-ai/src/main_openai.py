"""
DOSO AI FastAPI application using OpenAI Assistants API

This version of the main application uses OpenAI's Assistants API
for file storage, vector search, and agent execution instead of
PostgreSQL and Redis.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentalisation

from .config import settings
from .utils.telemetry import log_error, setup_telemetry
from .utils.upload_to_vector_store import file_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler for startup and shutdown
    """
    # Startup
    setup_telemetry()
    logging.info("Starting DOSO AI application (OpenAI Assistants version)")
    
    # Initialize OpenAI Assistant
    try:
        assistant_id = file_store.ensure_assistant_exists()
        logging.info(f"OpenAI Assistant initialized with ID: {assistant_id}")
    except Exception as e:
        logging.error(f"Error initializing OpenAI Assistant: {str(e)}")
        # Continue anyway, as we might be able to initialize it later
    
    yield
    
    # Shutdown
    logging.info("Shutting down DOSO AI application")


app = FastAPI(
    title="DOSO AI API (OpenAI Assistants)",
    description="Dealer Inventory Optimization System AI API using OpenAI Assistants",
    version="0.2.0",
    lifespan=lifespan,
    debug=settings.DEBUG,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure OpenTelemetry
FastAPIInstrumentalisation().instrument_app(app)

# Import and register routers
from .api.routers import market_analysis, orchestration_openai, allocation_tracking

# Import and register the OpenAI-based orchestration router
app.include_router(
    orchestration_openai.router,
    prefix="/api/v1/orchestration",
    tags=["Workflow Orchestration (OpenAI)"],
)

# Include the original router with a different prefix for backward compatibility
from .api.routers import orchestration
app.include_router(
    orchestration.router, 
    prefix="/api/v1/orchestration-legacy", 
    tags=["Workflow Orchestration (Legacy)"]
)

app.include_router(
    market_analysis.router,
    prefix="/api/v1/market-analysis",
    tags=["Market Analysis"],
)
app.include_router(
    allocation_tracking.router, 
    prefix="/api/v1/allocation", 
    tags=["Allocation Tracking"]
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    log_error(exc, path=request.url.path, method=request.method)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred",
            "error_type": type(exc).__name__,
        },
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "version": "0.2.0",
        "storage_type": "openai_assistants"
    }


@app.get("/assistant-info")
async def assistant_info():
    """Get information about the OpenAI Assistant"""
    try:
        assistant_id = file_store.ensure_assistant_exists()
        files = file_store.list_assistant_files()
        return {
            "assistant_id": assistant_id,
            "assistant_name": "DOSO AI Assistant",
            "file_count": len(files),
            "status": "active"
        }
    except Exception as e:
        log_error(e, operation="get_assistant_info")
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Error retrieving assistant info: {str(e)}",
                "error_type": type(e).__name__,
            },
        )
