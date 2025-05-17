"""
DOSO AI FastAPI application using openai-agents SDK with Assistants

This version of the main application uses the openai-agents SDK's built-in
support for OpenAI Assistants, replacing PostgreSQL and Redis storage.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentalisation

from .config import settings
from .utils.telemetry import log_error, setup_telemetry
from .agents.assistants_storage import assistants_storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler for startup and shutdown
    """
    # Startup
    setup_telemetry()
    logging.info("Starting DOSO AI application (OpenAI Assistants SDK version)")
    
    # Initialize Assistants Storage agent
    try:
        # Run a simple query to initialize the agent and its assistant
        logging.info("Initializing Assistants Storage agent...")
        await assistants_storage.search_documents("initialization")
        logging.info("Assistants Storage agent initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing Assistants Storage agent: {str(e)}")
        # Continue anyway, as we might be able to initialize it later
    
    yield
    
    # Shutdown
    logging.info("Shutting down DOSO AI application")


app = FastAPI(
    title="DOSO AI API (openai-agents Assistants SDK)",
    description="Dealer Inventory Optimization System AI API using openai-agents SDK",
    version="0.3.0",
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
from .api.routers import market_analysis, allocation_tracking, assistants_router

# Import and register the Assistants-based orchestration router
app.include_router(
    assistants_router.router,
    prefix="/api/v1/orchestration",
    tags=["Workflow Orchestration (Assistants SDK)"],
)

# Include other routers but make them use the assistants-based implementation
# under the hood (if needed)
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
        "version": "0.3.0",
        "storage_type": "openai_assistants_sdk"
    }


@app.get("/assistant-info")
async def assistant_info():
    """Get information about the OpenAI Assistant"""
    try:
        # Run a simple search to check connectivity
        results = await assistants_storage.search_documents("metadata")
        
        return {
            "status": "active",
            "storage_type": "openai_assistants_sdk",
            "connected": True,
            "agent_name": "Assistants Storage Agent",
            "result_count": len(results)
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
