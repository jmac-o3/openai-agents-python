"""
DOSO AI FastAPI application
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentalisation

from .config import settings
from .utils.telemetry import log_error, setup_telemetry


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler for startup and shutdown
    """
    # Startup
    setup_telemetry()
    logging.info("Starting DOSO AI application")
    
    # Initialize vector store schema
    from .db.vector_store import VectorStore
    vector_store = VectorStore()
    await vector_store.ensure_schema_exists()
    logging.info("Vector store schema initialized")
    
    yield
    
    # Shutdown
    logging.info("Shutting down DOSO AI application")


app = FastAPI(
    title="DOSO AI API",
    description="Dealer Inventory Optimization System AI API",
    version="0.1.0",
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
from .api.routers import market_analysis, orchestration, allocation_tracking

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
app.include_router(
    orchestration.router, 
    prefix="/api/v1/orchestration", 
    tags=["Workflow Orchestration"]
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
    return {"status": "healthy", "version": "0.1.0"}
