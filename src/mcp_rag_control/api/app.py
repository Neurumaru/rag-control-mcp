from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from datetime import datetime
from typing import Dict, Any

from ..models.request import HealthCheckResponse, ErrorResponse
from ..utils.logger import get_logger
from ..registry.module_registry import ModuleRegistry
from .routes import modules, pipelines, execute
from .middleware.security import (
    RateLimitMiddleware, SecurityHeadersMiddleware, 
    RequestLoggingMiddleware, RequestSizeMiddleware
)

logger = get_logger(__name__)
registry = ModuleRegistry()

app = FastAPI(
    title="MCP-RAG-Control API",
    description="Map-based control architecture for agent-based RAG systems",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add security middleware
app.add_middleware(RequestSizeMiddleware, max_size=10 * 1024 * 1024)  # 10MB limit
app.add_middleware(RateLimitMiddleware, calls_per_minute=60, burst_limit=10)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

start_time = time.time()

app.include_router(modules.router, prefix="/api/v1", tags=["modules"])
app.include_router(pipelines.router, prefix="/api/v1", tags=["pipelines"])
app.include_router(execute.router, prefix="/api/v1", tags=["execution"])


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            message=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.get("/", response_model=Dict[str, Any])
async def root():
    return {
        "message": "MCP-RAG-Control API",
        "version": "0.1.0",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    uptime = time.time() - start_time
    
    # Get registry statistics
    stats = registry.get_stats()
    
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="0.1.0",
        uptime=uptime,
        modules_count=stats["total_modules"],
        pipelines_count=0  # Will be implemented when pipeline registry is updated
    )