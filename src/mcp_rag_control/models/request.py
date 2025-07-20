"""Request and response schema definitions for MCP-RAG-Control API."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class RequestType(str, Enum):
    """Types of requests."""
    
    EXECUTE_PIPELINE = "execute_pipeline"
    HEALTH_CHECK = "health_check"
    MODULE_OPERATION = "module_operation"
    SYSTEM_INFO = "system_info"


class ResponseStatus(str, Enum):
    """Response status codes."""
    
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    TIMEOUT = "timeout"


class RequestSchema(BaseModel):
    """Base request schema."""
    
    id: UUID = Field(default_factory=uuid4)
    request_type: RequestType = Field(..., description="Type of request")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    client_id: Optional[str] = Field(None, description="Client identifier")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracking")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class ResponseSchema(BaseModel):
    """Base response schema."""
    
    id: UUID = Field(default_factory=uuid4)
    request_id: UUID = Field(..., description="ID of the original request")
    status: ResponseStatus = Field(..., description="Response status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    
    # Response data
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error_message: Optional[str] = Field(None, description="Error message if status is error")
    error_code: Optional[str] = Field(None, description="Error code for programmatic handling")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class ExecuteRequest(RequestSchema):
    """Request to execute a pipeline."""
    
    request_type: RequestType = Field(default=RequestType.EXECUTE_PIPELINE)
    pipeline_id: UUID = Field(..., description="Pipeline to execute")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for pipeline")
    
    # Execution options
    async_execution: bool = Field(default=False, description="Whether to execute asynchronously")
    enable_caching: Optional[bool] = Field(None, description="Override pipeline caching setting")
    timeout_seconds: Optional[float] = Field(None, description="Execution timeout")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Runtime variables")


class ExecuteResponse(ResponseSchema):
    """Response from pipeline execution."""
    
    execution_id: UUID = Field(..., description="Execution instance ID")
    pipeline_id: UUID = Field(..., description="Pipeline that was executed")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Pipeline output data")
    
    # Execution metadata
    execution_time_seconds: Optional[float] = Field(None)
    steps_completed: int = Field(default=0)
    steps_total: int = Field(default=0)
    cache_hits: int = Field(default=0)


class ModuleOperationRequest(RequestSchema):
    """Request to perform operation on a module."""
    
    request_type: RequestType = Field(default=RequestType.MODULE_OPERATION)
    module_id: UUID = Field(..., description="Module to operate on")
    operation: str = Field(..., description="Operation to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")
    
    # Options
    timeout_seconds: Optional[float] = Field(None)
    retry_count: int = Field(default=0)


class ModuleOperationResponse(ResponseSchema):
    """Response from module operation."""
    
    module_id: UUID = Field(..., description="Module that was operated on")
    operation: str = Field(..., description="Operation that was performed")
    result: Optional[Dict[str, Any]] = Field(None, description="Operation result")


class HealthCheckRequest(RequestSchema):
    """Request for health check."""
    
    request_type: RequestType = Field(default=RequestType.HEALTH_CHECK)
    target_type: str = Field(default="system", description="Type of target to check")
    target_id: Optional[UUID] = Field(None, description="Specific target ID (module, pipeline)")
    include_details: bool = Field(default=False, description="Include detailed health info")


class HealthCheckResponse(ResponseSchema):
    """Response from health check."""
    
    target_type: str = Field(..., description="Type of target checked")
    target_id: Optional[UUID] = Field(None)
    
    # Health status
    is_healthy: bool = Field(..., description="Overall health status")
    uptime_seconds: Optional[float] = Field(None)
    
    # Component health
    components: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Health status of individual components"
    )
    
    # Metrics
    metrics: Dict[str, Any] = Field(
        default_factory=dict, 
        description="System metrics"
    )


class SystemInfoRequest(RequestSchema):
    """Request for system information."""
    
    request_type: RequestType = Field(default=RequestType.SYSTEM_INFO)
    include_modules: bool = Field(default=True)
    include_pipelines: bool = Field(default=True) 
    include_metrics: bool = Field(default=False)


class SystemInfoResponse(ResponseSchema):
    """Response with system information."""
    
    # System info
    version: str = Field(..., description="System version")
    uptime_seconds: float = Field(..., description="System uptime")
    
    # Counts
    total_modules: int = Field(default=0)
    active_modules: int = Field(default=0)
    total_pipelines: int = Field(default=0)
    active_pipelines: int = Field(default=0)
    running_executions: int = Field(default=0)
    
    # Optional detailed info
    modules: Optional[List[Dict[str, Any]]] = Field(None)
    pipelines: Optional[List[Dict[str, Any]]] = Field(None)
    metrics: Optional[Dict[str, Any]] = Field(None)


class StreamResponse(BaseModel):
    """Response for streaming operations."""
    
    id: UUID = Field(default_factory=uuid4)
    execution_id: UUID = Field(..., description="Execution being streamed")
    sequence_number: int = Field(..., description="Sequence number for ordering")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Stream data
    event_type: str = Field(..., description="Type of stream event")
    data: Dict[str, Any] = Field(default_factory=dict)
    is_final: bool = Field(default=False, description="Whether this is the final message")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


# Additional API Models
class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")


class SuccessResponse(BaseModel):
    """Success response model."""
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    timestamp: str = Field(..., description="Response timestamp")


class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=10, ge=1, le=100, description="Number of items per page")


class FilterParams(BaseModel):
    """Filter parameters."""
    module_type: Optional[str] = Field(None, description="Filter by module type")
    status: Optional[str] = Field(None, description="Filter by status")
    search: Optional[str] = Field(None, description="Search term")


class SortParams(BaseModel):
    """Sort parameters."""
    sort_by: str = Field(default="created_at", description="Field to sort by")
    sort_order: str = Field(default="desc", description="Sort order (asc/desc)")


# Simple health check response for basic API endpoint
class HealthCheckResponse(BaseModel):
    """Simple health check response."""
    status: str = Field(default="healthy", description="System health status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(default="0.1.0", description="Application version")
    uptime: float = Field(..., description="System uptime in seconds")
    modules_count: int = Field(default=0, description="Number of registered modules")
    pipelines_count: int = Field(default=0, description="Number of registered pipelines")