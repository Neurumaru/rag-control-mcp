"""Pipeline schema definitions for MCP-RAG-Control system."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class PipelineStatus(str, Enum):
    """Status of a pipeline."""
    
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    RUNNING = "running"
    COMPLETED = "completed"


class StepType(str, Enum):
    """Types of pipeline steps."""
    
    INPUT = "input"
    RETRIEVAL = "retrieval"
    PROCESSING = "processing"
    GENERATION = "generation"
    OUTPUT = "output"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    PARALLEL = "parallel"


class PipelineStep(BaseModel):
    """Individual step in a pipeline."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Step name")
    step_type: StepType = Field(..., description="Type of step")
    module_id: UUID = Field(..., description="Module to execute this step")
    
    # Step configuration
    config: Dict[str, Any] = Field(default_factory=dict)
    input_mapping: Dict[str, str] = Field(default_factory=dict)
    output_mapping: Dict[str, str] = Field(default_factory=dict)
    
    # Flow control
    next_steps: List[UUID] = Field(default_factory=list)
    condition: Optional[str] = Field(None, description="Condition for conditional steps")
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    timeout_seconds: Optional[float] = Field(None)
    
    # Execution metadata
    execution_order: int = Field(default=0)
    is_parallel: bool = Field(default=False)
    depends_on: List[UUID] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            UUID: lambda v: str(v)
        }


class Pipeline(BaseModel):
    """Pipeline definition for orchestrating RAG workflows."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Pipeline name")
    description: Optional[str] = Field(None, description="Pipeline description")
    version: str = Field(default="1.0.0", description="Pipeline version")
    
    # Pipeline structure
    steps: List[PipelineStep] = Field(default_factory=list)
    entry_point: Optional[UUID] = Field(None, description="First step to execute")
    
    # Configuration
    config: Dict[str, Any] = Field(default_factory=dict)
    variables: Dict[str, Any] = Field(default_factory=dict)
    
    # Status and metadata
    status: PipelineStatus = Field(default=PipelineStatus.DRAFT)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Execution settings
    max_execution_time: Optional[float] = Field(None, description="Max execution time in seconds")
    enable_caching: bool = Field(default=True)
    enable_monitoring: bool = Field(default=True)
    
    # Categorization
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = Field(None)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    
    def add_step(self, step: PipelineStep) -> None:
        """Add a step to the pipeline."""
        step.execution_order = len(self.steps)
        self.steps.append(step)
        self.updated_at = datetime.utcnow()
        
        if self.entry_point is None:
            self.entry_point = step.id
    
    def remove_step(self, step_id: UUID) -> None:
        """Remove a step from the pipeline."""
        self.steps = [s for s in self.steps if s.id != step_id]
        
        # Update execution order
        for i, step in enumerate(self.steps):
            step.execution_order = i
        
        # Update entry point if necessary
        if self.entry_point == step_id:
            self.entry_point = self.steps[0].id if self.steps else None
        
        self.updated_at = datetime.utcnow()
    
    def get_step(self, step_id: UUID) -> Optional[PipelineStep]:
        """Get a step by ID."""
        return next((s for s in self.steps if s.id == step_id), None)
    
    def update_status(self, status: PipelineStatus) -> None:
        """Update pipeline status."""
        self.status = status
        self.updated_at = datetime.utcnow()
    
    def validate_structure(self) -> List[str]:
        """Validate pipeline structure and return any errors."""
        errors = []
        
        if not self.steps:
            errors.append("Pipeline must have at least one step")
            return errors
        
        if self.entry_point is None:
            errors.append("Pipeline must have an entry point")
        
        step_ids = {step.id for step in self.steps}
        
        for step in self.steps:
            # Check next_steps references
            for next_id in step.next_steps:
                if next_id not in step_ids:
                    errors.append(f"Step {step.name} references non-existent next step {next_id}")
            
            # Check dependencies
            for dep_id in step.depends_on:
                if dep_id not in step_ids:
                    errors.append(f"Step {step.name} depends on non-existent step {dep_id}")
        
        return errors


class PipelineExecution(BaseModel):
    """Execution instance of a pipeline."""
    
    id: UUID = Field(default_factory=uuid4)
    pipeline_id: UUID = Field(..., description="Pipeline being executed")
    pipeline_version: str = Field(..., description="Version of pipeline executed")
    
    # Execution state
    status: PipelineStatus = Field(default=PipelineStatus.RUNNING)
    current_step_id: Optional[UUID] = Field(None)
    completed_steps: List[UUID] = Field(default_factory=list)
    failed_steps: List[UUID] = Field(default_factory=list)
    
    # Input/Output
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    step_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)
    execution_time_seconds: Optional[float] = Field(None)
    
    # Error handling
    error_message: Optional[str] = Field(None)
    error_step_id: Optional[UUID] = Field(None)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    
    def mark_step_completed(self, step_id: UUID, output: Dict[str, Any]) -> None:
        """Mark a step as completed with its output."""
        if step_id not in self.completed_steps:
            self.completed_steps.append(step_id)
        
        self.step_outputs[str(step_id)] = output
    
    def mark_step_failed(self, step_id: UUID, error: str) -> None:
        """Mark a step as failed."""
        if step_id not in self.failed_steps:
            self.failed_steps.append(step_id)
        
        self.error_step_id = step_id
        self.error_message = error
        self.status = PipelineStatus.ERROR
    
    def complete_execution(self, output: Dict[str, Any]) -> None:
        """Mark execution as completed."""
        self.status = PipelineStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.output_data = output
        
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            self.execution_time_seconds = delta.total_seconds()


# API Request/Response Models
class PipelineRegistrationRequest(BaseModel):
    """Request model for registering a new pipeline."""
    
    name: str = Field(..., description="Human-readable pipeline name")
    description: Optional[str] = Field(None, description="Pipeline description")
    version: str = Field(default="1.0.0", description="Pipeline version")
    steps: List[PipelineStep] = Field(..., description="Pipeline steps")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Pipeline variables")
    config: Dict[str, Any] = Field(default_factory=dict, description="Pipeline configuration")
    tags: List[str] = Field(default_factory=list, description="Pipeline tags")


class PipelineUpdateRequest(BaseModel):
    """Request model for updating an existing pipeline."""
    
    name: Optional[str] = Field(None, description="Human-readable pipeline name")
    description: Optional[str] = Field(None, description="Pipeline description")
    version: Optional[str] = Field(None, description="Pipeline version")
    status: Optional[PipelineStatus] = Field(None, description="Pipeline status")
    steps: Optional[List[PipelineStep]] = Field(None, description="Pipeline steps")
    variables: Optional[Dict[str, Any]] = Field(None, description="Pipeline variables")
    config: Optional[Dict[str, Any]] = Field(None, description="Pipeline configuration")
    tags: Optional[List[str]] = Field(None, description="Pipeline tags")


class PipelineExecutionRequest(BaseModel):
    """Request model for executing a pipeline."""
    
    pipeline_id: str = Field(..., description="ID of the pipeline to execute")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for pipeline")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Runtime variables")
    async_execution: bool = Field(default=False, description="Execute asynchronously")


class PipelineExecutionResult(BaseModel):
    """Result model for pipeline execution."""
    
    execution_id: str = Field(..., description="Unique execution identifier")
    pipeline_id: str = Field(..., description="ID of the executed pipeline")
    status: str = Field(..., description="Execution status")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="Output data")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    started_at: Optional[str] = Field(None, description="Execution start time")
    completed_at: Optional[str] = Field(None, description="Execution completion time")
    execution_time_seconds: Optional[float] = Field(None, description="Total execution time")


class PipelineResponse(BaseModel):
    """Response model for pipeline operations."""
    
    pipeline: Pipeline
    message: str = Field(default="Operation completed successfully")


class PipelineListResponse(BaseModel):
    """Response model for listing pipelines."""
    
    pipelines: List[Pipeline]
    total: int
    message: str = Field(default="Pipelines retrieved successfully")