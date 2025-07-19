"""Pipeline registry for managing RAG pipelines."""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Set
from uuid import UUID

from ..models.pipeline import Pipeline, PipelineStatus, PipelineExecution


class PipelineRegistryError(Exception):
    """Base exception for pipeline registry errors."""
    pass


class PipelineNotFoundError(PipelineRegistryError):
    """Error when pipeline is not found."""
    pass


class PipelineValidationError(PipelineRegistryError):
    """Error when pipeline validation fails."""
    pass


class ExecutionNotFoundError(PipelineRegistryError):
    """Error when execution is not found."""
    pass


class PipelineRegistry:
    """Registry for managing RAG pipelines."""
    
    def __init__(self):
        """Initialize pipeline registry."""
        self._pipelines: Dict[UUID, Pipeline] = {}
        self._executions: Dict[UUID, PipelineExecution] = {}
        self._pipeline_executions: Dict[UUID, Set[UUID]] = {}  # pipeline_id -> execution_ids
        self._lock = asyncio.Lock()
    
    async def register_pipeline(self, pipeline: Pipeline) -> None:
        """Register a new pipeline."""
        async with self._lock:
            if pipeline.id in self._pipelines:
                raise PipelineRegistryError(f"Pipeline {pipeline.id} already registered")
            
            # Validate pipeline structure
            errors = pipeline.validate_structure()
            if errors:
                raise PipelineValidationError(f"Pipeline validation failed: {errors}")
            
            # Store pipeline
            self._pipelines[pipeline.id] = pipeline
            self._pipeline_executions[pipeline.id] = set()
            
            # Mark as active if it was draft
            if pipeline.status == PipelineStatus.DRAFT:
                pipeline.update_status(PipelineStatus.ACTIVE)
    
    async def unregister_pipeline(self, pipeline_id: UUID, force: bool = False) -> None:
        """Unregister a pipeline."""
        async with self._lock:
            if pipeline_id not in self._pipelines:
                raise PipelineNotFoundError(f"Pipeline {pipeline_id} not found")
            
            # Check for running executions
            running_executions = [
                exec_id for exec_id in self._pipeline_executions.get(pipeline_id, set())
                if self._executions.get(exec_id, {}).status == PipelineStatus.RUNNING
            ]
            
            if running_executions and not force:
                raise PipelineRegistryError(
                    f"Pipeline {pipeline_id} has running executions: {running_executions}. "
                    "Use force=True to remove anyway."
                )
            
            # Remove executions if forced
            if force:
                execution_ids = self._pipeline_executions.get(pipeline_id, set()).copy()
                for exec_id in execution_ids:
                    if exec_id in self._executions:
                        del self._executions[exec_id]
            
            # Remove pipeline
            del self._pipelines[pipeline_id]
            del self._pipeline_executions[pipeline_id]
    
    def get_pipeline(self, pipeline_id: UUID) -> Optional[Pipeline]:
        """Get pipeline by ID."""
        return self._pipelines.get(pipeline_id)
    
    def list_pipelines(
        self,
        status: Optional[PipelineStatus] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Pipeline]:
        """List pipelines with optional filtering."""
        pipelines = list(self._pipelines.values())
        
        if status:
            pipelines = [p for p in pipelines if p.status == status]
        
        if category:
            pipelines = [p for p in pipelines if p.category == category]
        
        if tags:
            pipelines = [p for p in pipelines if any(tag in p.tags for tag in tags)]
        
        return pipelines
    
    def find_pipelines_by_name(self, name: str, exact_match: bool = False) -> List[Pipeline]:
        """Find pipelines by name."""
        if exact_match:
            return [p for p in self._pipelines.values() if p.name == name]
        else:
            return [p for p in self._pipelines.values() if name.lower() in p.name.lower()]
    
    async def update_pipeline(self, pipeline_id: UUID, pipeline: Pipeline) -> None:
        """Update an existing pipeline."""
        async with self._lock:
            if pipeline_id not in self._pipelines:
                raise PipelineNotFoundError(f"Pipeline {pipeline_id} not found")
            
            # Validate new pipeline structure
            errors = pipeline.validate_structure()
            if errors:
                raise PipelineValidationError(f"Pipeline validation failed: {errors}")
            
            # Ensure ID matches
            pipeline.id = pipeline_id
            pipeline.updated_at = datetime.utcnow()
            
            # Update pipeline
            self._pipelines[pipeline_id] = pipeline
    
    async def activate_pipeline(self, pipeline_id: UUID) -> None:
        """Activate a pipeline."""
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            raise PipelineNotFoundError(f"Pipeline {pipeline_id} not found")
        
        pipeline.update_status(PipelineStatus.ACTIVE)
    
    async def deactivate_pipeline(self, pipeline_id: UUID) -> None:
        """Deactivate a pipeline."""
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            raise PipelineNotFoundError(f"Pipeline {pipeline_id} not found")
        
        pipeline.update_status(PipelineStatus.INACTIVE)
    
    async def create_execution(self, pipeline_id: UUID, input_data: Dict = None) -> PipelineExecution:
        """Create a new pipeline execution."""
        async with self._lock:
            pipeline = self.get_pipeline(pipeline_id)
            if not pipeline:
                raise PipelineNotFoundError(f"Pipeline {pipeline_id} not found")
            
            if pipeline.status != PipelineStatus.ACTIVE:
                raise PipelineRegistryError(f"Pipeline {pipeline_id} is not active")
            
            execution = PipelineExecution(
                pipeline_id=pipeline_id,
                pipeline_version=pipeline.version,
                input_data=input_data or {},
                status=PipelineStatus.RUNNING
            )
            
            # Store execution
            self._executions[execution.id] = execution
            self._pipeline_executions[pipeline_id].add(execution.id)
            
            return execution
    
    def get_execution(self, execution_id: UUID) -> Optional[PipelineExecution]:
        """Get execution by ID."""
        return self._executions.get(execution_id)
    
    def list_executions(
        self,
        pipeline_id: Optional[UUID] = None,
        status: Optional[PipelineStatus] = None,
        limit: Optional[int] = None
    ) -> List[PipelineExecution]:
        """List executions with optional filtering."""
        executions = list(self._executions.values())
        
        if pipeline_id:
            executions = [e for e in executions if e.pipeline_id == pipeline_id]
        
        if status:
            executions = [e for e in executions if e.status == status]
        
        # Sort by start time (newest first)
        executions.sort(key=lambda x: x.started_at, reverse=True)
        
        if limit:
            executions = executions[:limit]
        
        return executions
    
    async def update_execution(self, execution_id: UUID, **updates) -> None:
        """Update execution status and data."""
        async with self._lock:
            execution = self.get_execution(execution_id)
            if not execution:
                raise ExecutionNotFoundError(f"Execution {execution_id} not found")
            
            for key, value in updates.items():
                if hasattr(execution, key):
                    setattr(execution, key, value)
    
    async def complete_execution(
        self, 
        execution_id: UUID, 
        output_data: Dict = None,
        status: PipelineStatus = PipelineStatus.COMPLETED
    ) -> None:
        """Mark execution as completed."""
        execution = self.get_execution(execution_id)
        if not execution:
            raise ExecutionNotFoundError(f"Execution {execution_id} not found")
        
        execution.complete_execution(output_data or {})
        execution.status = status
    
    async def fail_execution(self, execution_id: UUID, error_message: str, error_step_id: UUID = None) -> None:
        """Mark execution as failed."""
        execution = self.get_execution(execution_id)
        if not execution:
            raise ExecutionNotFoundError(f"Execution {execution_id} not found")
        
        execution.status = PipelineStatus.ERROR
        execution.error_message = error_message
        execution.error_step_id = error_step_id
        execution.completed_at = datetime.utcnow()
        
        if execution.started_at and execution.completed_at:
            delta = execution.completed_at - execution.started_at
            execution.execution_time_seconds = delta.total_seconds()
    
    def get_pipeline_executions(self, pipeline_id: UUID) -> List[PipelineExecution]:
        """Get all executions for a pipeline."""
        execution_ids = self._pipeline_executions.get(pipeline_id, set())
        executions = [self._executions[eid] for eid in execution_ids if eid in self._executions]
        
        # Sort by start time (newest first)
        executions.sort(key=lambda x: x.started_at, reverse=True)
        
        return executions
    
    def get_running_executions(self) -> List[PipelineExecution]:
        """Get all currently running executions."""
        return [e for e in self._executions.values() if e.status == PipelineStatus.RUNNING]
    
    def get_execution_stats(self, pipeline_id: Optional[UUID] = None) -> Dict[str, any]:
        """Get execution statistics."""
        executions = list(self._executions.values())
        
        if pipeline_id:
            executions = [e for e in executions if e.pipeline_id == pipeline_id]
        
        total_executions = len(executions)
        completed_executions = len([e for e in executions if e.status == PipelineStatus.COMPLETED])
        failed_executions = len([e for e in executions if e.status == PipelineStatus.ERROR])
        running_executions = len([e for e in executions if e.status == PipelineStatus.RUNNING])
        
        # Calculate average execution time for completed executions
        completed_with_time = [
            e for e in executions 
            if e.status == PipelineStatus.COMPLETED and e.execution_time_seconds is not None
        ]
        
        avg_execution_time = None
        if completed_with_time:
            avg_execution_time = sum(e.execution_time_seconds for e in completed_with_time) / len(completed_with_time)
        
        return {
            "total_executions": total_executions,
            "completed_executions": completed_executions,
            "failed_executions": failed_executions,
            "running_executions": running_executions,
            "success_rate": completed_executions / total_executions if total_executions > 0 else 0,
            "average_execution_time_seconds": avg_execution_time,
        }
    
    def get_pipeline_stats(self) -> Dict[str, any]:
        """Get pipeline statistics."""
        total_pipelines = len(self._pipelines)
        active_pipelines = len([p for p in self._pipelines.values() if p.status == PipelineStatus.ACTIVE])
        inactive_pipelines = len([p for p in self._pipelines.values() if p.status == PipelineStatus.INACTIVE])
        error_pipelines = len([p for p in self._pipelines.values() if p.status == PipelineStatus.ERROR])
        
        # Get pipeline categories
        categories = {}
        for pipeline in self._pipelines.values():
            if pipeline.category:
                categories[pipeline.category] = categories.get(pipeline.category, 0) + 1
        
        # Get total steps across all pipelines
        total_steps = sum(len(p.steps) for p in self._pipelines.values())
        
        return {
            "total_pipelines": total_pipelines,
            "active_pipelines": active_pipelines,
            "inactive_pipelines": inactive_pipelines,
            "error_pipelines": error_pipelines,
            "categories": categories,
            "total_steps": total_steps,
            "average_steps_per_pipeline": total_steps / total_pipelines if total_pipelines > 0 else 0,
        }
    
    async def cleanup_old_executions(self, max_age_days: int = 30, max_count: int = 1000) -> int:
        """Cleanup old executions."""
        async with self._lock:
            cutoff_date = datetime.utcnow().timestamp() - (max_age_days * 24 * 60 * 60)
            
            # Find executions to remove
            to_remove = []
            
            for execution in self._executions.values():
                # Remove if too old or if we have too many
                if (execution.started_at.timestamp() < cutoff_date or 
                    len(self._executions) > max_count):
                    # Don't remove running executions
                    if execution.status != PipelineStatus.RUNNING:
                        to_remove.append(execution.id)
            
            # Remove executions
            removed_count = 0
            for exec_id in to_remove:
                if exec_id in self._executions:
                    execution = self._executions[exec_id]
                    pipeline_id = execution.pipeline_id
                    
                    del self._executions[exec_id]
                    
                    if pipeline_id in self._pipeline_executions:
                        self._pipeline_executions[pipeline_id].discard(exec_id)
                    
                    removed_count += 1
            
            return removed_count
    
    def validate_all_pipelines(self) -> Dict[UUID, List[str]]:
        """Validate all registered pipelines."""
        validation_results = {}
        
        for pipeline_id, pipeline in self._pipelines.items():
            errors = pipeline.validate_structure()
            if errors:
                validation_results[pipeline_id] = errors
        
        return validation_results