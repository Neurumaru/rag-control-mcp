from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime

from ...models.pipeline import (
    Pipeline, PipelineStatus, PipelineRegistrationRequest, 
    PipelineUpdateRequest, PipelineResponse, PipelineListResponse
)
from ...models.request import PaginationParams, FilterParams
from ...registry.pipeline_registry import PipelineRegistry
from ...utils.logger import get_logger

logger = get_logger(__name__)
registry = PipelineRegistry()

router = APIRouter()


@router.post("/pipelines", response_model=PipelineResponse)
async def register_pipeline(request: PipelineRegistrationRequest):
    try:
        pipeline = Pipeline(
            name=request.name,
            description=request.description,
            version=request.version,
            steps=request.steps,
            category=getattr(request, 'category', 'default'),
            tags=request.tags,
            metadata=getattr(request, 'metadata', {}),
            variables=request.variables,
            config=request.config
        )
        
        # Set entry point to first step if not set
        if request.steps and not pipeline.entry_point:
            pipeline.entry_point = request.steps[0].id
        
        # Register pipeline using the registry
        await registry.register_pipeline(pipeline)
        
        logger.info(f"Pipeline '{request.name}' registered successfully with ID {pipeline.id}")
        
        return PipelineResponse(
            pipeline=pipeline,
            message=f"Pipeline '{request.name}' registered successfully"
        )
    except Exception as e:
        logger.error(f"Failed to register pipeline '{request.name}': {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/pipelines", response_model=PipelineListResponse)
async def list_pipelines(
    pagination: PaginationParams = Depends(),
    filters: FilterParams = Depends()
):
    try:
        # Get pipelines from registry with basic filtering
        status_filter = getattr(filters, 'status', None)
        if status_filter:
            status_filter = PipelineStatus(status_filter)
        
        pipelines_list = registry.list_pipelines(
            status=status_filter,
            category=getattr(filters, 'category', None)
        )
        
        # Apply search filter
        if getattr(filters, 'search', None):
            search_term = filters.search.lower()
            pipelines_list = [
                p for p in pipelines_list
                if search_term in p.name.lower() or
                   (p.description and search_term in p.description.lower())
            ]
        
        # Apply pagination
        total = len(pipelines_list)
        start = (pagination.page - 1) * pagination.page_size
        end = start + pagination.page_size
        paginated_pipelines = pipelines_list[start:end]
        
        logger.info(f"Listed {len(paginated_pipelines)} pipelines (total: {total})")
        
        return PipelineListResponse(
            pipelines=paginated_pipelines,
            total=total,
            message=f"Retrieved {len(paginated_pipelines)} pipelines"
        )
    except Exception as e:
        logger.error(f"Failed to list pipelines: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipelines/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(pipeline_id: str):
    try:
        pipeline_uuid = UUID(pipeline_id)
        pipeline = registry.get_pipeline(pipeline_uuid)
        
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        logger.info(f"Retrieved pipeline '{pipeline.name}' with ID {pipeline_id}")
        
        return PipelineResponse(
            pipeline=pipeline,
            message="Pipeline retrieved successfully"
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pipeline ID format")
    except Exception as e:
        logger.error(f"Failed to get pipeline {pipeline_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/pipelines/{pipeline_id}", response_model=PipelineResponse)
async def update_pipeline(pipeline_id: str, request: PipelineUpdateRequest):
    try:
        pipeline_uuid = UUID(pipeline_id)
        
        # Get existing pipeline
        existing_pipeline = registry.get_pipeline(pipeline_uuid)
        if not existing_pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        # Create updated pipeline with new values
        update_data = request.dict(exclude_unset=True)
        
        # Create new pipeline object with updated fields
        updated_pipeline = Pipeline(
            id=pipeline_uuid,
            name=update_data.get('name', existing_pipeline.name),
            description=update_data.get('description', existing_pipeline.description),
            version=update_data.get('version', existing_pipeline.version),
            steps=update_data.get('steps', existing_pipeline.steps),
            entry_point=update_data.get('entry_point', existing_pipeline.entry_point),
            category=update_data.get('category', existing_pipeline.category),
            status=update_data.get('status', existing_pipeline.status),
            tags=update_data.get('tags', existing_pipeline.tags),
            metadata=update_data.get('metadata', existing_pipeline.metadata),
            created_at=existing_pipeline.created_at,
            updated_at=datetime.utcnow()
        )
        
        # Update pipeline using the registry
        await registry.update_pipeline(pipeline_uuid, updated_pipeline)
        
        logger.info(f"Updated pipeline '{updated_pipeline.name}' with ID {pipeline_id}")
        
        return PipelineResponse(
            pipeline=updated_pipeline,
            message="Pipeline updated successfully"
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pipeline ID format")
    except Exception as e:
        logger.error(f"Failed to update pipeline {pipeline_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    try:
        pipeline_uuid = UUID(pipeline_id)
        
        # Get pipeline name before deletion for logging
        pipeline = registry.get_pipeline(pipeline_uuid)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        # Delete pipeline using registry
        await registry.unregister_pipeline(pipeline_uuid)
        
        logger.info(f"Deleted pipeline '{pipeline.name}' with ID {pipeline_id}")
        
        return {
            "message": f"Pipeline '{pipeline.name}' deleted successfully",
            "timestamp": datetime.now().isoformat()
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pipeline ID format")
    except Exception as e:
        logger.error(f"Failed to delete pipeline {pipeline_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipelines/{pipeline_id}/activate")
async def activate_pipeline(pipeline_id: str):
    try:
        pipeline_uuid = UUID(pipeline_id)
        
        # Activate pipeline using registry
        await registry.activate_pipeline(pipeline_uuid)
        
        # Get updated pipeline for response
        pipeline = registry.get_pipeline(pipeline_uuid)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        logger.info(f"Activated pipeline '{pipeline.name}' with ID {pipeline_id}")
        
        return {
            "message": f"Pipeline '{pipeline.name}' activated successfully",
            "timestamp": datetime.now().isoformat()
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pipeline ID format")
    except Exception as e:
        logger.error(f"Failed to activate pipeline {pipeline_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipelines/{pipeline_id}/deactivate")
async def deactivate_pipeline(pipeline_id: str):
    try:
        pipeline_uuid = UUID(pipeline_id)
        
        # Deactivate pipeline using registry
        await registry.deactivate_pipeline(pipeline_uuid)
        
        # Get updated pipeline for response
        pipeline = registry.get_pipeline(pipeline_uuid)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        logger.info(f"Deactivated pipeline '{pipeline.name}' with ID {pipeline_id}")
        
        return {
            "message": f"Pipeline '{pipeline.name}' deactivated successfully",
            "timestamp": datetime.now().isoformat()
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pipeline ID format")
    except Exception as e:
        logger.error(f"Failed to deactivate pipeline {pipeline_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Pipeline execution endpoints
@router.post("/pipelines/{pipeline_id}/execute")
async def execute_pipeline(pipeline_id: str, input_data: dict = None):
    try:
        pipeline_uuid = UUID(pipeline_id)
        
        # Create execution using registry
        execution = await registry.create_execution(pipeline_uuid, input_data)
        
        logger.info(f"Started execution {execution.id} for pipeline {pipeline_id}")
        
        return {
            "execution_id": str(execution.id),
            "pipeline_id": pipeline_id,
            "status": execution.status.value,
            "message": "Pipeline execution started",
            "timestamp": datetime.now().isoformat()
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pipeline ID format")
    except Exception as e:
        logger.error(f"Failed to execute pipeline {pipeline_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipelines/{pipeline_id}/executions")
async def get_pipeline_executions(pipeline_id: str):
    try:
        pipeline_uuid = UUID(pipeline_id)
        
        # Get executions from registry
        executions = registry.get_pipeline_executions(pipeline_uuid)
        
        logger.info(f"Retrieved {len(executions)} executions for pipeline {pipeline_id}")
        
        return {
            "executions": [
                {
                    "id": str(exec.id),
                    "status": exec.status.value,
                    "started_at": exec.started_at.isoformat(),
                    "completed_at": exec.completed_at.isoformat() if exec.completed_at else None,
                    "execution_time_seconds": exec.execution_time_seconds,
                    "error_message": exec.error_message
                }
                for exec in executions
            ],
            "total": len(executions),
            "message": f"Retrieved {len(executions)} executions"
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pipeline ID format")
    except Exception as e:
        logger.error(f"Failed to get executions for pipeline {pipeline_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))