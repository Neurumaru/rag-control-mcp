from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
import uuid
from datetime import datetime

from ...models.pipeline import (
    Pipeline, PipelineStatus, PipelineRegistrationRequest, 
    PipelineUpdateRequest, PipelineResponse, PipelineListResponse
)
from ...models.request import PaginationParams, FilterParams
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

pipelines_store: dict = {}


@router.post("/pipelines", response_model=PipelineResponse)
async def register_pipeline(request: PipelineRegistrationRequest):
    pipeline_id = str(uuid.uuid4())
    
    pipeline = Pipeline(
        id=pipeline_id,
        name=request.name,
        description=request.description,
        version=request.version,
        steps=request.steps,
        entry_point=request.entry_point,
        metadata=request.metadata,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    
    pipelines_store[pipeline_id] = pipeline
    
    return PipelineResponse(
        pipeline=pipeline,
        message=f"Pipeline '{request.name}' registered successfully"
    )


@router.get("/pipelines", response_model=PipelineListResponse)
async def list_pipelines(
    pagination: PaginationParams = Depends(),
    filters: FilterParams = Depends()
):
    pipelines_list = list(pipelines_store.values())
    
    if filters.status:
        pipelines_list = [p for p in pipelines_list if p.status == filters.status]
    
    if filters.search:
        search_term = filters.search.lower()
        pipelines_list = [
            p for p in pipelines_list
            if search_term in p.name.lower() or
               (p.description and search_term in p.description.lower())
        ]
    
    total = len(pipelines_list)
    start = (pagination.page - 1) * pagination.page_size
    end = start + pagination.page_size
    paginated_pipelines = pipelines_list[start:end]
    
    return PipelineListResponse(
        pipelines=paginated_pipelines,
        total=total,
        message=f"Retrieved {len(paginated_pipelines)} pipelines"
    )


@router.get("/pipelines/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(pipeline_id: str):
    if pipeline_id not in pipelines_store:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    pipeline = pipelines_store[pipeline_id]
    return PipelineResponse(
        pipeline=pipeline,
        message="Pipeline retrieved successfully"
    )


@router.put("/pipelines/{pipeline_id}", response_model=PipelineResponse)
async def update_pipeline(pipeline_id: str, request: PipelineUpdateRequest):
    if pipeline_id not in pipelines_store:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    pipeline = pipelines_store[pipeline_id]
    
    update_data = request.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(pipeline, field, value)
    
    pipeline.updated_at = datetime.now().isoformat()
    pipelines_store[pipeline_id] = pipeline
    
    return PipelineResponse(
        pipeline=pipeline,
        message="Pipeline updated successfully"
    )


@router.delete("/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    if pipeline_id not in pipelines_store:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    pipeline = pipelines_store.pop(pipeline_id)
    
    return {
        "message": f"Pipeline '{pipeline.name}' deleted successfully",
        "timestamp": datetime.now().isoformat()
    }


@router.post("/pipelines/{pipeline_id}/activate")
async def activate_pipeline(pipeline_id: str):
    if pipeline_id not in pipelines_store:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    pipeline = pipelines_store[pipeline_id]
    pipeline.status = PipelineStatus.ACTIVE
    pipeline.updated_at = datetime.now().isoformat()
    
    return {
        "message": f"Pipeline '{pipeline.name}' activated successfully",
        "timestamp": datetime.now().isoformat()
    }


@router.post("/pipelines/{pipeline_id}/deactivate")
async def deactivate_pipeline(pipeline_id: str):
    if pipeline_id not in pipelines_store:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    pipeline = pipelines_store[pipeline_id]
    pipeline.status = PipelineStatus.INACTIVE
    pipeline.updated_at = datetime.now().isoformat()
    
    return {
        "message": f"Pipeline '{pipeline.name}' deactivated successfully",
        "timestamp": datetime.now().isoformat()
    }