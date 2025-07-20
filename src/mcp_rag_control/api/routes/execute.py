from fastapi import APIRouter, HTTPException
import uuid
from datetime import datetime
from typing import Dict, Any

from ...models.pipeline import PipelineExecutionRequest, PipelineExecutionResult
from ...utils.logger import get_logger
from .pipelines import pipelines_store

logger = get_logger(__name__)

router = APIRouter()


@router.post("/execute", response_model=PipelineExecutionResult)
async def execute_pipeline(request: PipelineExecutionRequest):
    if request.pipeline_id not in pipelines_store:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    pipeline = pipelines_store[request.pipeline_id]
    execution_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        result = await _execute_pipeline_steps(
            pipeline, request.query, request.parameters, request.context
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return PipelineExecutionResult(
            execution_id=execution_id,
            pipeline_id=request.pipeline_id,
            status="completed",
            query=request.query,
            result=result,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_duration=duration
        )
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return PipelineExecutionResult(
            execution_id=execution_id,
            pipeline_id=request.pipeline_id,
            status="failed",
            query=request.query,
            result={},
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_duration=duration,
            error_message=str(e)
        )


async def _execute_pipeline_steps(
    pipeline, query: str, parameters: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, Any]:
    result = {
        "query": query,
        "pipeline_name": pipeline.name,
        "response": f"Mock response for query: {query}",
        "metadata": {
            "steps_executed": len(pipeline.steps),
            "parameters": parameters,
            "context": context
        }
    }
    
    return result


@router.get("/executions/{execution_id}")
async def get_execution_result(execution_id: str):
    return {
        "message": "Execution result retrieval not implemented yet",
        "execution_id": execution_id,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/executions")
async def list_executions():
    return {
        "message": "Execution history listing not implemented yet",
        "timestamp": datetime.now().isoformat()
    }