from fastapi import APIRouter, HTTPException
from uuid import UUID, uuid4
from datetime import datetime
from typing import Dict, Any

from ...models.pipeline import PipelineExecutionRequest, PipelineExecutionResult, PipelineStatus
from ...models.request import QueryRequest, QueryResponse
from ...utils.logger import get_logger
from .pipelines import registry as pipeline_registry

logger = get_logger(__name__)

router = APIRouter()


@router.post("/execute", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    try:
        # Determine pipeline to use
        pipeline = None
        if request.pipeline_id:
            # Use specific pipeline
            try:
                pipeline_uuid = UUID(request.pipeline_id)
                pipeline = pipeline_registry.get_pipeline(pipeline_uuid)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid pipeline ID format")
        else:
            # Use first active pipeline
            active_pipelines = pipeline_registry.list_pipelines(status=PipelineStatus.ACTIVE)
            if not active_pipelines:
                raise HTTPException(status_code=400, detail="No active pipelines available")
            pipeline = active_pipelines[0]
        
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        # Create execution
        execution = await pipeline_registry.create_execution(
            pipeline.id, 
            {"query": request.query, "context": request.context, "parameters": request.parameters}
        )
        
        try:
            # Execute pipeline steps (mock implementation for now)
            result = await _execute_pipeline_steps(
                pipeline, request.query, request.parameters, request.context
            )
            
            # Complete execution
            await pipeline_registry.complete_execution(execution.id, result)
            
            # Get updated execution
            updated_execution = pipeline_registry.get_execution(execution.id)
            
            logger.info(f"Successfully executed pipeline '{pipeline.name}' with execution ID {execution.id}")
            
            return QueryResponse(
                query=request.query,
                response=result.get("response", "No response generated"),
                pipeline_id=str(pipeline.id),
                execution_id=str(execution.id),
                metadata={
                    "pipeline_name": pipeline.name,
                    "steps_count": len(pipeline.steps),
                    "execution_time": updated_execution.execution_time_seconds,
                    "timestamp": datetime.now().isoformat()
                },
                sources=result.get("sources", []),
                execution_time=updated_execution.execution_time_seconds
            )
            
        except Exception as e:
            # Fail execution
            await pipeline_registry.fail_execution(execution.id, str(e))
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _execute_pipeline_steps(
    pipeline, query: str, parameters: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Mock implementation of pipeline execution.
    In a real implementation, this would:
    1. Execute each step in the pipeline
    2. Pass data between steps
    3. Handle step failures and retries
    4. Return structured results
    """
    
    # Simulate some processing time
    import asyncio
    await asyncio.sleep(0.1)
    
    result = {
        "response": f"Processed query '{query}' using pipeline '{pipeline.name}'. This is a comprehensive mock response with enhanced capabilities.",
        "sources": [
            {
                "type": "mock_vector_store",
                "content": f"Retrieved content relevant to: {query}",
                "score": 0.95,
                "metadata": {"source": "mock_database", "timestamp": datetime.now().isoformat()}
            },
            {
                "type": "mock_knowledge_base", 
                "content": f"Additional context for: {query}",
                "score": 0.87,
                "metadata": {"source": "mock_kb", "timestamp": datetime.now().isoformat()}
            }
        ],
        "metadata": {
            "steps_executed": len(pipeline.steps),
            "steps_details": [
                {"step_id": step.id, "step_name": step.name, "status": "completed"}
                for step in pipeline.steps
            ],
            "parameters": parameters,
            "context": context,
            "processing_time_ms": 100
        }
    }
    
    return result


@router.get("/executions/{execution_id}")
async def get_execution_result(execution_id: str):
    try:
        execution_uuid = UUID(execution_id)
        execution = pipeline_registry.get_execution(execution_uuid)
        
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        logger.info(f"Retrieved execution result for {execution_id}")
        
        return {
            "execution_id": execution_id,
            "pipeline_id": str(execution.pipeline_id),
            "status": execution.status.value,
            "started_at": execution.started_at.isoformat(),
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "execution_time_seconds": execution.execution_time_seconds,
            "input_data": execution.input_data,
            "output_data": execution.output_data,
            "error_message": execution.error_message,
            "timestamp": datetime.now().isoformat()
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid execution ID format")
    except Exception as e:
        logger.error(f"Failed to get execution result {execution_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/executions")
async def list_executions(
    pipeline_id: str = None,
    status: str = None,
    limit: int = 50
):
    try:
        # Convert parameters
        pipeline_uuid = None
        if pipeline_id:
            try:
                pipeline_uuid = UUID(pipeline_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid pipeline ID format")
        
        status_filter = None
        if status:
            try:
                status_filter = PipelineStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid status value")
        
        # Get executions from registry
        executions = pipeline_registry.list_executions(
            pipeline_id=pipeline_uuid,
            status=status_filter,
            limit=limit
        )
        
        logger.info(f"Listed {len(executions)} executions")
        
        return {
            "executions": [
                {
                    "id": str(exec.id),
                    "pipeline_id": str(exec.pipeline_id),
                    "status": exec.status.value,
                    "started_at": exec.started_at.isoformat(),
                    "completed_at": exec.completed_at.isoformat() if exec.completed_at else None,
                    "execution_time_seconds": exec.execution_time_seconds,
                    "error_message": exec.error_message
                }
                for exec in executions
            ],
            "total": len(executions),
            "message": f"Retrieved {len(executions)} executions",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list executions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))