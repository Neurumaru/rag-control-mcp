from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
import uuid
from datetime import datetime

from ...models.module import (
    Module, ModuleStatus, ModuleRegistrationRequest, 
    ModuleUpdateRequest, ModuleResponse, ModuleListResponse
)
from ...models.request import PaginationParams, FilterParams
from ...registry.module_registry import ModuleRegistry
from ...utils.logger import get_logger
from uuid import UUID

logger = get_logger(__name__)
registry = ModuleRegistry()

router = APIRouter()


@router.post("/modules", response_model=ModuleResponse)
async def register_module(request: ModuleRegistrationRequest):
    try:
        # Convert string dependency IDs to UUIDs
        dependencies = [UUID(dep_id) for dep_id in request.dependencies]
        
        module = Module(
            name=request.name,
            module_type=request.module_type,
            description=request.description,
            version=request.version,
            mcp_server_url=request.mcp_server_url,
            mcp_protocol_version=request.mcp_protocol_version,
            config=request.config,
            input_schema=request.input_schema,
            output_schema=request.output_schema,
            supported_operations=request.supported_operations,
            dependencies=dependencies,
            tags=request.tags
        )
        
        # Register module using the registry
        await registry.register_module(module)
        
        logger.info(f"Module '{request.name}' registered successfully with ID {module.id}")
        
        return ModuleResponse(
            module=module,
            message=f"Module '{request.name}' registered successfully"
        )
    except Exception as e:
        logger.error(f"Failed to register module '{request.name}': {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/modules", response_model=ModuleListResponse)
async def list_modules(
    pagination: PaginationParams = Depends(),
    filters: FilterParams = Depends()
):
    modules_list = list(modules_store.values())
    
    if filters.module_type:
        modules_list = [m for m in modules_list if m.type == filters.module_type]
    
    if filters.status:
        modules_list = [m for m in modules_list if m.status == filters.status]
    
    if filters.search:
        search_term = filters.search.lower()
        modules_list = [
            m for m in modules_list
            if search_term in m.name.lower() or
               (m.description and search_term in m.description.lower())
        ]
    
    total = len(modules_list)
    start = (pagination.page - 1) * pagination.page_size
    end = start + pagination.page_size
    paginated_modules = modules_list[start:end]
    
    return ModuleListResponse(
        modules=paginated_modules,
        total=total,
        message=f"Retrieved {len(paginated_modules)} modules"
    )


@router.get("/modules/{module_id}", response_model=ModuleResponse)
async def get_module(module_id: str):
    if module_id not in modules_store:
        raise HTTPException(status_code=404, detail="Module not found")
    
    module = modules_store[module_id]
    return ModuleResponse(
        module=module,
        message="Module retrieved successfully"
    )


@router.put("/modules/{module_id}", response_model=ModuleResponse)
async def update_module(module_id: str, request: ModuleUpdateRequest):
    if module_id not in modules_store:
        raise HTTPException(status_code=404, detail="Module not found")
    
    module = modules_store[module_id]
    
    update_data = request.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(module, field, value)
    
    modules_store[module_id] = module
    
    return ModuleResponse(
        module=module,
        message="Module updated successfully"
    )


@router.delete("/modules/{module_id}")
async def delete_module(module_id: str):
    if module_id not in modules_store:
        raise HTTPException(status_code=404, detail="Module not found")
    
    module = modules_store.pop(module_id)
    
    return {
        "message": f"Module '{module.name}' deleted successfully",
        "timestamp": datetime.now().isoformat()
    }


@router.post("/modules/{module_id}/activate")
async def activate_module(module_id: str):
    if module_id not in modules_store:
        raise HTTPException(status_code=404, detail="Module not found")
    
    module = modules_store[module_id]
    module.status = ModuleStatus.ACTIVE
    
    return {
        "message": f"Module '{module.name}' activated successfully",
        "timestamp": datetime.now().isoformat()
    }


@router.post("/modules/{module_id}/deactivate")
async def deactivate_module(module_id: str):
    if module_id not in modules_store:
        raise HTTPException(status_code=404, detail="Module not found")
    
    module = modules_store[module_id]
    module.status = ModuleStatus.INACTIVE
    
    return {
        "message": f"Module '{module.name}' deactivated successfully",
        "timestamp": datetime.now().isoformat()
    }