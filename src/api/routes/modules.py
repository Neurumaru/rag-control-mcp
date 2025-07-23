import uuid
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from ...models.module import (
    Module,
    ModuleListResponse,
    ModuleRegistrationRequest,
    ModuleResponse,
    ModuleStatus,
    ModuleUpdateRequest,
)
from ...models.request import FilterParams, PaginationParams
from ...registry.module_registry import ModuleRegistry
from ...utils.logger import get_logger

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
            tags=request.tags,
        )

        # Register module using the registry
        await registry.register_module(module)

        logger.info(f"Module '{request.name}' registered successfully with ID {module.id}")

        return ModuleResponse(
            module=module, message=f"Module '{request.name}' registered successfully"
        )
    except Exception as e:
        logger.error(f"Failed to register module '{request.name}': {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/modules", response_model=ModuleListResponse)
async def list_modules(pagination: PaginationParams = Depends(), filters: FilterParams = Depends()):
    try:
        # Get modules from registry
        modules_list = registry.list_modules()

        # Apply filters
        if filters.module_type:
            modules_list = [m for m in modules_list if m.module_type == filters.module_type]

        if filters.status:
            modules_list = [m for m in modules_list if m.status == filters.status]

        if filters.search:
            search_term = filters.search.lower()
            modules_list = [
                m
                for m in modules_list
                if search_term in m.name.lower()
                or (m.description and search_term in m.description.lower())
            ]

        # Apply pagination
        total = len(modules_list)
        start = (pagination.page - 1) * pagination.page_size
        end = start + pagination.page_size
        paginated_modules = modules_list[start:end]

        logger.info(f"Listed {len(paginated_modules)} modules (total: {total})")

        return ModuleListResponse(
            modules=paginated_modules,
            total=total,
            message=f"Retrieved {len(paginated_modules)} modules",
        )
    except Exception as e:
        logger.error(f"Failed to list modules: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/modules/{module_id}", response_model=ModuleResponse)
async def get_module(module_id: str):
    try:
        module_uuid = UUID(module_id)
        module = await registry.get_module(module_uuid)

        if not module:
            raise HTTPException(status_code=404, detail="Module not found")

        logger.info(f"Retrieved module '{module.name}' with ID {module_id}")

        return ModuleResponse(module=module, message="Module retrieved successfully")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid module ID format")
    except Exception as e:
        logger.error(f"Failed to get module {module_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/modules/{module_id}", response_model=ModuleResponse)
async def update_module(module_id: str, request: ModuleUpdateRequest):
    try:
        module_uuid = UUID(module_id)

        # Convert string dependency IDs to UUIDs if provided
        dependencies = None
        if request.dependencies is not None:
            dependencies = [UUID(dep_id) for dep_id in request.dependencies]

        updated_module = await registry.update_module(
            module_uuid,
            name=request.name,
            description=request.description,
            version=request.version,
            status=request.status,
            config=request.config,
            input_schema=request.input_schema,
            output_schema=request.output_schema,
            supported_operations=request.supported_operations,
            dependencies=dependencies,
            tags=request.tags,
        )

        if not updated_module:
            raise HTTPException(status_code=404, detail="Module not found")

        logger.info(f"Updated module '{updated_module.name}' with ID {module_id}")

        return ModuleResponse(module=updated_module, message="Module updated successfully")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid module ID format")
    except Exception as e:
        logger.error(f"Failed to update module {module_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/modules/{module_id}")
async def delete_module(module_id: str):
    try:
        module_uuid = UUID(module_id)

        # Get module name before deletion for logging
        module = await registry.get_module(module_uuid)
        if not module:
            raise HTTPException(status_code=404, detail="Module not found")

        success = await registry.delete_module(module_uuid)
        if not success:
            raise HTTPException(status_code=404, detail="Module not found")

        logger.info(f"Deleted module '{module.name}' with ID {module_id}")

        return {
            "message": f"Module '{module.name}' deleted successfully",
            "timestamp": datetime.now().isoformat(),
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid module ID format")
    except Exception as e:
        logger.error(f"Failed to delete module {module_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/modules/{module_id}/activate")
async def activate_module(module_id: str):
    try:
        module_uuid = UUID(module_id)

        updated_module = await registry.update_module(module_uuid, status=ModuleStatus.ACTIVE)

        if not updated_module:
            raise HTTPException(status_code=404, detail="Module not found")

        logger.info(f"Activated module '{updated_module.name}' with ID {module_id}")

        return {
            "message": f"Module '{updated_module.name}' activated successfully",
            "timestamp": datetime.now().isoformat(),
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid module ID format")
    except Exception as e:
        logger.error(f"Failed to activate module {module_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/modules/{module_id}/deactivate")
async def deactivate_module(module_id: str):
    try:
        module_uuid = UUID(module_id)

        updated_module = await registry.update_module(module_uuid, status=ModuleStatus.INACTIVE)

        if not updated_module:
            raise HTTPException(status_code=404, detail="Module not found")

        logger.info(f"Deactivated module '{updated_module.name}' with ID {module_id}")

        return {
            "message": f"Module '{updated_module.name}' deactivated successfully",
            "timestamp": datetime.now().isoformat(),
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid module ID format")
    except Exception as e:
        logger.error(f"Failed to deactivate module {module_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
