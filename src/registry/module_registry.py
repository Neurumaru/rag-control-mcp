"""Module registry for managing MCP modules."""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Set
from uuid import UUID

from ..adapters import BaseAdapter, VectorAdapter, DatabaseAdapter
from ..models.module import Module, ModuleStatus, ModuleType, ModuleHealthCheck
from ..storage.json_storage import JSONStorage
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModuleRegistryError(Exception):
    """Base exception for module registry errors."""
    pass


class ModuleNotFoundError(ModuleRegistryError):
    """Error when module is not found."""
    pass


class ModuleDependencyError(ModuleRegistryError):
    """Error with module dependencies."""
    pass


class ModuleRegistry:
    """Registry for managing MCP modules."""
    
    def __init__(self, storage: Optional[JSONStorage] = None):
        """Initialize module registry."""
        self._modules: Dict[UUID, Module] = {}
        self._adapters: Dict[UUID, BaseAdapter] = {}
        self._dependency_graph: Dict[UUID, Set[UUID]] = {}
        self._reverse_dependencies: Dict[UUID, Set[UUID]] = {}
        self._lock = asyncio.Lock()
        self._storage = storage or JSONStorage()
        
        # Load existing modules from storage
        self._load_modules_from_storage()
    
    async def register_module(self, module: Module) -> None:
        """Register a new module."""
        async with self._lock:
            if module.id in self._modules:
                raise ModuleRegistryError(f"Module {module.id} already registered")
            
            # Validate dependencies
            for dep_id in module.dependencies:
                if dep_id not in self._modules:
                    raise ModuleDependencyError(f"Dependency {dep_id} not found")
            
            # Create adapter
            adapter = self._create_adapter(module)
            
            # Store module and adapter
            self._modules[module.id] = module
            self._adapters[module.id] = adapter
            
            # Update dependency graph
            self._dependency_graph[module.id] = set(module.dependencies)
            self._reverse_dependencies[module.id] = set()
            
            for dep_id in module.dependencies:
                self._reverse_dependencies[dep_id].add(module.id)
            
            # Try to connect adapter
            try:
                await adapter.connect()
                module.update_status(ModuleStatus.ACTIVE)
            except Exception as e:
                module.update_status(ModuleStatus.ERROR)
                # Don't raise here - allow registration but mark as error
            
            # Save to storage
            self._save_module_to_storage(module)
    
    async def unregister_module(self, module_id: UUID, force: bool = False) -> None:
        """Unregister a module."""
        async with self._lock:
            if module_id not in self._modules:
                raise ModuleNotFoundError(f"Module {module_id} not found")
            
            # Check for dependent modules
            dependents = self._reverse_dependencies.get(module_id, set())
            if dependents and not force:
                raise ModuleDependencyError(
                    f"Module {module_id} has dependents: {dependents}. Use force=True to remove anyway."
                )
            
            # Disconnect adapter
            adapter = self._adapters.get(module_id)
            if adapter:
                await adapter.disconnect()
            
            # Remove from registry
            del self._modules[module_id]
            del self._adapters[module_id]
            
            # Update dependency graph
            deps = self._dependency_graph.get(module_id, set())
            for dep_id in deps:
                self._reverse_dependencies[dep_id].discard(module_id)
            
            del self._dependency_graph[module_id]
            del self._reverse_dependencies[module_id]
            
            # If force removal, update dependent modules
            if force and dependents:
                for dependent_id in dependents:
                    dependent = self._modules.get(dependent_id)
                    if dependent:
                        dependent.remove_dependency(module_id)
                        self._dependency_graph[dependent_id].discard(module_id)
                        # Update dependent module in storage
                        self._save_module_to_storage(dependent)
            
            # Remove from storage
            self._delete_module_from_storage(module_id)
    
    def get_module(self, module_id: UUID) -> Optional[Module]:
        """Get module by ID."""
        return self._modules.get(module_id)
    
    def get_adapter(self, module_id: UUID) -> Optional[BaseAdapter]:
        """Get adapter by module ID."""
        return self._adapters.get(module_id)
    
    def list_modules(
        self, 
        module_type: Optional[ModuleType] = None,
        status: Optional[ModuleStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[Module]:
        """List modules with optional filtering."""
        modules = list(self._modules.values())
        
        if module_type:
            modules = [m for m in modules if m.module_type == module_type]
        
        if status:
            modules = [m for m in modules if m.status == status]
        
        if tags:
            modules = [m for m in modules if any(tag in m.tags for tag in tags)]
        
        return modules
    
    def find_modules_by_name(self, name: str, exact_match: bool = False) -> List[Module]:
        """Find modules by name."""
        if exact_match:
            return [m for m in self._modules.values() if m.name == name]
        else:
            return [m for m in self._modules.values() if name.lower() in m.name.lower()]
    
    def get_module_dependencies(self, module_id: UUID) -> List[UUID]:
        """Get direct dependencies of a module."""
        return list(self._dependency_graph.get(module_id, set()))
    
    def get_module_dependents(self, module_id: UUID) -> List[UUID]:
        """Get modules that depend on this module."""
        return list(self._reverse_dependencies.get(module_id, set()))
    
    def get_dependency_chain(self, module_id: UUID) -> List[UUID]:
        """Get full dependency chain for a module."""
        visited = set()
        chain = []
        
        def _dfs(mid: UUID):
            if mid in visited:
                return
            visited.add(mid)
            
            for dep_id in self._dependency_graph.get(mid, set()):
                _dfs(dep_id)
            
            chain.append(mid)
        
        _dfs(module_id)
        return chain
    
    def validate_dependencies(self) -> List[str]:
        """Validate all module dependencies."""
        errors = []
        
        # Check for circular dependencies
        for module_id in self._modules:
            if self._has_circular_dependency(module_id):
                errors.append(f"Circular dependency detected involving module {module_id}")
        
        # Check for missing dependencies
        for module_id, deps in self._dependency_graph.items():
            for dep_id in deps:
                if dep_id not in self._modules:
                    errors.append(f"Module {module_id} depends on non-existent module {dep_id}")
        
        return errors
    
    def _has_circular_dependency(self, start_id: UUID) -> bool:
        """Check if a module has circular dependencies."""
        visited = set()
        rec_stack = set()
        
        def _dfs(module_id: UUID) -> bool:
            visited.add(module_id)
            rec_stack.add(module_id)
            
            for dep_id in self._dependency_graph.get(module_id, set()):
                if dep_id not in visited:
                    if _dfs(dep_id):
                        return True
                elif dep_id in rec_stack:
                    return True
            
            rec_stack.remove(module_id)
            return False
        
        return _dfs(start_id)
    
    async def health_check_module(self, module_id: UUID) -> ModuleHealthCheck:
        """Perform health check on a specific module."""
        if module_id not in self._modules:
            raise ModuleNotFoundError(f"Module {module_id} not found")
        
        module = self._modules[module_id]
        adapter = self._adapters[module_id]
        
        start_time = datetime.utcnow()
        
        try:
            if adapter:
                health_info = await adapter.health_check()
                status = ModuleStatus.ACTIVE if health_info["status"] == "healthy" else ModuleStatus.ERROR
                response_time = health_info.get("response_time_ms", 0)
                error_message = health_info.get("error")
            else:
                status = ModuleStatus.ERROR
                response_time = 0
                error_message = "No adapter available"
            
            # Update module status
            module.update_status(status)
            module.last_health_check = start_time
            
            return ModuleHealthCheck(
                module_id=module_id,
                status=status,
                response_time_ms=response_time,
                error_message=error_message,
                timestamp=start_time
            )
            
        except Exception as e:
            module.update_status(ModuleStatus.ERROR)
            return ModuleHealthCheck(
                module_id=module_id,
                status=ModuleStatus.ERROR,
                response_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                error_message=str(e),
                timestamp=start_time
            )
    
    async def health_check_all(self) -> List[ModuleHealthCheck]:
        """Perform health check on all modules."""
        tasks = []
        for module_id in self._modules:
            tasks.append(self.health_check_module(module_id))
        
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    async def start_all_modules(self) -> Dict[UUID, bool]:
        """Start all modules in dependency order."""
        results = {}
        
        # Get modules in dependency order
        for module_id in self._modules:
            chain = self.get_dependency_chain(module_id)
            
            for mid in chain:
                if mid in results:
                    continue
                
                try:
                    adapter = self._adapters.get(mid)
                    if adapter and not adapter.is_connected():
                        await adapter.connect()
                        self._modules[mid].update_status(ModuleStatus.ACTIVE)
                    results[mid] = True
                except Exception:
                    self._modules[mid].update_status(ModuleStatus.ERROR)
                    results[mid] = False
        
        return results
    
    async def stop_all_modules(self) -> Dict[UUID, bool]:
        """Stop all modules in reverse dependency order."""
        results = {}
        
        # Get modules in reverse dependency order
        modules_by_depth = {}
        for module_id in self._modules:
            depth = len(self.get_dependency_chain(module_id)) - 1
            if depth not in modules_by_depth:
                modules_by_depth[depth] = []
            modules_by_depth[depth].append(module_id)
        
        # Stop modules from deepest to shallowest
        for depth in sorted(modules_by_depth.keys(), reverse=True):
            for module_id in modules_by_depth[depth]:
                try:
                    adapter = self._adapters.get(module_id)
                    if adapter:
                        await adapter.disconnect()
                    self._modules[module_id].update_status(ModuleStatus.INACTIVE)
                    results[module_id] = True
                except Exception:
                    results[module_id] = False
        
        return results
    
    def _create_adapter(self, module: Module) -> BaseAdapter:
        """Create appropriate adapter for module type."""
        if module.module_type == ModuleType.VECTOR_STORE:
            return VectorAdapter(module)
        elif module.module_type == ModuleType.DATABASE:
            return DatabaseAdapter(module)
        else:
            # For other module types, use base adapter
            return BaseAdapter(module)
    
    def get_stats(self) -> Dict[str, any]:
        """Get registry statistics."""
        total_modules = len(self._modules)
        active_modules = len([m for m in self._modules.values() if m.status == ModuleStatus.ACTIVE])
        error_modules = len([m for m in self._modules.values() if m.status == ModuleStatus.ERROR])
        
        module_types = {}
        for module in self._modules.values():
            module_type = module.module_type.value
            module_types[module_type] = module_types.get(module_type, 0) + 1
        
        return {
            "total_modules": total_modules,
            "active_modules": active_modules,
            "error_modules": error_modules,
            "inactive_modules": total_modules - active_modules - error_modules,
            "module_types": module_types,
            "total_dependencies": sum(len(deps) for deps in self._dependency_graph.values()),
        }
    
    def _load_modules_from_storage(self) -> None:
        """Load modules from storage on startup."""
        try:
            modules = self._storage.load_all_modules()
            for module in modules:
                self._modules[module.id] = module
                self._update_dependency_graph(module)
                logger.info(f"Loaded module {module.id} from storage")
        except Exception as e:
            logger.error(f"Failed to load modules from storage: {e}")
    
    def _save_module_to_storage(self, module: Module) -> None:
        """Save a module to storage."""
        try:
            self._storage.save_module(module)
        except Exception as e:
            logger.error(f"Failed to save module {module.id} to storage: {e}")
    
    def _delete_module_from_storage(self, module_id: UUID) -> None:
        """Delete a module from storage."""
        try:
            self._storage.delete_module(module_id)
        except Exception as e:
            logger.error(f"Failed to delete module {module_id} from storage: {e}")
    
    def _update_dependency_graph(self, module: Module) -> None:
        """Update dependency graph for a module."""
        self._dependency_graph[module.id] = set(module.dependencies)
        if module.id not in self._reverse_dependencies:
            self._reverse_dependencies[module.id] = set()
        
        for dep_id in module.dependencies:
            if dep_id not in self._reverse_dependencies:
                self._reverse_dependencies[dep_id] = set()
            self._reverse_dependencies[dep_id].add(module.id)