"""Tests for ModuleRegistry class."""

import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from src.mcp_rag_control.registry.module_registry import (
    ModuleRegistry, ModuleRegistryError, ModuleNotFoundError, ModuleDependencyError
)
from src.mcp_rag_control.models.module import Module, ModuleType, ModuleStatus, ModuleConfig


class TestModuleRegistry:
    """Test cases for ModuleRegistry."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        # Create registry with empty in-memory storage for testing
        registry = ModuleRegistry()
        # Clear any loaded modules for clean test state
        registry._modules.clear()
        registry._dependency_graph.clear()
        return registry
    
    @pytest.fixture
    def vector_module(self):
        """Create a vector store module for testing."""
        return Module(
            id=uuid4(),
            name="test_vector_store",
            module_type=ModuleType.VECTOR_STORE,
            mcp_server_url="https://vector.example.com/mcp",
            config=ModuleConfig(host="localhost", port=8080)
        )
    
    @pytest.fixture
    def database_module(self):
        """Create a database module for testing."""
        return Module(
            id=uuid4(),
            name="test_database",
            module_type=ModuleType.DATABASE_CONNECTOR,
            mcp_server_url="https://db.example.com/mcp",
            config=ModuleConfig(host="localhost", port=5432)
        )
    
    @pytest.mark.asyncio
    async def test_register_module_success(self, registry, vector_module):
        """Test successful module registration."""
        with patch.object(registry, '_create_adapter') as mock_create:
            mock_adapter = AsyncMock()
            mock_create.return_value = mock_adapter
            
            await registry.register_module(vector_module)
            
            # Verify module is registered
            assert registry.get_module(vector_module.id) == vector_module
            assert registry.get_adapter(vector_module.id) == mock_adapter
            
            # Verify adapter connection was attempted
            mock_adapter.connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_register_duplicate_module(self, registry, vector_module):
        """Test registering duplicate module raises error."""
        with patch.object(registry, '_create_adapter') as mock_create:
            mock_adapter = AsyncMock()
            mock_create.return_value = mock_adapter
            
            # Register once
            await registry.register_module(vector_module)
            
            # Try to register again
            with pytest.raises(ModuleRegistryError, match="already registered"):
                await registry.register_module(vector_module)
    
    @pytest.mark.asyncio
    async def test_register_module_with_missing_dependency(self, registry, vector_module):
        """Test registering module with missing dependency."""
        # Add non-existent dependency
        vector_module.dependencies = [uuid4()]
        
        with pytest.raises(ModuleDependencyError, match="Dependency .* not found"):
            await registry.register_module(vector_module)
    
    @pytest.mark.asyncio
    async def test_register_module_with_valid_dependency(self, registry, vector_module, database_module):
        """Test registering module with valid dependency."""
        with patch.object(registry, '_create_adapter') as mock_create:
            mock_adapter = AsyncMock()
            mock_create.return_value = mock_adapter
            
            # Register database module first
            await registry.register_module(database_module)
            
            # Add database as dependency of vector module
            vector_module.dependencies = [database_module.id]
            
            # Register vector module
            await registry.register_module(vector_module)
            
            # Verify both are registered
            assert registry.get_module(vector_module.id) == vector_module
            assert registry.get_module(database_module.id) == database_module
            
            # Verify dependency tracking
            deps = registry.get_module_dependencies(vector_module.id)
            assert database_module.id in deps
            
            dependents = registry.get_module_dependents(database_module.id)
            assert vector_module.id in dependents
    
    @pytest.mark.asyncio
    async def test_unregister_module_success(self, registry, vector_module):
        """Test successful module unregistration."""
        with patch.object(registry, '_create_adapter') as mock_create:
            mock_adapter = AsyncMock()
            mock_create.return_value = mock_adapter
            
            # Register module
            await registry.register_module(vector_module)
            
            # Unregister module
            await registry.unregister_module(vector_module.id)
            
            # Verify module is removed
            assert registry.get_module(vector_module.id) is None
            assert registry.get_adapter(vector_module.id) is None
            
            # Verify adapter was disconnected
            mock_adapter.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_unregister_nonexistent_module(self, registry):
        """Test unregistering non-existent module raises error."""
        with pytest.raises(ModuleNotFoundError, match="not found"):
            await registry.unregister_module(uuid4())
    
    @pytest.mark.asyncio
    async def test_unregister_module_with_dependents(self, registry, vector_module, database_module):
        """Test unregistering module with dependents."""
        with patch.object(registry, '_create_adapter') as mock_create:
            mock_adapter = AsyncMock()
            mock_create.return_value = mock_adapter
            
            # Register both modules with dependency
            await registry.register_module(database_module)
            vector_module.dependencies = [database_module.id]
            await registry.register_module(vector_module)
            
            # Try to unregister database (has dependents)
            with pytest.raises(ModuleDependencyError, match="has dependents"):
                await registry.unregister_module(database_module.id)
    
    @pytest.mark.asyncio
    async def test_unregister_module_with_dependents_force(self, registry, vector_module, database_module):
        """Test force unregistering module with dependents."""
        with patch.object(registry, '_create_adapter') as mock_create:
            mock_adapter = AsyncMock()
            mock_create.return_value = mock_adapter
            
            # Register both modules with dependency
            await registry.register_module(database_module)
            vector_module.dependencies = [database_module.id]
            await registry.register_module(vector_module)
            
            # Force unregister database
            await registry.unregister_module(database_module.id, force=True)
            
            # Verify database is removed
            assert registry.get_module(database_module.id) is None
            
            # Verify dependency was removed from vector module
            vector_deps = registry.get_module_dependencies(vector_module.id)
            assert database_module.id not in vector_deps
    
    def test_list_modules_no_filter(self, registry, vector_module, database_module):
        """Test listing all modules without filters."""
        # Manually add modules to registry (bypass async registration)
        registry._modules[vector_module.id] = vector_module
        registry._modules[database_module.id] = database_module
        
        modules = registry.list_modules()
        assert len(modules) == 2
        assert vector_module in modules
        assert database_module in modules
    
    def test_list_modules_by_type(self, registry, vector_module, database_module):
        """Test listing modules filtered by type."""
        registry._modules[vector_module.id] = vector_module
        registry._modules[database_module.id] = database_module
        
        vector_modules = registry.list_modules(module_type=ModuleType.VECTOR_STORE)
        assert len(vector_modules) == 1
        assert vector_modules[0] == vector_module
        
        db_modules = registry.list_modules(module_type=ModuleType.DATABASE_CONNECTOR)
        assert len(db_modules) == 1
        assert db_modules[0] == database_module
    
    def test_list_modules_by_status(self, registry, vector_module):
        """Test listing modules filtered by status."""
        vector_module.update_status(ModuleStatus.ACTIVE)
        registry._modules[vector_module.id] = vector_module
        
        active_modules = registry.list_modules(status=ModuleStatus.ACTIVE)
        assert len(active_modules) == 1
        assert active_modules[0] == vector_module
        
        inactive_modules = registry.list_modules(status=ModuleStatus.INACTIVE)
        assert len(inactive_modules) == 0
    
    def test_find_modules_by_name(self, registry, vector_module, database_module):
        """Test finding modules by name."""
        registry._modules[vector_module.id] = vector_module
        registry._modules[database_module.id] = database_module
        
        # Exact match
        exact_matches = registry.find_modules_by_name("test_vector_store", exact_match=True)
        assert len(exact_matches) == 1
        assert exact_matches[0] == vector_module
        
        # Partial match
        partial_matches = registry.find_modules_by_name("test", exact_match=False)
        assert len(partial_matches) == 2
    
    def test_get_dependency_chain(self, registry):
        """Test getting dependency chain for modules."""
        # Create modules with chain: A -> B -> C
        module_c = Module(id=uuid4(), name="C", module_type=ModuleType.DATABASE_CONNECTOR, 
                         mcp_server_url="https://c.example.com/mcp")
        module_b = Module(id=uuid4(), name="B", module_type=ModuleType.VECTOR_STORE, 
                         mcp_server_url="https://b.example.com/mcp", dependencies=[module_c.id])
        module_a = Module(id=uuid4(), name="A", module_type=ModuleType.LLM_GENERATOR, 
                         mcp_server_url="https://a.example.com/mcp", dependencies=[module_b.id])
        
        # Add to registry
        registry._modules[module_c.id] = module_c
        registry._modules[module_b.id] = module_b
        registry._modules[module_a.id] = module_a
        registry._dependency_graph[module_c.id] = set()
        registry._dependency_graph[module_b.id] = {module_c.id}
        registry._dependency_graph[module_a.id] = {module_b.id}
        
        # Get dependency chain for A
        chain = registry.get_dependency_chain(module_a.id)
        
        # Chain should be [C, B, A] (dependency order)
        assert len(chain) == 3
        assert chain[0] == module_c.id
        assert chain[1] == module_b.id
        assert chain[2] == module_a.id
    
    def test_validate_dependencies_circular(self, registry):
        """Test validation detects circular dependencies."""
        # Create circular dependency: A -> B -> A
        module_a = Module(id=uuid4(), name="A", module_type=ModuleType.DATABASE_CONNECTOR, 
                         mcp_server_url="https://a.example.com/mcp")
        module_b = Module(id=uuid4(), name="B", module_type=ModuleType.VECTOR_STORE, 
                         mcp_server_url="https://b.example.com/mcp")
        
        registry._modules[module_a.id] = module_a
        registry._modules[module_b.id] = module_b
        registry._dependency_graph[module_a.id] = {module_b.id}
        registry._dependency_graph[module_b.id] = {module_a.id}
        
        errors = registry.validate_dependencies()
        assert len(errors) >= 1
        assert any("Circular dependency" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_health_check_module(self, registry, vector_module):
        """Test health check for specific module."""
        with patch.object(registry, '_create_adapter') as mock_create:
            mock_adapter = AsyncMock()
            mock_adapter.health_check.return_value = {
                "status": "healthy",
                "response_time_ms": 100.0
            }
            mock_create.return_value = mock_adapter
            
            await registry.register_module(vector_module)
            
            health_check = await registry.health_check_module(vector_module.id)
            
            assert health_check.module_id == vector_module.id
            assert health_check.status == ModuleStatus.ACTIVE
            assert health_check.response_time_ms == 100.0
    
    def test_get_stats(self, registry, vector_module, database_module):
        """Test getting registry statistics."""
        vector_module.update_status(ModuleStatus.ACTIVE)
        database_module.update_status(ModuleStatus.ERROR)
        
        registry._modules[vector_module.id] = vector_module
        registry._modules[database_module.id] = database_module
        
        stats = registry.get_stats()
        
        assert stats["total_modules"] == 2
        assert stats["active_modules"] == 1
        assert stats["error_modules"] == 1
        assert stats["inactive_modules"] == 0
        assert stats["module_types"]["vector_store"] == 1
        assert stats["module_types"]["database_connector"] == 1