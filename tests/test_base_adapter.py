"""Tests for BaseAdapter class."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4

from src.mcp_rag_control.adapters.base_adapter import BaseAdapter, ConnectionError, OperationError
from src.mcp_rag_control.models.module import Module, ModuleType, ModuleStatus


class TestableAdapter(BaseAdapter):
    """Testable implementation of BaseAdapter."""
    
    async def _get_health_details(self):
        return {"test": "details"}
    
    async def execute_operation(self, operation, parameters):
        return {"result": "success"}
    
    async def get_schema(self):
        return {"schema": "test"}


class TestBaseAdapter:
    """Test cases for BaseAdapter."""
    
    @pytest.mark.asyncio
    async def test_init_with_valid_url(self, sample_module, mock_config):
        """Test adapter initialization with valid URL."""
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            adapter = TestableAdapter(sample_module)
            assert adapter.module == sample_module
            assert not adapter.is_connected()
    
    @pytest.mark.asyncio
    async def test_init_with_invalid_url(self, sample_module, mock_config):
        """Test adapter initialization with invalid URL."""
        sample_module.mcp_server_url = "invalid://url with spaces"
        
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            with pytest.raises(ValueError, match="Invalid MCP server URL"):
                TestableAdapter(sample_module)
    
    @pytest.mark.asyncio
    async def test_connect_success(self, sample_module, mock_config, mock_httpx_client):
        """Test successful connection to MCP server."""
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            with patch('httpx.AsyncClient', return_value=mock_httpx_client):
                adapter = TestableAdapter(sample_module)
                
                await adapter.connect()
                
                assert adapter.is_connected()
                assert adapter.module.status == ModuleStatus.ACTIVE
                mock_httpx_client.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, sample_module, mock_config):
        """Test connection failure to MCP server."""
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client.get.side_effect = Exception("Connection failed")
                mock_client_class.return_value = mock_client
                
                adapter = TestableAdapter(sample_module)
                
                with pytest.raises(ConnectionError, match="Failed to connect"):
                    await adapter.connect()
                
                assert not adapter.is_connected()
                assert adapter.module.status == ModuleStatus.ERROR
    
    @pytest.mark.asyncio
    async def test_disconnect(self, sample_module, mock_config, mock_httpx_client):
        """Test disconnection from MCP server."""
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            with patch('httpx.AsyncClient', return_value=mock_httpx_client):
                adapter = TestableAdapter(sample_module)
                
                # Connect first
                await adapter.connect()
                assert adapter.is_connected()
                
                # Then disconnect
                await adapter.disconnect()
                
                assert not adapter.is_connected()
                assert adapter.module.status == ModuleStatus.INACTIVE
                mock_httpx_client.aclose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, sample_module, mock_config, mock_httpx_client):
        """Test successful health check."""
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            with patch('httpx.AsyncClient', return_value=mock_httpx_client):
                adapter = TestableAdapter(sample_module)
                await adapter.connect()
                
                health_info = await adapter.health_check()
                
                assert health_info["status"] == "healthy"
                assert "response_time_ms" in health_info
                assert health_info["connected"] is True
                assert health_info["module_id"] == str(sample_module.id)
                assert "test" in health_info  # From _get_health_details
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, sample_module, mock_config):
        """Test health check failure."""
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            adapter = TestableAdapter(sample_module)
            # Don't connect, so health check should fail
            
            health_info = await adapter.health_check()
            
            assert health_info["status"] == "unhealthy"
            assert "error" in health_info
            assert health_info["connected"] is False
    
    @pytest.mark.asyncio
    async def test_batch_execute_success(self, sample_module, mock_config, mock_httpx_client):
        """Test successful batch execution."""
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            with patch('httpx.AsyncClient', return_value=mock_httpx_client):
                adapter = TestableAdapter(sample_module)
                await adapter.connect()
                
                operations = [
                    {"operation": "test1", "parameters": {}},
                    {"operation": "test2", "parameters": {}}
                ]
                
                results = await adapter.batch_execute(operations)
                
                assert len(results) == 2
                assert all(result["success"] for result in results)
                assert all("result" in result for result in results)
    
    @pytest.mark.asyncio
    async def test_batch_execute_partial_failure(self, sample_module, mock_config, mock_httpx_client):
        """Test batch execution with partial failures."""
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            with patch('httpx.AsyncClient', return_value=mock_httpx_client):
                adapter = TestableAdapter(sample_module)
                await adapter.connect()
                
                # Override execute_operation to fail on second call
                call_count = 0
                async def mock_execute(operation, parameters):
                    nonlocal call_count
                    call_count += 1
                    if call_count == 2:
                        raise Exception("Test error")
                    return {"result": "success"}
                
                adapter.execute_operation = mock_execute
                
                operations = [
                    {"operation": "test1", "parameters": {}},
                    {"operation": "test2", "parameters": {}}
                ]
                
                results = await adapter.batch_execute(operations)
                
                assert len(results) == 2
                assert results[0]["success"] is True
                assert results[1]["success"] is False
                assert "error" in results[1]
    
    @pytest.mark.asyncio
    async def test_context_manager(self, sample_module, mock_config, mock_httpx_client):
        """Test adapter as async context manager."""
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            with patch('httpx.AsyncClient', return_value=mock_httpx_client):
                adapter = TestableAdapter(sample_module)
                
                async with adapter:
                    assert adapter.is_connected()
                    assert adapter.module.status == ModuleStatus.ACTIVE
                
                assert not adapter.is_connected()
                assert adapter.module.status == ModuleStatus.INACTIVE
    
    def test_get_supported_operations(self, sample_module, mock_config):
        """Test getting supported operations."""
        sample_module.supported_operations = ["op1", "op2", "op3"]
        
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            adapter = TestableAdapter(sample_module)
            operations = adapter.get_supported_operations()
            
            assert operations == ["op1", "op2", "op3"]
            # Ensure it returns a copy, not the original list
            operations.append("op4")
            assert "op4" not in adapter.get_supported_operations()