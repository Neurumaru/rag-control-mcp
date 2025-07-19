"""Tests for VectorAdapter class."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4

from src.mcp_rag_control.adapters.vector_adapter import VectorAdapter
from src.mcp_rag_control.models.module import Module, ModuleType, ModuleConfig


class TestVectorAdapter:
    """Test cases for VectorAdapter."""
    
    @pytest.fixture
    def vector_module(self):
        """Create a vector store module for testing."""
        return Module(
            id=uuid4(),
            name="test_vector_store",
            module_type=ModuleType.VECTOR_STORE,
            mcp_server_url="https://vector.example.com/mcp",
            config=ModuleConfig(
                host="localhost",
                port=8080,
                collection_name="test_collection",
                dimensions=512
            ),
            supported_operations=["search", "add", "delete", "update", "create_collection"]
        )
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        from src.mcp_rag_control.utils.config import AppConfig
        return AppConfig(debug=True, environment="test")
    
    @pytest.mark.asyncio
    async def test_search_vectors_success(self, vector_module, mock_config):
        """Test successful vector search operation."""
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            with patch('httpx.AsyncClient') as mock_client_class:
                # Setup mock client
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # Mock successful search response
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "jsonrpc": "2.0",
                    "result": {
                        "results": [
                            {"id": "1", "score": 0.95, "metadata": {"title": "Test 1"}},
                            {"id": "2", "score": 0.87, "metadata": {"title": "Test 2"}}
                        ]
                    },
                    "id": "test_id"
                }
                mock_client.post.return_value = mock_response
                
                # Mock health check
                health_response = MagicMock()
                health_response.status_code = 200
                health_response.raise_for_status.return_value = None
                mock_client.get.return_value = health_response
                
                adapter = VectorAdapter(vector_module)
                await adapter.connect()
                
                # Test search
                results = await adapter.search_vectors([0.1, 0.2, 0.3], top_k=2)
                
                assert "results" in results
                assert len(results["results"]) == 2
                assert results["results"][0]["score"] == 0.95
    
    @pytest.mark.asyncio
    async def test_add_vectors_success(self, vector_module, mock_config):
        """Test successful vector addition operation."""
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # Mock successful add response
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "jsonrpc": "2.0",
                    "result": {"added_count": 2, "ids": ["1", "2"]},
                    "id": "test_id"
                }
                mock_client.post.return_value = mock_response
                
                # Mock health check
                health_response = MagicMock()
                health_response.status_code = 200
                health_response.raise_for_status.return_value = None
                mock_client.get.return_value = health_response
                
                adapter = VectorAdapter(vector_module)
                await adapter.connect()
                
                # Test add vectors
                vectors = [
                    {"id": "1", "vector": [0.1, 0.2, 0.3], "metadata": {"title": "Test 1"}},
                    {"id": "2", "vector": [0.4, 0.5, 0.6], "metadata": {"title": "Test 2"}}
                ]
                
                result = await adapter.add_vectors(vectors)
                
                assert result["added_count"] == 2
                assert "1" in result["ids"]
                assert "2" in result["ids"]
    
    @pytest.mark.asyncio
    async def test_delete_vectors_success(self, vector_module, mock_config):
        """Test successful vector deletion operation."""
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # Mock successful delete response
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "jsonrpc": "2.0",
                    "result": {"deleted_count": 2},
                    "id": "test_id"
                }
                mock_client.post.return_value = mock_response
                
                # Mock health check
                health_response = MagicMock()
                health_response.status_code = 200
                health_response.raise_for_status.return_value = None
                mock_client.get.return_value = health_response
                
                adapter = VectorAdapter(vector_module)
                await adapter.connect()
                
                # Test delete vectors
                result = await adapter.delete_vectors(["1", "2"])
                
                assert result["deleted_count"] == 2
    
    @pytest.mark.asyncio
    async def test_get_health_details(self, vector_module, mock_config):
        """Test getting vector store health details."""
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # Mock collection stats response
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "jsonrpc": "2.0",
                    "result": {
                        "collection_name": "test_collection",
                        "total_vectors": 1000,
                        "dimensions": 512,
                        "index_size_mb": 25.5,
                        "memory_usage_mb": 50.2
                    },
                    "id": "test_id"
                }
                mock_client.post.return_value = mock_response
                
                # Mock health check
                health_response = MagicMock()
                health_response.status_code = 200
                health_response.raise_for_status.return_value = None
                mock_client.get.return_value = health_response
                
                adapter = VectorAdapter(vector_module)
                await adapter.connect()
                
                # Test health details
                health_details = await adapter._get_health_details()
                
                assert health_details["collection_name"] == "test_collection"
                assert health_details["total_vectors"] == 1000
                assert health_details["dimensions"] == 512
                assert health_details["index_size_mb"] == 25.5
    
    @pytest.mark.asyncio
    async def test_create_collection_success(self, vector_module, mock_config):
        """Test successful collection creation."""
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                
                # Mock successful creation response
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "jsonrpc": "2.0",
                    "result": {
                        "collection_name": "new_collection",
                        "dimensions": 768,
                        "created": True
                    },
                    "id": "test_id"
                }
                mock_client.post.return_value = mock_response
                
                # Mock health check
                health_response = MagicMock()
                health_response.status_code = 200
                health_response.raise_for_status.return_value = None
                mock_client.get.return_value = health_response
                
                adapter = VectorAdapter(vector_module)
                await adapter.connect()
                
                # Test collection creation
                result = await adapter.create_collection("new_collection", dimensions=768)
                
                assert result["collection_name"] == "new_collection"
                assert result["dimensions"] == 768
                assert result["created"] is True
    
    @pytest.mark.asyncio
    async def test_execute_operation_unsupported(self, vector_module, mock_config):
        """Test executing unsupported operation raises error."""
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            adapter = VectorAdapter(vector_module)
            
            from src.mcp_rag_control.adapters.base_adapter import OperationError
            with pytest.raises(OperationError, match="Unsupported operation"):
                await adapter.execute_operation("unsupported_op", {})
    
    @pytest.mark.asyncio
    async def test_get_schema(self, vector_module, mock_config):
        """Test getting adapter schema."""
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            adapter = VectorAdapter(vector_module)
            schema = await adapter.get_schema()
            
            assert "vector_store" in schema
            assert "operations" in schema["vector_store"]
            assert "search" in schema["vector_store"]["operations"]
            assert "add" in schema["vector_store"]["operations"]
    
    def test_validate_vector_dimensions(self, vector_module, mock_config):
        """Test vector dimension validation."""
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            adapter = VectorAdapter(vector_module)
            
            # Valid dimensions
            assert adapter._validate_vector_dimensions([0.1, 0.2, 0.3], expected_dim=3) is True
            
            # Invalid dimensions
            assert adapter._validate_vector_dimensions([0.1, 0.2], expected_dim=3) is False
            assert adapter._validate_vector_dimensions([], expected_dim=3) is False
            assert adapter._validate_vector_dimensions(None, expected_dim=3) is False
    
    def test_validate_vector_values(self, vector_module, mock_config):
        """Test vector value validation."""
        with patch('src.mcp_rag_control.adapters.base_adapter.get_config', return_value=mock_config):
            adapter = VectorAdapter(vector_module)
            
            # Valid vectors
            assert adapter._validate_vector_values([0.1, 0.2, 0.3]) is True
            assert adapter._validate_vector_values([-1.0, 0.0, 1.0]) is True
            
            # Invalid vectors
            assert adapter._validate_vector_values([float('inf'), 0.2, 0.3]) is False
            assert adapter._validate_vector_values([0.1, float('nan'), 0.3]) is False
            assert adapter._validate_vector_values(["a", 0.2, 0.3]) is False