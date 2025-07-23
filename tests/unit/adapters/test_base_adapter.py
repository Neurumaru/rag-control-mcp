"""Unit tests for BaseAdapter class using unittest framework."""

import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID, uuid4

import httpx
from pydantic import ValidationError

from src.adapters.base_adapter import (
    AdapterError,
    BaseAdapter,
    ConnectionError,
    MCPRequest,
    MCPResponse,
    OperationError,
)
from src.models.module import Module, ModuleStatus, ModuleType


class ConcreteAdapter(BaseAdapter):
    """Concrete implementation of BaseAdapter for testing abstract methods."""

    async def _get_health_details(self):
        """Test implementation of abstract method."""
        return {"test_detail": "healthy", "custom_metric": 100}

    async def execute_operation(self, operation: str, parameters):
        """Test implementation of abstract method."""
        if operation == "test_operation":
            return {"result": "success", "parameters": parameters}
        elif operation == "error_operation":
            raise OperationError("Test operation error")
        else:
            return {"result": "unknown_operation"}

    async def get_schema(self):
        """Test implementation of abstract method."""
        return {
            "operations": ["test_operation", "error_operation"],
            "parameters": {"test_param": "string"},
        }


class TestMCPRequest(unittest.TestCase):
    """Test cases for MCPRequest model."""

    def test_mcp_request_creation(self):
        """Test MCPRequest creation with valid data."""
        request = MCPRequest(method="test_method", params={"param1": "value1"}, id="test_id")

        self.assertEqual(request.jsonrpc, "2.0")
        self.assertEqual(request.method, "test_method")
        self.assertEqual(request.params, {"param1": "value1"})
        self.assertEqual(request.id, "test_id")

    def test_mcp_request_defaults(self):
        """Test MCPRequest creation with default values."""
        request = MCPRequest(method="test_method", params={}, id="test_id")

        self.assertEqual(request.jsonrpc, "2.0")

    def test_mcp_request_serialization(self):
        """Test MCPRequest JSON serialization."""
        request = MCPRequest(method="test_method", params={"param1": "value1"}, id="test_id")

        expected_dict = {
            "jsonrpc": "2.0",
            "method": "test_method",
            "params": {"param1": "value1"},
            "id": "test_id",
        }

        self.assertEqual(request.dict(), expected_dict)

    def test_mcp_request_validation_error(self):
        """Test MCPRequest validation with missing required fields."""
        with self.assertRaises(ValidationError):
            MCPRequest(method="test_method")  # Missing params and id


class TestMCPResponse(unittest.TestCase):
    """Test cases for MCPResponse model."""

    def test_mcp_response_success(self):
        """Test MCPResponse creation for successful response."""
        response = MCPResponse(result={"data": "test_data"}, id="test_id")

        self.assertEqual(response.jsonrpc, "2.0")
        self.assertEqual(response.result, {"data": "test_data"})
        self.assertIsNone(response.error)
        self.assertEqual(response.id, "test_id")

    def test_mcp_response_error(self):
        """Test MCPResponse creation for error response."""
        response = MCPResponse(error={"code": -1, "message": "Test error"}, id="test_id")

        self.assertEqual(response.jsonrpc, "2.0")
        self.assertIsNone(response.result)
        self.assertEqual(response.error, {"code": -1, "message": "Test error"})
        self.assertEqual(response.id, "test_id")

    def test_mcp_response_defaults(self):
        """Test MCPResponse creation with default values."""
        response = MCPResponse(id="test_id")

        self.assertEqual(response.jsonrpc, "2.0")
        self.assertIsNone(response.result)
        self.assertIsNone(response.error)

    def test_mcp_response_serialization(self):
        """Test MCPResponse JSON serialization."""
        response = MCPResponse(result={"data": "test_data"}, id="test_id")

        expected_dict = {
            "jsonrpc": "2.0",
            "result": {"data": "test_data"},
            "error": None,
            "id": "test_id",
        }

        self.assertEqual(response.dict(), expected_dict)


class TestBaseAdapter(unittest.IsolatedAsyncioTestCase):
    """Test cases for BaseAdapter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.module = Module(
            id=uuid4(),
            name="test_module",
            module_type=ModuleType.VECTOR_STORE,
            config={"test_config": "value"},
            mcp_server_url="http://localhost:8000",
            supported_operations=["test_operation", "error_operation"],
        )

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    def test_adapter_initialization(self, mock_get_logger, mock_get_config, mock_validate_url):
        """Test BaseAdapter initialization."""
        mock_validate_url.return_value = True
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        adapter = ConcreteAdapter(self.module)

        self.assertEqual(adapter.module, self.module)
        self.assertIsNone(adapter.client)
        self.assertFalse(adapter._connected)
        self.assertIsNone(adapter._last_health_check)
        mock_validate_url.assert_called_once_with(self.module.mcp_server_url)

    @patch("src.adapters.base_adapter.validate_url")
    def test_adapter_initialization_invalid_url(self, mock_validate_url):
        """Test BaseAdapter initialization with invalid URL."""
        mock_validate_url.return_value = False

        with self.assertRaises(ValueError) as context:
            ConcreteAdapter(self.module)

        self.assertIn("Invalid MCP server URL", str(context.exception))

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    async def test_async_context_manager(self, mock_get_logger, mock_get_config, mock_validate_url):
        """Test BaseAdapter as async context manager."""
        mock_validate_url.return_value = True
        mock_config = MagicMock()
        mock_config.mcp.timeout_seconds = 30
        mock_config.mcp.max_connections = 10
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        with patch.object(ConcreteAdapter, "_test_connection") as mock_test_connection:
            mock_test_connection.return_value = None

            async with ConcreteAdapter(self.module) as adapter:
                self.assertTrue(adapter._connected)
                self.assertIsNotNone(adapter.client)

            # After exiting context, should be disconnected
            self.assertFalse(adapter._connected)
            self.assertIsNone(adapter.client)

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    @patch("src.adapters.base_adapter.httpx.AsyncClient")
    async def test_connect_success(
        self, mock_client_class, mock_get_logger, mock_get_config, mock_validate_url
    ):
        """Test successful connection to MCP server."""
        mock_validate_url.return_value = True
        mock_config = MagicMock()
        mock_config.mcp.timeout_seconds = 30
        mock_config.mcp.max_connections = 10
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        adapter = ConcreteAdapter(self.module)

        with patch.object(adapter, "_test_connection") as mock_test_connection:
            mock_test_connection.return_value = None

            await adapter.connect()

            self.assertTrue(adapter._connected)
            self.assertEqual(adapter.client, mock_client)
            self.assertEqual(adapter.module.status, ModuleStatus.ACTIVE)
            mock_test_connection.assert_called_once()

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    async def test_connect_failure(self, mock_get_logger, mock_get_config, mock_validate_url):
        """Test connection failure."""
        mock_validate_url.return_value = True
        mock_config = MagicMock()
        mock_config.mcp.timeout_seconds = 30
        mock_config.mcp.max_connections = 10
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        adapter = ConcreteAdapter(self.module)

        with patch.object(adapter, "_test_connection") as mock_test_connection:
            mock_test_connection.side_effect = Exception("Connection failed")

            with self.assertRaises(ConnectionError) as context:
                await adapter.connect()

            self.assertIn("Failed to connect", str(context.exception))
            self.assertFalse(adapter._connected)
            self.assertEqual(adapter.module.status, ModuleStatus.ERROR)

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    async def test_disconnect(self, mock_get_logger, mock_get_config, mock_validate_url):
        """Test disconnection from MCP server."""
        mock_validate_url.return_value = True
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        adapter = ConcreteAdapter(self.module)
        mock_client = AsyncMock()
        adapter.client = mock_client
        adapter._connected = True

        await adapter.disconnect()

        self.assertFalse(adapter._connected)
        self.assertIsNone(adapter.client)
        self.assertEqual(adapter.module.status, ModuleStatus.INACTIVE)
        mock_client.aclose.assert_called_once()

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    async def test_test_connection_success(
        self, mock_get_logger, mock_get_config, mock_validate_url
    ):
        """Test successful connection test."""
        mock_validate_url.return_value = True
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        adapter = ConcreteAdapter(self.module)
        mock_client = AsyncMock()
        adapter.client = mock_client

        # Mock successful health check response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.elapsed = MagicMock()
        mock_response.elapsed.total_seconds.return_value = 0.1
        mock_client.get.return_value = mock_response

        await adapter._test_connection()

        expected_url = f"{self.module.mcp_server_url}/health"
        mock_client.get.assert_called_once_with(expected_url)

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    async def test_test_connection_no_client(
        self, mock_get_logger, mock_get_config, mock_validate_url
    ):
        """Test connection test without client."""
        mock_validate_url.return_value = True
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        adapter = ConcreteAdapter(self.module)
        adapter.client = None

        with self.assertRaises(ConnectionError) as context:
            await adapter._test_connection()

        self.assertIn("Client not initialized", str(context.exception))

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    async def test_test_connection_invalid_url(
        self, mock_get_logger, mock_get_config, mock_validate_url
    ):
        """Test connection test with invalid URL at runtime."""
        # Initially valid, then invalid
        mock_validate_url.side_effect = [True, False]
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        adapter = ConcreteAdapter(self.module)
        mock_client = AsyncMock()
        adapter.client = mock_client

        with self.assertRaises(ConnectionError) as context:
            await adapter._test_connection()

        self.assertIn("Invalid MCP server URL", str(context.exception))

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    async def test_send_request_success(self, mock_get_logger, mock_get_config, mock_validate_url):
        """Test successful MCP request."""
        mock_validate_url.return_value = True
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        adapter = ConcreteAdapter(self.module)
        mock_client = AsyncMock()
        adapter.client = mock_client
        adapter._connected = True

        # Mock successful response
        mock_http_response = MagicMock()
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {"data": "test_result"},
            "id": "test_id",
        }
        mock_client.post.return_value = mock_http_response

        response = await adapter.send_request("test_method", {"param": "value"})

        self.assertIsInstance(response, MCPResponse)
        self.assertEqual(response.result, {"data": "test_result"})
        self.assertIsNone(response.error)

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    async def test_send_request_not_connected(
        self, mock_get_logger, mock_get_config, mock_validate_url
    ):
        """Test MCP request when not connected."""
        mock_validate_url.return_value = True
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        adapter = ConcreteAdapter(self.module)
        adapter._connected = False

        with self.assertRaises(ConnectionError) as context:
            await adapter.send_request("test_method", {"param": "value"})

        self.assertIn("Not connected to MCP server", str(context.exception))

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    async def test_send_request_mcp_error(
        self, mock_get_logger, mock_get_config, mock_validate_url
    ):
        """Test MCP request with MCP error response."""
        mock_validate_url.return_value = True
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        adapter = ConcreteAdapter(self.module)
        mock_client = AsyncMock()
        adapter.client = mock_client
        adapter._connected = True

        # Mock error response
        mock_http_response = MagicMock()
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = {
            "jsonrpc": "2.0",
            "error": {"code": -1, "message": "Test MCP error"},
            "id": "test_id",
        }
        mock_client.post.return_value = mock_http_response

        with self.assertRaises(OperationError) as context:
            await adapter.send_request("test_method", {"param": "value"})

        self.assertIn("MCP error", str(context.exception))

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    async def test_health_check_success(self, mock_get_logger, mock_get_config, mock_validate_url):
        """Test successful health check."""
        mock_validate_url.return_value = True
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        adapter = ConcreteAdapter(self.module)

        with patch.object(adapter, "_test_connection") as mock_test_connection:
            mock_test_connection.return_value = None

            health_info = await adapter.health_check()

            self.assertEqual(health_info["status"], "healthy")
            self.assertIn("response_time_ms", health_info)
            self.assertEqual(health_info["connected"], adapter._connected)
            self.assertEqual(health_info["module_id"], str(self.module.id))
            self.assertEqual(health_info["module_type"], self.module.module_type.value)
            self.assertIn("test_detail", health_info)
            self.assertEqual(health_info["test_detail"], "healthy")

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    async def test_health_check_failure(self, mock_get_logger, mock_get_config, mock_validate_url):
        """Test health check failure."""
        mock_validate_url.return_value = True
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        adapter = ConcreteAdapter(self.module)

        with patch.object(adapter, "_test_connection") as mock_test_connection:
            mock_test_connection.side_effect = Exception("Health check failed")

            health_info = await adapter.health_check()

            self.assertEqual(health_info["status"], "unhealthy")
            self.assertIn("error", health_info)
            self.assertEqual(health_info["connected"], False)
            self.assertIn("response_time_ms", health_info)

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    def test_is_connected(self, mock_get_logger, mock_get_config, mock_validate_url):
        """Test connection status check."""
        mock_validate_url.return_value = True
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        adapter = ConcreteAdapter(self.module)

        self.assertFalse(adapter.is_connected())

        adapter._connected = True
        self.assertTrue(adapter.is_connected())

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    def test_get_supported_operations(self, mock_get_logger, mock_get_config, mock_validate_url):
        """Test getting supported operations."""
        mock_validate_url.return_value = True
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        adapter = ConcreteAdapter(self.module)
        operations = adapter.get_supported_operations()

        self.assertEqual(operations, ["test_operation", "error_operation"])
        # Ensure it returns a copy
        operations.append("new_operation")
        self.assertEqual(adapter.get_supported_operations(), ["test_operation", "error_operation"])

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    async def test_validate_parameters(self, mock_get_logger, mock_get_config, mock_validate_url):
        """Test parameter validation."""
        mock_validate_url.return_value = True
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        adapter = ConcreteAdapter(self.module)

        # Default implementation should return True
        result = await adapter.validate_parameters("test_operation", {"param": "value"})
        self.assertTrue(result)

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    async def test_batch_execute_success(self, mock_get_logger, mock_get_config, mock_validate_url):
        """Test successful batch operation execution."""
        mock_validate_url.return_value = True
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        adapter = ConcreteAdapter(self.module)

        operations = [
            {"operation": "test_operation", "parameters": {"param1": "value1"}},
            {"operation": "test_operation", "parameters": {"param2": "value2"}},
        ]

        results = await adapter.batch_execute(operations)

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["success"])
        self.assertEqual(results[0]["result"]["result"], "success")
        self.assertTrue(results[1]["success"])
        self.assertEqual(results[1]["result"]["result"], "success")

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    async def test_batch_execute_with_errors(
        self, mock_get_logger, mock_get_config, mock_validate_url
    ):
        """Test batch execution with some operations failing."""
        mock_validate_url.return_value = True
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        adapter = ConcreteAdapter(self.module)

        operations = [
            {"operation": "test_operation", "parameters": {"param1": "value1"}},
            {"operation": "error_operation", "parameters": {}},
        ]

        results = await adapter.batch_execute(operations)

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["success"])
        self.assertFalse(results[1]["success"])
        self.assertIn("error", results[1])

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    async def test_stream_execute(self, mock_get_logger, mock_get_config, mock_validate_url):
        """Test streaming operation execution."""
        mock_validate_url.return_value = True
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        adapter = ConcreteAdapter(self.module)

        results = []
        async for result in adapter.stream_execute("test_operation", {"param": "value"}):
            results.append(result)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["result"], "success")

    @patch("src.adapters.base_adapter.validate_url")
    @patch("src.adapters.base_adapter.get_config")
    @patch("src.adapters.base_adapter.get_logger")
    async def test_concrete_adapter_abstract_methods(
        self, mock_get_logger, mock_get_config, mock_validate_url
    ):
        """Test concrete implementation of abstract methods."""
        mock_validate_url.return_value = True
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        adapter = ConcreteAdapter(self.module)

        # Test execute_operation
        result1 = await adapter.execute_operation("test_operation", {"param": "value"})
        self.assertEqual(result1["result"], "success")

        with self.assertRaises(OperationError):
            await adapter.execute_operation("error_operation", {})

        result2 = await adapter.execute_operation("unknown_operation", {})
        self.assertEqual(result2["result"], "unknown_operation")

        # Test get_schema
        schema = await adapter.get_schema()
        self.assertIn("operations", schema)
        self.assertIn("test_operation", schema["operations"])

        # Test _get_health_details
        health_details = await adapter._get_health_details()
        self.assertEqual(health_details["test_detail"], "healthy")
        self.assertEqual(health_details["custom_metric"], 100)


class TestAdapterExceptions(unittest.TestCase):
    """Test cases for adapter exception classes."""

    def test_adapter_error(self):
        """Test AdapterError exception."""
        error = AdapterError("Test adapter error")
        self.assertEqual(str(error), "Test adapter error")
        self.assertIsInstance(error, Exception)

    def test_connection_error(self):
        """Test ConnectionError exception."""
        error = ConnectionError("Test connection error")
        self.assertEqual(str(error), "Test connection error")
        self.assertIsInstance(error, AdapterError)

    def test_operation_error(self):
        """Test OperationError exception."""
        error = OperationError("Test operation error")
        self.assertEqual(str(error), "Test operation error")
        self.assertIsInstance(error, AdapterError)


if __name__ == "__main__":
    unittest.main()
