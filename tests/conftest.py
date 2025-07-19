"""Pytest configuration and fixtures for MCP-RAG-Control tests."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from src.mcp_rag_control.models.module import Module, ModuleType, ModuleConfig
from src.mcp_rag_control.utils.config import AppConfig


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return AppConfig(
        debug=True,
        environment="test"
    )


@pytest.fixture
def sample_module():
    """Create a sample module for testing."""
    return Module(
        id=uuid4(),
        name="test_module",
        module_type=ModuleType.VECTOR_STORE,
        description="Test module for unit tests",
        mcp_server_url="https://test.example.com/mcp",
        config=ModuleConfig(
            host="localhost",
            port=8080,
            database_name="test_db"
        )
    )


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for testing."""
    client = AsyncMock()
    
    # Mock successful health check
    health_response = MagicMock()
    health_response.status_code = 200
    health_response.raise_for_status.return_value = None
    health_response.elapsed = None
    
    client.get.return_value = health_response
    client.post.return_value = MagicMock()
    client.aclose.return_value = None
    
    return client