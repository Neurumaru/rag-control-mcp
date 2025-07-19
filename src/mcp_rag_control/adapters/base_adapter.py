"""Base adapter interface for MCP-RAG-Control system."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx
from pydantic import BaseModel

from ..models.module import Module, ModuleStatus


class AdapterError(Exception):
    """Base exception for adapter errors."""
    pass


class ConnectionError(AdapterError):
    """Error connecting to external system."""
    pass


class OperationError(AdapterError):
    """Error performing operation."""
    pass


class MCPRequest(BaseModel):
    """MCP protocol request."""
    
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any]
    id: str


class MCPResponse(BaseModel):
    """MCP protocol response."""
    
    jsonrpc: str = "2.0"
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    id: str


class BaseAdapter(ABC):
    """Base adapter for MCP communication."""
    
    def __init__(self, module: Module):
        """Initialize adapter with module configuration."""
        self.module = module
        self.client: Optional[httpx.AsyncClient] = None
        self._connected = False
        self._last_health_check = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        try:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_connections=10)
            )
            
            # Test connection with health check
            await self._test_connection()
            self._connected = True
            self.module.update_status(ModuleStatus.ACTIVE)
            
        except Exception as e:
            self.module.update_status(ModuleStatus.ERROR)
            raise ConnectionError(f"Failed to connect to {self.module.mcp_server_url}: {str(e)}")
    
    async def disconnect(self) -> None:
        """Close connection to the MCP server."""
        if self.client:
            await self.client.aclose()
            self.client = None
        
        self._connected = False
        self.module.update_status(ModuleStatus.INACTIVE)
    
    async def _test_connection(self) -> None:
        """Test connection to MCP server."""
        if not self.client:
            raise ConnectionError("Client not initialized")
        
        try:
            response = await self.client.get(f"{self.module.mcp_server_url}/health")
            response.raise_for_status()
        except httpx.RequestError as e:
            raise ConnectionError(f"Connection test failed: {str(e)}")
    
    async def send_request(self, method: str, params: Dict[str, Any]) -> MCPResponse:
        """Send MCP request and return response."""
        if not self._connected or not self.client:
            raise ConnectionError("Not connected to MCP server")
        
        request = MCPRequest(
            method=method,
            params=params,
            id=f"req_{int(time.time())}"
        )
        
        try:
            response = await self.client.post(
                f"{self.module.mcp_server_url}/mcp",
                json=request.dict(),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            mcp_response = MCPResponse(**response.json())
            
            if mcp_response.error:
                raise OperationError(f"MCP error: {mcp_response.error}")
            
            return mcp_response
            
        except httpx.RequestError as e:
            raise OperationError(f"Request failed: {str(e)}")
        except Exception as e:
            raise OperationError(f"Unexpected error: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the adapter."""
        start_time = time.time()
        
        try:
            await self._test_connection()
            response_time = (time.time() - start_time) * 1000
            
            health_info = {
                "status": "healthy",
                "response_time_ms": response_time,
                "connected": self._connected,
                "module_id": str(self.module.id),
                "module_type": self.module.module_type.value,
                "last_check": time.time()
            }
            
            # Add adapter-specific health info
            health_info.update(await self._get_health_details())
            
            self._last_health_check = time.time()
            return health_info
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": (time.time() - start_time) * 1000,
                "connected": False,
                "module_id": str(self.module.id),
                "last_check": time.time()
            }
    
    @abstractmethod
    async def _get_health_details(self) -> Dict[str, Any]:
        """Get adapter-specific health details."""
        pass
    
    @abstractmethod
    async def execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an operation on the external system."""
        pass
    
    @abstractmethod
    async def get_schema(self) -> Dict[str, Any]:
        """Get the schema for this adapter."""
        pass
    
    def is_connected(self) -> bool:
        """Check if adapter is connected."""
        return self._connected
    
    def get_supported_operations(self) -> List[str]:
        """Get list of supported operations."""
        return self.module.supported_operations.copy()
    
    async def validate_parameters(self, operation: str, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for an operation."""
        # Default implementation - can be overridden
        return True
    
    async def batch_execute(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple operations in batch."""
        results = []
        
        for op in operations:
            try:
                result = await self.execute_operation(
                    op.get("operation", ""),
                    op.get("parameters", {})
                )
                results.append({
                    "success": True,
                    "result": result,
                    "operation": op.get("operation")
                })
            except Exception as e:
                results.append({
                    "success": False,
                    "error": str(e),
                    "operation": op.get("operation")
                })
        
        return results
    
    async def stream_execute(self, operation: str, parameters: Dict[str, Any]):
        """Execute operation with streaming response."""
        # Default implementation yields single result
        result = await self.execute_operation(operation, parameters)
        yield result