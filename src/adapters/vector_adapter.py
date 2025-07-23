"""Vector database adapter for MCP-RAG-Control system."""

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base_adapter import BaseAdapter, OperationError


class VectorAdapter(BaseAdapter):
    """Adapter for vector database operations via MCP."""
    
    async def _get_health_details(self) -> Dict[str, Any]:
        """Get vector database specific health details."""
        try:
            # Get collection info
            response = await self.send_request("vector.info", {
                "collection_name": self.module.config.collection_name
            })
            
            return {
                "collection_name": self.module.config.collection_name,
                "dimensions": self.module.config.dimensions,
                "index_type": response.result.get("index_type"),
                "total_vectors": response.result.get("total_vectors", 0),
                "index_size_mb": response.result.get("index_size_mb", 0)
            }
        except Exception:
            return {"collection_name": self.module.config.collection_name}
    
    async def execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute vector database operation."""
        if operation not in self.get_supported_operations():
            raise OperationError(f"Unsupported operation: {operation}")
        
        method_map = {
            "search": self._search_vectors,
            "add": self._add_vectors,
            "delete": self._delete_vectors,
            "update": self._update_vectors,
            "get": self._get_vectors,
            "create_collection": self._create_collection,
            "delete_collection": self._delete_collection,
            "list_collections": self._list_collections,
            "get_collection_info": self._get_collection_info,
            "validate": self._validate_vector
        }
        
        handler = method_map.get(operation)
        if not handler:
            raise OperationError(f"Operation handler not found: {operation}")
        
        return await handler(parameters)
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get vector adapter schema."""
        return {
            "type": "vector_database",
            "operations": {
                "search": {
                    "description": "Search for similar vectors",
                    "parameters": {
                        "query_vector": {"type": "array", "description": "Query vector"},
                        "top_k": {"type": "integer", "default": 10},
                        "threshold": {"type": "number", "optional": True},
                        "filter": {"type": "object", "optional": True}
                    },
                    "returns": {
                        "results": {"type": "array", "description": "Search results with scores"}
                    }
                },
                "add": {
                    "description": "Add vectors to collection",
                    "parameters": {
                        "vectors": {"type": "array", "description": "Vectors to add"},
                        "ids": {"type": "array", "description": "Vector IDs"},
                        "metadata": {"type": "array", "optional": True}
                    },
                    "returns": {
                        "added_count": {"type": "integer"}
                    }
                },
                "delete": {
                    "description": "Delete vectors from collection",
                    "parameters": {
                        "ids": {"type": "array", "description": "Vector IDs to delete"}
                    },
                    "returns": {
                        "deleted_count": {"type": "integer"}
                    }
                },
                "validate": {
                    "description": "Validate vector data",
                    "parameters": {
                        "vector": {"type": "array", "description": "Vector to validate"},
                        "expected_dimensions": {"type": "integer", "optional": True}
                    },
                    "returns": {
                        "valid": {"type": "boolean"},
                        "message": {"type": "string"}
                    }
                }
            }
        }
    
    async def _search_vectors(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Search for similar vectors."""
        query_vector = parameters.get("query_vector")
        if not query_vector:
            raise OperationError("query_vector is required")
        
        # Validate vector dimensions
        if len(query_vector) != self.module.config.dimensions:
            raise OperationError(
                f"Query vector dimensions ({len(query_vector)}) "
                f"don't match collection dimensions ({self.module.config.dimensions})"
            )
        
        mcp_params = {
            "collection_name": self.module.config.collection_name,
            "query_vector": query_vector,
            "top_k": parameters.get("top_k", 10),
            "include_metadata": parameters.get("include_metadata", True)
        }
        
        # Add optional parameters
        if "threshold" in parameters:
            mcp_params["threshold"] = parameters["threshold"]
        
        if "filter" in parameters:
            mcp_params["filter"] = parameters["filter"]
        
        response = await self.send_request("vector.search", mcp_params)
        
        return {
            "results": response.result.get("results", []),
            "total_found": response.result.get("total_found", 0),
            "search_time_ms": response.result.get("search_time_ms", 0)
        }
    
    async def _add_vectors(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Add vectors to collection."""
        vectors = parameters.get("vectors")
        ids = parameters.get("ids")
        
        if not vectors:
            raise OperationError("vectors is required")
        
        if not ids:
            raise OperationError("ids is required")
        
        if len(vectors) != len(ids):
            raise OperationError("Number of vectors must match number of IDs")
        
        # Validate vector dimensions
        for i, vector in enumerate(vectors):
            if len(vector) != self.module.config.dimensions:
                raise OperationError(
                    f"Vector {i} dimensions ({len(vector)}) "
                    f"don't match collection dimensions ({self.module.config.dimensions})"
                )
        
        mcp_params = {
            "collection_name": self.module.config.collection_name,
            "vectors": vectors,
            "ids": ids
        }
        
        if "metadata" in parameters:
            mcp_params["metadata"] = parameters["metadata"]
        
        response = await self.send_request("vector.add", mcp_params)
        
        return {
            "added_count": response.result.get("added_count", 0),
            "failed_ids": response.result.get("failed_ids", [])
        }
    
    async def _delete_vectors(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Delete vectors from collection."""
        ids = parameters.get("ids")
        
        if not ids:
            raise OperationError("ids is required")
        
        mcp_params = {
            "collection_name": self.module.config.collection_name,
            "ids": ids
        }
        
        response = await self.send_request("vector.delete", mcp_params)
        
        return {
            "deleted_count": response.result.get("deleted_count", 0),
            "not_found_ids": response.result.get("not_found_ids", [])
        }
    
    async def _update_vectors(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Update vectors in collection."""
        vectors = parameters.get("vectors")
        ids = parameters.get("ids")
        
        if not vectors:
            raise OperationError("vectors is required")
        
        if not ids:
            raise OperationError("ids is required")
        
        mcp_params = {
            "collection_name": self.module.config.collection_name,
            "vectors": vectors,
            "ids": ids
        }
        
        if "metadata" in parameters:
            mcp_params["metadata"] = parameters["metadata"]
        
        response = await self.send_request("vector.update", mcp_params)
        
        return {
            "updated_count": response.result.get("updated_count", 0),
            "not_found_ids": response.result.get("not_found_ids", [])
        }
    
    async def _get_vectors(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get vectors by IDs."""
        ids = parameters.get("ids")
        
        if not ids:
            raise OperationError("ids is required")
        
        mcp_params = {
            "collection_name": self.module.config.collection_name,
            "ids": ids,
            "include_vectors": parameters.get("include_vectors", True),
            "include_metadata": parameters.get("include_metadata", True)
        }
        
        response = await self.send_request("vector.get", mcp_params)
        
        return {
            "vectors": response.result.get("vectors", []),
            "found_count": response.result.get("found_count", 0),
            "not_found_ids": response.result.get("not_found_ids", [])
        }
    
    async def _create_collection(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new vector collection."""
        collection_name = parameters.get("collection_name")
        dimensions = parameters.get("dimensions")
        
        if not collection_name:
            raise OperationError("collection_name is required")
        
        if not dimensions:
            raise OperationError("dimensions is required")
        
        mcp_params = {
            "collection_name": collection_name,
            "dimensions": dimensions,
            "index_type": parameters.get("index_type", "HNSW"),
            "metric": parameters.get("metric", "cosine")
        }
        
        if "index_config" in parameters:
            mcp_params["index_config"] = parameters["index_config"]
        
        response = await self.send_request("vector.create_collection", mcp_params)
        
        return {
            "collection_name": collection_name,
            "created": response.result.get("created", False)
        }
    
    async def _delete_collection(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a vector collection."""
        collection_name = parameters.get("collection_name", self.module.config.collection_name)
        
        mcp_params = {
            "collection_name": collection_name
        }
        
        response = await self.send_request("vector.delete_collection", mcp_params)
        
        return {
            "collection_name": collection_name,
            "deleted": response.result.get("deleted", False)
        }
    
    async def _list_collections(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """List all vector collections."""
        response = await self.send_request("vector.list_collections", {})
        
        return {
            "collections": response.result.get("collections", []),
            "total_count": response.result.get("total_count", 0)
        }
    
    async def _get_collection_info(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get information about a collection."""
        collection_name = parameters.get("collection_name", self.module.config.collection_name)
        
        mcp_params = {
            "collection_name": collection_name
        }
        
        response = await self.send_request("vector.info", mcp_params)
        
        return response.result
    
    async def similarity_search_with_score(
        self, 
        query_vector: List[float], 
        top_k: int = 10,
        threshold: Optional[float] = None,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """High-level similarity search method."""
        parameters = {
            "query_vector": query_vector,
            "top_k": top_k,
            "include_metadata": True
        }
        
        if threshold is not None:
            parameters["threshold"] = threshold
        
        if filter_criteria:
            parameters["filter"] = filter_criteria
        
        result = await self._search_vectors(parameters)
        return result["results"]
    
    async def add_documents(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """High-level method to add documents with embeddings."""
        if not ids:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{"text": text} for text in texts]
        else:
            # Ensure text is included in metadata
            for i, metadata in enumerate(metadatas):
                metadata["text"] = texts[i]
        
        parameters = {
            "vectors": embeddings,
            "ids": ids,
            "metadata": metadatas
        }
        
        return await self._add_vectors(parameters)
    
    async def _validate_vector(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate vector data through MCP interface."""
        vector = parameters.get("vector")
        expected_dimensions = parameters.get("expected_dimensions")
        
        if not vector:
            return {"valid": False, "message": "Vector is required"}
        
        if not isinstance(vector, list):
            return {"valid": False, "message": "Vector must be a list"}
        
        # Check dimensions
        if expected_dimensions and len(vector) != expected_dimensions:
            return {
                "valid": False, 
                "message": f"Vector dimensions ({len(vector)}) don't match expected ({expected_dimensions})"
            }
        
        # Check for invalid values
        try:
            for i, val in enumerate(vector):
                if not isinstance(val, (int, float)):
                    return {"valid": False, "message": f"Vector element {i} is not a number"}
                if not isinstance(val, (int, float)) or val != val:  # NaN check
                    return {"valid": False, "message": f"Vector element {i} is NaN"}
                if val == float('inf') or val == float('-inf'):
                    return {"valid": False, "message": f"Vector element {i} is infinite"}
        except Exception as e:
            return {"valid": False, "message": f"Vector validation error: {str(e)}"}
        
        # Send validation request to MCP server
        mcp_params = {
            "vector": vector,
            "collection_name": self.module.config.collection_name
        }
        
        if expected_dimensions:
            mcp_params["expected_dimensions"] = expected_dimensions
        
        try:
            response = await self.send_request("vector.validate", mcp_params)
            return response.result
        except Exception:
            # Fallback to local validation if MCP server doesn't support validation
            return {"valid": True, "message": "Vector passed local validation"}