"""MCP adapters for connecting to various external systems."""

from .base_adapter import BaseAdapter, AdapterError, ConnectionError
from .vector_adapter import VectorAdapter
from .database_adapter import DatabaseAdapter

__all__ = [
    "BaseAdapter",
    "AdapterError", 
    "ConnectionError",
    "VectorAdapter",
    "DatabaseAdapter",
]