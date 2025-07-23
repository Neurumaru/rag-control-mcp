"""MCP adapters for connecting to various external systems."""

from .base_adapter import AdapterError, BaseAdapter, ConnectionError
from .database_adapter import DatabaseAdapter
from .vector_adapter import VectorAdapter

__all__ = [
    "BaseAdapter",
    "AdapterError",
    "ConnectionError",
    "VectorAdapter",
    "DatabaseAdapter",
]
