"""MCP-RAG-Control: Agent-based RAG system with map-based control architecture."""

__version__ = "0.1.0"
__author__ = "MCP-RAG-Control Team"
__email__ = "team@mcp-rag-control.com"

from .models import Module, Pipeline
from .adapters import BaseAdapter, VectorAdapter, DatabaseAdapter
from .registry import ModuleRegistry, PipelineRegistry

__all__ = [
    "Module",
    "Pipeline", 
    "BaseAdapter",
    "VectorAdapter", 
    "DatabaseAdapter",
    "ModuleRegistry",
    "PipelineRegistry",
]