"""Module schema definitions for MCP-RAG-Control system."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ModuleType(str, Enum):
    """Types of modules in the system."""
    
    VECTOR_STORE = "vector_store"
    DATABASE = "database" 
    EMBEDDING = "embedding"
    LLM = "llm"
    RETRIEVER = "retriever"
    PROCESSOR = "processor"
    FILTER = "filter"
    TRANSFORM = "transform"


class ModuleStatus(str, Enum):
    """Status of a module."""
    
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"


class ModuleConfig(BaseModel):
    """Configuration for a module."""
    
    host: Optional[str] = None
    port: Optional[int] = None
    database_name: Optional[str] = None
    collection_name: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    dimensions: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    custom_params: Dict[str, Any] = Field(default_factory=dict)


class Module(BaseModel):
    """Module definition for MCP-RAG-Control system."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Human-readable module name")
    module_type: ModuleType = Field(..., description="Type of the module")
    description: Optional[str] = Field(None, description="Module description")
    version: str = Field(default="1.0.0", description="Module version")
    
    # MCP connection details
    mcp_server_url: str = Field(..., description="MCP server URL")
    mcp_protocol_version: str = Field(default="1.0", description="MCP protocol version")
    
    # Configuration
    config: ModuleConfig = Field(default_factory=ModuleConfig)
    
    # Status and metadata
    status: ModuleStatus = Field(default=ModuleStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_health_check: Optional[datetime] = None
    
    # Capabilities
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    supported_operations: List[str] = Field(default_factory=list)
    
    # Dependencies
    dependencies: List[UUID] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    
    def update_status(self, status: ModuleStatus) -> None:
        """Update module status and timestamp."""
        self.status = status
        self.updated_at = datetime.utcnow()
    
    def add_dependency(self, module_id: UUID) -> None:
        """Add a dependency to this module."""
        if module_id not in self.dependencies:
            self.dependencies.append(module_id)
            self.updated_at = datetime.utcnow()
    
    def remove_dependency(self, module_id: UUID) -> None:
        """Remove a dependency from this module."""
        if module_id in self.dependencies:
            self.dependencies.remove(module_id)
            self.updated_at = datetime.utcnow()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to this module."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from this module."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()


class ModuleHealthCheck(BaseModel):
    """Health check result for a module."""
    
    module_id: UUID
    status: ModuleStatus
    response_time_ms: float
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }