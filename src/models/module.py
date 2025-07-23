"""Module schema definitions for MCP-RAG-Control system."""

from datetime import datetime, UTC
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DataType(str, Enum):
    """Data types in the RAG system."""
    
    TEXT = "text"
    EMBEDDINGS = "embeddings"
    VECTORS = "vectors"
    DOCUMENTS = "documents"
    CONTEXT = "context"
    RESPONSE = "response"
    METADATA = "metadata"
    MULTIMODAL = "multimodal"  # 이미지, 오디오 등
    STRUCTURED = "structured"  # JSON, 테이블 등
    STREAM = "stream"  # 스트리밍 데이터


class ModuleType(str, Enum):
    """Module types based on data transformation patterns in RAG systems."""
    
    # Input Processing Modules (Text → Text/Structured)
    TEXT_PREPROCESSOR = "text_preprocessor"  # 텍스트 정제, 분할, 정규화
    QUERY_ANALYZER = "query_analyzer"        # 질문 분석, 의도 파악
    TEXT_SPLITTER = "text_splitter"          # 문서 분할
    
    # Embedding Modules (Text → Embeddings)
    EMBEDDING_ENCODER = "embedding_encoder"   # 텍스트를 벡터로 변환
    RERANKING_ENCODER = "reranking_encoder"   # 재순위화용 임베딩
    
    # Vector Operations (Embeddings → Vectors/Documents)
    VECTOR_STORE = "vector_store"            # 벡터 저장 및 검색
    SIMILARITY_SEARCH = "similarity_search"  # 유사도 검색
    VECTOR_INDEX = "vector_index"            # 벡터 인덱싱
    
    # Document Processing (Documents → Documents/Context)
    DOCUMENT_LOADER = "document_loader"      # 문서 로딩
    DOCUMENT_FILTER = "document_filter"      # 문서 필터링
    DOCUMENT_RANKER = "document_ranker"      # 문서 재순위화
    CONTEXT_BUILDER = "context_builder"      # 컨텍스트 구성
    
    # Generation Modules (Context+Query → Response)
    LLM_GENERATOR = "llm_generator"          # LLM 응답 생성
    PROMPT_TEMPLATE = "prompt_template"      # 프롬프트 템플릿
    RESPONSE_FORMATTER = "response_formatter" # 응답 포매팅
    
    # Memory and State (Any → Any with persistence)
    MEMORY_STORE = "memory_store"            # 대화 기록 저장
    CACHE_MANAGER = "cache_manager"          # 캐싱 관리
    SESSION_MANAGER = "session_manager"      # 세션 관리
    
    # Data Sources (External → Text/Documents)
    WEB_SCRAPER = "web_scraper"              # 웹 스크래핑
    DATABASE_CONNECTOR = "database_connector" # 데이터베이스 연결
    API_CONNECTOR = "api_connector"          # API 연결
    FILE_LOADER = "file_loader"              # 파일 로딩
    
    # Multimodal Support (Multimodal → Text/Embeddings)
    IMAGE_PROCESSOR = "image_processor"      # 이미지 처리
    AUDIO_PROCESSOR = "audio_processor"      # 오디오 처리
    OCR_PROCESSOR = "ocr_processor"          # OCR 처리
    
    # Quality and Evaluation (Any → Metadata)
    QUALITY_CHECKER = "quality_checker"      # 품질 검사
    BIAS_DETECTOR = "bias_detector"          # 편향 탐지
    FACTUALITY_CHECKER = "factuality_checker" # 사실 확인
    
    # Flow Control (Any → Any with logic)
    CONDITIONAL_ROUTER = "conditional_router" # 조건부 라우팅
    PARALLEL_PROCESSOR = "parallel_processor" # 병렬 처리
    AGGREGATOR = "aggregator"               # 결과 집계
    
    # Stream Processing (Stream → Stream)
    STREAM_PROCESSOR = "stream_processor"    # 실시간 스트림 처리
    STREAM_AGGREGATOR = "stream_aggregator"  # 스트림 집계
    
    # Advanced RAG (Any → Any with AI)
    QUERY_EXPANDER = "query_expander"        # 쿼리 확장 (Text → Text)
    ANSWER_SYNTHESIZER = "answer_synthesizer" # 답변 합성 (Multiple Context → Response)
    CITATION_GENERATOR = "citation_generator" # 인용 생성 (Response → Response)
    
    # Custom and Legacy
    CUSTOM_TRANSFORM = "custom_transform"    # 사용자 정의 변환
    LEGACY_ADAPTER = "legacy_adapter"        # 기존 시스템 어댑터


class ModuleStatus(str, Enum):
    """Status of a module."""
    
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"


class DataSchema(BaseModel):
    """Schema definition for data flowing through modules."""
    
    data_type: DataType = Field(..., description="Primary data type")
    format: str = Field(..., description="Specific format (e.g., 'json', 'plain_text', 'vector_array')")
    
    # Schema details
    required_fields: List[str] = Field(default_factory=list)
    optional_fields: List[str] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    
    # For vector data
    vector_dimension: Optional[int] = Field(None, description="Vector dimension for embedding data")
    distance_metric: Optional[str] = Field(None, description="Distance metric for vector operations")
    
    # For text data
    max_length: Optional[int] = Field(None, description="Maximum text length")
    encoding: Optional[str] = Field(default="utf-8", description="Text encoding")
    language: Optional[str] = Field(None, description="Expected language")
    
    # For structured data
    schema_definition: Optional[Dict[str, Any]] = Field(None, description="JSON schema or structure definition")
    
    # Validation rules
    validation_rules: List[str] = Field(default_factory=list)


class ModuleCapabilities(BaseModel):
    """Capabilities and transformation patterns of a module."""
    
    # Primary transformation
    input_schema: DataSchema = Field(..., description="Input data schema")
    output_schema: DataSchema = Field(..., description="Output data schema")
    
    # Additional inputs (for modules that need multiple inputs)
    auxiliary_inputs: Dict[str, DataSchema] = Field(default_factory=dict)
    
    # Transformation type
    transformation_type: str = Field(..., description="Type of transformation performed")
    is_stateful: bool = Field(default=False, description="Whether module maintains state")
    supports_streaming: bool = Field(default=False, description="Whether module supports streaming")
    supports_batch: bool = Field(default=True, description="Whether module supports batch processing")
    
    # Performance characteristics
    expected_latency_ms: Optional[float] = Field(None, description="Expected processing latency")
    max_batch_size: Optional[int] = Field(None, description="Maximum batch size")
    memory_requirements_mb: Optional[float] = Field(None, description="Memory requirements")
    
    # Quality and reliability
    error_rate_threshold: float = Field(default=0.05, description="Acceptable error rate")
    supports_fallback: bool = Field(default=False, description="Whether module has fallback behavior")
    
    # Monitoring
    metrics_collected: List[str] = Field(default_factory=list, description="Metrics this module collects")


class ModuleConfig(BaseModel):
    """Configuration for a module."""
    
    # Connection settings
    host: Optional[str] = None
    port: Optional[int] = None
    database_name: Optional[str] = None
    collection_name: Optional[str] = None
    api_key: Optional[str] = None
    
    # Model/Algorithm settings
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    dimensions: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    
    # Processing settings
    batch_size: Optional[int] = None
    timeout_seconds: Optional[float] = None
    retry_attempts: Optional[int] = None
    
    # Quality settings
    similarity_threshold: Optional[float] = None
    min_relevance_score: Optional[float] = None
    max_results: Optional[int] = None
    
    # Custom parameters
    custom_params: Dict[str, Any] = Field(default_factory=dict)
    
    # Resource limits
    max_memory_mb: Optional[float] = None
    max_cpu_percent: Optional[float] = None


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
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_health_check: Optional[datetime] = None
    
    # Capabilities (새로운 데이터 플로우 기반)
    capabilities: Optional[ModuleCapabilities] = Field(None, description="Module transformation capabilities")
    
    # Legacy schema (호환성을 위해 유지)
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
        self.updated_at = datetime.now(UTC)
    
    def add_dependency(self, module_id: UUID) -> None:
        """Add a dependency to this module."""
        if module_id not in self.dependencies:
            self.dependencies.append(module_id)
            self.updated_at = datetime.now(UTC)
    
    def remove_dependency(self, module_id: UUID) -> None:
        """Remove a dependency from this module."""
        if module_id in self.dependencies:
            self.dependencies.remove(module_id)
            self.updated_at = datetime.now(UTC)


# API Request/Response Models
class ModuleRegistrationRequest(BaseModel):
    """Request model for registering a new module."""
    
    name: str = Field(..., description="Human-readable module name")
    module_type: ModuleType = Field(..., description="Type of the module")
    description: Optional[str] = Field(None, description="Module description")
    version: str = Field(default="1.0.0", description="Module version")
    mcp_server_url: str = Field(..., description="MCP server URL")
    mcp_protocol_version: str = Field(default="1.0", description="MCP protocol version")
    config: ModuleConfig = Field(default_factory=ModuleConfig)
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    supported_operations: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list, description="Dependency module IDs as strings")
    tags: List[str] = Field(default_factory=list)


class ModuleUpdateRequest(BaseModel):
    """Request model for updating an existing module."""
    
    name: Optional[str] = Field(None, description="Human-readable module name")
    description: Optional[str] = Field(None, description="Module description")
    version: Optional[str] = Field(None, description="Module version")
    status: Optional[ModuleStatus] = Field(None, description="Module status")
    config: Optional[ModuleConfig] = Field(None, description="Module configuration")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="Input schema")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="Output schema")
    supported_operations: Optional[List[str]] = Field(None, description="Supported operations")
    dependencies: Optional[List[str]] = Field(None, description="Dependency module IDs as strings")
    tags: Optional[List[str]] = Field(None, description="Module tags")


class ModuleResponse(BaseModel):
    """Response model for module operations."""
    
    module: Module
    message: str = Field(default="Operation completed successfully")


class ModuleListResponse(BaseModel):
    """Response model for listing modules."""
    
    modules: List[Module]
    total: int
    message: str = Field(default="Modules retrieved successfully")


class ModuleHealthCheck(BaseModel):
    """Health check result for a module."""
    
    module_id: UUID
    status: ModuleStatus
    response_time_ms: float
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }