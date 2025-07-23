"""Routing decision schemas and models for intelligent RAG routing."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

from .module import DataType, ModuleType


class QueryType(str, Enum):
    """Types of user queries based on content analysis."""

    GREETING = "greeting"  # 인사말, 단순 대화
    FACTUAL_SIMPLE = "factual_simple"  # 간단한 사실 질문
    FACTUAL_COMPLEX = "factual_complex"  # 복잡한 사실 질문
    PROCEDURAL = "procedural"  # 절차/방법 질문
    ANALYTICAL = "analytical"  # 분석/비교 질문
    CREATIVE = "creative"  # 창작/아이디어 질문
    TECHNICAL = "technical"  # 기술적 질문
    CONVERSATIONAL = "conversational"  # 일반 대화
    DOMAIN_SPECIFIC = "domain_specific"  # 특정 도메인 지식 필요
    MULTIMODAL = "multimodal"  # 멀티모달 입력


class QueryComplexity(str, Enum):
    """Query complexity levels for routing decisions."""

    TRIVIAL = "trivial"  # 즉시 답 가능 (예: 인사)
    SIMPLE = "simple"  # 기본 지식으로 답 가능
    MODERATE = "moderate"  # 일부 추론 필요
    COMPLEX = "complex"  # 복잡한 추론/검색 필요
    EXPERT = "expert"  # 전문 지식 필요


class KnowledgeRequirement(str, Enum):
    """External knowledge requirement levels."""

    NONE = "none"  # 외부 지식 불필요
    MINIMAL = "minimal"  # 최소한의 검증만 필요
    MODERATE = "moderate"  # 일반적인 RAG 검색 필요
    EXTENSIVE = "extensive"  # 광범위한 검색 필요
    SPECIALIZED = "specialized"  # 전문 데이터베이스 필요


class RoutingPath(str, Enum):
    """Available routing paths in the system."""

    DIRECT_LLM = "direct_llm"  # LLM 직접 응답
    SIMPLE_RAG = "simple_rag"  # 기본 RAG 파이프라인
    ENHANCED_RAG = "enhanced_rag"  # 고급 RAG (재순위화 포함)
    HYBRID_PARALLEL = "hybrid_parallel"  # 병렬 처리 후 집계
    MULTI_STEP = "multi_step"  # 다단계 검색 및 추론
    SPECIALIZED = "specialized"  # 특화된 도메인 처리


class RoutingDecision(BaseModel):
    """Core routing decision with confidence scores."""

    query_id: str = Field(..., description="Unique query identifier")
    selected_path: RoutingPath = Field(..., description="Selected routing path")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Decision confidence")

    # Classification results
    query_type: QueryType = Field(..., description="Classified query type")
    complexity: QueryComplexity = Field(..., description="Query complexity level")
    knowledge_requirement: KnowledgeRequirement = Field(
        ..., description="External knowledge requirement"
    )

    # Cost and performance estimates
    estimated_cost: float = Field(..., ge=0.0, description="Estimated processing cost")
    estimated_time: float = Field(..., ge=0.0, description="Estimated response time (seconds)")
    cost_savings: float = Field(default=0.0, description="Cost savings vs full RAG pipeline")

    # Reasoning and context
    reasoning: str = Field(..., description="Human-readable decision reasoning")
    alternative_paths: List[RoutingPath] = Field(
        default_factory=list, description="Alternative viable paths"
    )
    context_factors: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context influencing decision"
    )

    # Metadata
    decision_timestamp: datetime = Field(default_factory=datetime.now)
    model_version: str = Field(default="1.0", description="Decision model version")

    @validator("confidence")
    def validate_confidence(cls, v):
        """Ensure confidence is within valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    @validator("cost_savings")
    def validate_cost_savings(cls, v):
        """Ensure cost savings is non-negative percentage."""
        if v < 0.0:
            raise ValueError("Cost savings cannot be negative")
        return v


class DecisionContext(BaseModel):
    """Context information for routing decisions."""

    # Query information
    query_text: str = Field(..., description="Original user query")
    query_language: str = Field(default="ko", description="Query language")
    query_length: int = Field(..., description="Query character length")

    # User context
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    conversation_history: List[str] = Field(
        default_factory=list, description="Recent conversation history"
    )

    # System context
    available_modules: List[ModuleType] = Field(
        default_factory=list, description="Currently available modules"
    )
    system_load: float = Field(default=0.5, description="Current system load (0.0-1.0)")
    cost_budget: Optional[float] = Field(None, description="Available cost budget")

    # Performance requirements
    max_response_time: Optional[float] = Field(None, description="Maximum allowed response time")
    quality_threshold: float = Field(default=0.8, description="Minimum quality threshold")
    cost_preference: float = Field(
        default=0.5, description="Cost vs quality preference (0=cost, 1=quality)"
    )

    # Domain and specialization
    domain_context: Optional[str] = Field(None, description="Specific domain context")
    specialized_knowledge: List[str] = Field(
        default_factory=list, description="Required specialized knowledge areas"
    )


class RoutingMetrics(BaseModel):
    """Metrics for routing decision evaluation."""

    decision_id: str = Field(..., description="Decision identifier")

    # Performance metrics
    actual_response_time: float = Field(..., description="Actual response time")
    actual_cost: float = Field(..., description="Actual processing cost")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Response quality score")

    # Accuracy metrics
    user_satisfaction: Optional[float] = Field(None, description="User satisfaction score")
    routing_accuracy: bool = Field(..., description="Whether routing was optimal")
    cost_efficiency: float = Field(..., description="Cost efficiency vs baseline")

    # Comparison with alternatives
    baseline_cost: float = Field(..., description="Cost if full RAG was used")
    cost_savings_realized: float = Field(..., description="Actual cost savings achieved")
    quality_vs_baseline: float = Field(..., description="Quality compared to full RAG")

    # Feedback
    feedback_timestamp: datetime = Field(default_factory=datetime.now)
    notes: Optional[str] = Field(None, description="Additional notes or feedback")


class RoutingConfiguration(BaseModel):
    """Configuration for the routing system."""

    # Decision thresholds
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence for routing")
    quality_threshold: float = Field(default=0.8, description="Minimum quality threshold")

    # Cost optimization
    cost_weight: float = Field(default=0.3, description="Weight for cost in decisions")
    quality_weight: float = Field(default=0.7, description="Weight for quality in decisions")
    max_cost_per_query: Optional[float] = Field(None, description="Maximum cost per query")

    # Performance requirements
    max_response_time: float = Field(default=30.0, description="Maximum response time")
    fallback_path: RoutingPath = Field(
        default=RoutingPath.DIRECT_LLM, description="Fallback when routing fails"
    )

    # Model settings
    classification_model: str = Field(default="gpt-3.5-turbo", description="Classification model")
    decision_model: str = Field(default="gpt-4", description="Decision making model")
    enable_learning: bool = Field(default=True, description="Enable learning from feedback")

    # Caching and optimization
    enable_caching: bool = Field(default=True, description="Enable decision caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")


# Extended ModuleType enum for routing
class RoutingModuleType(str, Enum):
    """Extended module types for intelligent routing system."""

    # Existing module types (imported from base)
    # ... (all existing ModuleType values)

    # New routing-specific module types
    QUERY_CLASSIFIER = "query_classifier"  # Query classification and analysis
    ROUTE_CONTROLLER = "route_controller"  # Routing decision controller
    KNOWLEDGE_ASSESSOR = "knowledge_assessor"  # Knowledge requirement assessment
    DECISION_ENGINE = "decision_engine"  # LLM-based decision making
    COST_OPTIMIZER = "cost_optimizer"  # Cost-performance optimization
    ROUTING_ADAPTER = "routing_adapter"  # MCP routing adapter

    # Hybrid processing modules
    PARALLEL_PROCESSOR = "parallel_processor"  # Parallel path execution
    RESULT_AGGREGATOR = "result_aggregator"  # Result aggregation and synthesis
    QUALITY_EVALUATOR = "quality_evaluator"  # Response quality evaluation
    FEEDBACK_PROCESSOR = "feedback_processor"  # User feedback processing


# Routing-specific data flow patterns
INTELLIGENT_ROUTING_PATTERNS = {
    "query_analysis": [
        (DataType.TEXT, RoutingModuleType.QUERY_CLASSIFIER, DataType.STRUCTURED),
        (
            DataType.STRUCTURED,
            RoutingModuleType.KNOWLEDGE_ASSESSOR,
            DataType.STRUCTURED,
        ),
    ],
    "routing_decision": [
        (DataType.STRUCTURED, RoutingModuleType.DECISION_ENGINE, DataType.STRUCTURED),
        (DataType.STRUCTURED, RoutingModuleType.ROUTE_CONTROLLER, DataType.STRUCTURED),
    ],
    "cost_optimization": [
        (DataType.STRUCTURED, RoutingModuleType.COST_OPTIMIZER, DataType.STRUCTURED),
    ],
    "parallel_processing": [
        (DataType.TEXT, RoutingModuleType.PARALLEL_PROCESSOR, DataType.MULTIMODAL),
        (DataType.MULTIMODAL, RoutingModuleType.RESULT_AGGREGATOR, DataType.STRUCTURED),
    ],
}
