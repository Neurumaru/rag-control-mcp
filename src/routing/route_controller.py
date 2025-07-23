"""Central routing controller for intelligent RAG path selection."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel

from ..models.routing_schemas import (
    DecisionContext,
    KnowledgeRequirement,
    QueryComplexity,
    QueryType,
    RoutingConfiguration,
    RoutingDecision,
    RoutingMetrics,
    RoutingPath,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)
from .query_classifier import QueryClassifier


class RouteMapping(BaseModel):
    """Mapping configuration for routing decisions."""

    query_type: QueryType
    complexity_range: Tuple[QueryComplexity, QueryComplexity]
    knowledge_requirement: KnowledgeRequirement
    recommended_path: RoutingPath
    alternative_paths: List[RoutingPath]
    confidence_threshold: float = 0.7


class RouteController:
    """Central controller for intelligent routing decisions."""

    def __init__(self, config: Optional[RoutingConfiguration] = None):
        """Initialize the route controller.

        Args:
            config: Routing configuration settings
        """
        self.config = config or RoutingConfiguration()
        self.classifier = QueryClassifier(config)

        # Initialize routing mappings
        self._routing_mappings = self._initialize_routing_mappings()

        # Performance tracking
        self._decision_cache: Dict[str, RoutingDecision] = {}
        self._performance_history: List[RoutingMetrics] = []

        logger.info("RouteController initialized with intelligent routing mappings")

    async def make_routing_decision(
        self, query: str, context: Optional[DecisionContext] = None
    ) -> RoutingDecision:
        """Make an intelligent routing decision for a query.

        Args:
            query: User query text
            context: Additional context for decision making

        Returns:
            Complete routing decision with path and reasoning
        """
        query_id = str(uuid4())
        start_time = datetime.now()

        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, context)
            if self.config.enable_caching and cache_key in self._decision_cache:
                cached_decision = self._decision_cache[cache_key]
                logger.info(f"Using cached routing decision for query: {query[:50]}...")
                return cached_decision

            # Classify the query
            query_type, complexity, knowledge_req, confidence = self.classifier.classify_query(
                query, context
            )

            # Find best routing path
            selected_path, alternative_paths = self._select_optimal_path(
                query_type, complexity, knowledge_req, context
            )

            # Calculate cost estimates
            estimated_cost, estimated_time = self._estimate_costs(selected_path, query, context)

            # Calculate cost savings vs full RAG
            baseline_cost = self._estimate_costs(RoutingPath.ENHANCED_RAG, query, context)[0]
            cost_savings = max(0.0, (baseline_cost - estimated_cost) / baseline_cost * 100)

            # Generate reasoning
            reasoning = self._generate_reasoning(
                query_type, complexity, knowledge_req, selected_path, confidence
            )

            # Create routing decision
            decision = RoutingDecision(
                query_id=query_id,
                selected_path=selected_path,
                confidence=confidence,
                query_type=query_type,
                complexity=complexity,
                knowledge_requirement=knowledge_req,
                estimated_cost=estimated_cost,
                estimated_time=estimated_time,
                cost_savings=cost_savings,
                reasoning=reasoning,
                alternative_paths=alternative_paths,
                context_factors=self._extract_context_factors(context),
                decision_timestamp=start_time,
            )

            # Cache the decision
            if self.config.enable_caching:
                self._decision_cache[cache_key] = decision

            logger.info(
                f"Routing decision made: {selected_path.value} "
                f"(confidence: {confidence:.2f}, cost_savings: {cost_savings:.1f}%)"
            )

            return decision

        except Exception as e:
            logger.error(f"Error making routing decision: {e}")
            # Return fallback decision
            return self._create_fallback_decision(query_id, query, context)

    def _initialize_routing_mappings(self) -> List[RouteMapping]:
        """Initialize the routing decision mappings."""
        return [
            # Direct LLM routes (highest cost savings)
            RouteMapping(
                query_type=QueryType.GREETING,
                complexity_range=(QueryComplexity.TRIVIAL, QueryComplexity.SIMPLE),
                knowledge_requirement=KnowledgeRequirement.NONE,
                recommended_path=RoutingPath.DIRECT_LLM,
                alternative_paths=[],
                confidence_threshold=0.9,
            ),
            RouteMapping(
                query_type=QueryType.CONVERSATIONAL,
                complexity_range=(QueryComplexity.TRIVIAL, QueryComplexity.SIMPLE),
                knowledge_requirement=KnowledgeRequirement.NONE,
                recommended_path=RoutingPath.DIRECT_LLM,
                alternative_paths=[RoutingPath.SIMPLE_RAG],
                confidence_threshold=0.8,
            ),
            RouteMapping(
                query_type=QueryType.CREATIVE,
                complexity_range=(QueryComplexity.SIMPLE, QueryComplexity.MODERATE),
                knowledge_requirement=KnowledgeRequirement.MINIMAL,
                recommended_path=RoutingPath.DIRECT_LLM,
                alternative_paths=[RoutingPath.HYBRID_PARALLEL],
                confidence_threshold=0.7,
            ),
            # Simple RAG routes
            RouteMapping(
                query_type=QueryType.FACTUAL_SIMPLE,
                complexity_range=(QueryComplexity.SIMPLE, QueryComplexity.MODERATE),
                knowledge_requirement=KnowledgeRequirement.MINIMAL,
                recommended_path=RoutingPath.SIMPLE_RAG,
                alternative_paths=[RoutingPath.DIRECT_LLM, RoutingPath.ENHANCED_RAG],
                confidence_threshold=0.7,
            ),
            RouteMapping(
                query_type=QueryType.PROCEDURAL,
                complexity_range=(QueryComplexity.SIMPLE, QueryComplexity.MODERATE),
                knowledge_requirement=KnowledgeRequirement.MODERATE,
                recommended_path=RoutingPath.SIMPLE_RAG,
                alternative_paths=[RoutingPath.ENHANCED_RAG],
                confidence_threshold=0.8,
            ),
            # Enhanced RAG routes
            RouteMapping(
                query_type=QueryType.FACTUAL_COMPLEX,
                complexity_range=(QueryComplexity.MODERATE, QueryComplexity.COMPLEX),
                knowledge_requirement=KnowledgeRequirement.EXTENSIVE,
                recommended_path=RoutingPath.ENHANCED_RAG,
                alternative_paths=[RoutingPath.MULTI_STEP],
                confidence_threshold=0.8,
            ),
            RouteMapping(
                query_type=QueryType.ANALYTICAL,
                complexity_range=(QueryComplexity.MODERATE, QueryComplexity.EXPERT),
                knowledge_requirement=KnowledgeRequirement.EXTENSIVE,
                recommended_path=RoutingPath.ENHANCED_RAG,
                alternative_paths=[RoutingPath.MULTI_STEP, RoutingPath.HYBRID_PARALLEL],
                confidence_threshold=0.7,
            ),
            # Specialized routes
            RouteMapping(
                query_type=QueryType.TECHNICAL,
                complexity_range=(QueryComplexity.COMPLEX, QueryComplexity.EXPERT),
                knowledge_requirement=KnowledgeRequirement.SPECIALIZED,
                recommended_path=RoutingPath.SPECIALIZED,
                alternative_paths=[RoutingPath.ENHANCED_RAG, RoutingPath.MULTI_STEP],
                confidence_threshold=0.8,
            ),
            RouteMapping(
                query_type=QueryType.DOMAIN_SPECIFIC,
                complexity_range=(QueryComplexity.MODERATE, QueryComplexity.EXPERT),
                knowledge_requirement=KnowledgeRequirement.SPECIALIZED,
                recommended_path=RoutingPath.SPECIALIZED,
                alternative_paths=[RoutingPath.ENHANCED_RAG],
                confidence_threshold=0.8,
            ),
            # Hybrid routes for complex scenarios
            RouteMapping(
                query_type=QueryType.MULTIMODAL,
                complexity_range=(QueryComplexity.MODERATE, QueryComplexity.EXPERT),
                knowledge_requirement=KnowledgeRequirement.EXTENSIVE,
                recommended_path=RoutingPath.HYBRID_PARALLEL,
                alternative_paths=[RoutingPath.MULTI_STEP],
                confidence_threshold=0.7,
            ),
        ]

    def _select_optimal_path(
        self,
        query_type: QueryType,
        complexity: QueryComplexity,
        knowledge_req: KnowledgeRequirement,
        context: Optional[DecisionContext],
    ) -> Tuple[RoutingPath, List[RoutingPath]]:
        """Select the optimal routing path based on classification results."""

        # Find matching routing mappings
        matching_mappings = []
        for mapping in self._routing_mappings:
            if (
                mapping.query_type == query_type
                and self._complexity_in_range(complexity, mapping.complexity_range)
                and mapping.knowledge_requirement == knowledge_req
            ):
                matching_mappings.append(mapping)

        # If exact match found, use it
        if matching_mappings:
            best_mapping = matching_mappings[0]  # Could add scoring logic here
            return best_mapping.recommended_path, best_mapping.alternative_paths

        # Fallback logic based on knowledge requirement and complexity
        return self._fallback_path_selection(query_type, complexity, knowledge_req, context)

    def _complexity_in_range(
        self,
        complexity: QueryComplexity,
        complexity_range: Tuple[QueryComplexity, QueryComplexity],
    ) -> bool:
        """Check if complexity falls within the specified range."""
        complexity_order = [
            QueryComplexity.TRIVIAL,
            QueryComplexity.SIMPLE,
            QueryComplexity.MODERATE,
            QueryComplexity.COMPLEX,
            QueryComplexity.EXPERT,
        ]

        complexity_idx = complexity_order.index(complexity)
        min_idx = complexity_order.index(complexity_range[0])
        max_idx = complexity_order.index(complexity_range[1])

        return min_idx <= complexity_idx <= max_idx

    def _fallback_path_selection(
        self,
        query_type: QueryType,
        complexity: QueryComplexity,
        knowledge_req: KnowledgeRequirement,
        context: Optional[DecisionContext],
    ) -> Tuple[RoutingPath, List[RoutingPath]]:
        """Fallback path selection when no exact mapping is found."""

        # Simple heuristic-based selection
        if knowledge_req == KnowledgeRequirement.NONE:
            return RoutingPath.DIRECT_LLM, [RoutingPath.SIMPLE_RAG]

        elif knowledge_req == KnowledgeRequirement.MINIMAL:
            if complexity <= QueryComplexity.SIMPLE:
                return RoutingPath.DIRECT_LLM, [RoutingPath.SIMPLE_RAG]
            else:
                return RoutingPath.SIMPLE_RAG, [
                    RoutingPath.DIRECT_LLM,
                    RoutingPath.ENHANCED_RAG,
                ]

        elif knowledge_req == KnowledgeRequirement.MODERATE:
            return RoutingPath.SIMPLE_RAG, [RoutingPath.ENHANCED_RAG]

        elif knowledge_req == KnowledgeRequirement.EXTENSIVE:
            if complexity >= QueryComplexity.COMPLEX:
                return RoutingPath.ENHANCED_RAG, [RoutingPath.MULTI_STEP]
            else:
                return RoutingPath.ENHANCED_RAG, [RoutingPath.SIMPLE_RAG]

        else:  # SPECIALIZED
            return RoutingPath.SPECIALIZED, [
                RoutingPath.ENHANCED_RAG,
                RoutingPath.MULTI_STEP,
            ]

    def _estimate_costs(
        self, path: RoutingPath, query: str, context: Optional[DecisionContext]
    ) -> Tuple[float, float]:
        """Estimate cost and time for a routing path.

        Returns:
            Tuple of (estimated_cost, estimated_time_seconds)
        """
        # Base costs (in arbitrary units, could be actual $ amounts)
        cost_multipliers = {
            RoutingPath.DIRECT_LLM: 0.1,  # Lowest cost - just LLM inference
            RoutingPath.SIMPLE_RAG: 0.4,  # Embedding + vector search + LLM
            RoutingPath.ENHANCED_RAG: 1.0,  # Full RAG with reranking
            RoutingPath.HYBRID_PARALLEL: 0.6,  # Parallel processing, some savings
            RoutingPath.MULTI_STEP: 1.5,  # Multiple iterations
            RoutingPath.SPECIALIZED: 2.0,  # Specialized processing
        }

        time_multipliers = {
            RoutingPath.DIRECT_LLM: 0.2,  # Fastest - direct response
            RoutingPath.SIMPLE_RAG: 0.6,  # Moderate - single retrieval
            RoutingPath.ENHANCED_RAG: 1.0,  # Standard RAG time
            RoutingPath.HYBRID_PARALLEL: 0.8,  # Parallel processing benefits
            RoutingPath.MULTI_STEP: 2.0,  # Sequential processing
            RoutingPath.SPECIALIZED: 1.5,  # Complex but optimized
        }

        # Base estimates
        base_cost = 1.0
        base_time = 5.0  # seconds

        # Query length factor
        query_length_factor = min(2.0, len(query) / 100)

        # Context factors
        complexity_factor = 1.0
        if context:
            if context.quality_threshold > 0.9:
                complexity_factor *= 1.3
            if context.max_response_time and context.max_response_time < 10:
                complexity_factor *= 0.8  # Need faster processing

        estimated_cost = (
            base_cost * cost_multipliers[path] * query_length_factor * complexity_factor
        )

        estimated_time = (
            base_time * time_multipliers[path] * query_length_factor * complexity_factor
        )

        return estimated_cost, estimated_time

    def _generate_reasoning(
        self,
        query_type: QueryType,
        complexity: QueryComplexity,
        knowledge_req: KnowledgeRequirement,
        selected_path: RoutingPath,
        confidence: float,
    ) -> str:
        """Generate human-readable reasoning for the routing decision."""

        reasoning_parts = []

        # Query analysis
        reasoning_parts.append(f"쿼리 분석: {query_type.value} 유형, {complexity.value} 복잡도")

        # Knowledge requirement
        knowledge_desc = {
            KnowledgeRequirement.NONE: "외부 지식이 필요하지 않음",
            KnowledgeRequirement.MINIMAL: "최소한의 검증만 필요",
            KnowledgeRequirement.MODERATE: "일반적인 RAG 검색 필요",
            KnowledgeRequirement.EXTENSIVE: "광범위한 검색 필요",
            KnowledgeRequirement.SPECIALIZED: "전문 지식 데이터베이스 필요",
        }
        reasoning_parts.append(f"지식 요구사항: {knowledge_desc[knowledge_req]}")

        # Path selection reasoning
        path_reasoning = {
            RoutingPath.DIRECT_LLM: "LLM의 내재 지식으로 충분히 답변 가능하여 직접 응답 선택",
            RoutingPath.SIMPLE_RAG: "기본적인 검색과 생성으로 효율적인 답변 가능",
            RoutingPath.ENHANCED_RAG: "재순위화를 포함한 고급 RAG로 높은 품질 답변 제공",
            RoutingPath.HYBRID_PARALLEL: "병렬 처리로 속도와 품질의 균형 달성",
            RoutingPath.MULTI_STEP: "복잡한 질문을 단계별로 해결하기 위한 다단계 처리",
            RoutingPath.SPECIALIZED: "전문 도메인 처리를 위한 특화된 파이프라인 사용",
        }
        reasoning_parts.append(f"경로 선택: {path_reasoning[selected_path]}")

        # Confidence explanation
        if confidence >= 0.8:
            confidence_desc = "높은 신뢰도로 최적 경로 확신"
        elif confidence >= 0.6:
            confidence_desc = "적절한 신뢰도로 권장 경로 제안"
        else:
            confidence_desc = "낮은 신뢰도로 대안 경로 고려 필요"

        reasoning_parts.append(f"신뢰도: {confidence:.2f} - {confidence_desc}")

        return " | ".join(reasoning_parts)

    def _extract_context_factors(self, context: Optional[DecisionContext]) -> Dict[str, Any]:
        """Extract relevant context factors for decision documentation."""
        factors = {}

        if context:
            if context.system_load:
                factors["system_load"] = context.system_load
            if context.cost_budget:
                factors["cost_budget"] = context.cost_budget
            if context.max_response_time:
                factors["max_response_time"] = context.max_response_time
            if context.quality_threshold:
                factors["quality_threshold"] = context.quality_threshold
            if context.domain_context:
                factors["domain_context"] = context.domain_context
            if context.conversation_history:
                factors["conversation_length"] = len(context.conversation_history)

        return factors

    def _generate_cache_key(self, query: str, context: Optional[DecisionContext]) -> str:
        """Generate cache key for routing decisions."""
        context_key = ""
        if context:
            context_key = f"_{context.domain_context or ''}_{len(context.conversation_history)}"

        return f"{hash(query.lower().strip())}{context_key}"

    def _create_fallback_decision(
        self, query_id: str, query: str, context: Optional[DecisionContext]
    ) -> RoutingDecision:
        """Create a fallback routing decision when classification fails."""
        return RoutingDecision(
            query_id=query_id,
            selected_path=self.config.fallback_path,
            confidence=0.5,
            query_type=QueryType.CONVERSATIONAL,
            complexity=QueryComplexity.MODERATE,
            knowledge_requirement=KnowledgeRequirement.MODERATE,
            estimated_cost=1.0,
            estimated_time=10.0,
            cost_savings=0.0,
            reasoning="분류 실패로 인한 안전한 기본 경로 선택",
            alternative_paths=[RoutingPath.SIMPLE_RAG, RoutingPath.ENHANCED_RAG],
            context_factors=self._extract_context_factors(context),
        )

    async def record_performance_metrics(self, metrics: RoutingMetrics) -> None:
        """Record performance metrics for routing decisions."""
        self._performance_history.append(metrics)

        # Keep only recent metrics (last 1000)
        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-1000:]

        logger.info(f"Performance metrics recorded for decision {metrics.decision_id}")

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing performance statistics."""
        if not self._performance_history:
            return {"message": "No performance data available"}

        total_decisions = len(self._performance_history)
        total_cost_savings = sum(m.cost_savings_realized for m in self._performance_history)
        avg_quality = sum(m.quality_score for m in self._performance_history) / total_decisions
        avg_response_time = (
            sum(m.actual_response_time for m in self._performance_history) / total_decisions
        )

        # Path usage statistics
        path_usage = {}
        for metrics in self._performance_history:
            # Note: We'd need to store the path in metrics for this to work
            pass

        return {
            "total_decisions": total_decisions,
            "average_cost_savings": (
                total_cost_savings / total_decisions if total_decisions > 0 else 0
            ),
            "average_quality_score": avg_quality,
            "average_response_time": avg_response_time,
            "cache_hit_rate": len(self._decision_cache) / max(1, total_decisions),
            "path_usage_statistics": path_usage,
        }
