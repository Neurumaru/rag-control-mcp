"""Cost optimization algorithms for intelligent routing decisions."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from ..models.routing_schemas import (
    DecisionContext,
    KnowledgeRequirement,
    QueryComplexity,
    RoutingConfiguration,
    RoutingPath,
)
from ..utils.logger import logger


class OptimizationStrategy(str, Enum):
    """Cost optimization strategies."""

    AGGRESSIVE_COST = "aggressive_cost"  # Minimize cost above all
    BALANCED = "balanced"  # Balance cost and quality
    QUALITY_FIRST = "quality_first"  # Prioritize quality, optimize cost secondarily
    ADAPTIVE = "adaptive"  # Adapt based on context and history


class CostMetrics(BaseModel):
    """Cost-related metrics for routing paths."""

    # Direct costs
    compute_cost: float = 0.0  # Processing cost
    api_cost: float = 0.0  # External API costs
    storage_cost: float = 0.0  # Storage/retrieval costs

    # Time costs
    processing_time: float = 0.0  # Expected processing time
    user_waiting_time: float = 0.0  # User perceived waiting time

    # Resource costs
    memory_usage: float = 0.0  # Memory requirements
    bandwidth_usage: float = 0.0  # Network bandwidth

    # Quality trade-offs
    quality_score: float = 0.0  # Expected quality score
    confidence_score: float = 0.0  # Confidence in result

    # Total cost calculation
    total_cost: float = 0.0
    cost_efficiency: float = 0.0  # Quality per unit cost


class CostOptimization(BaseModel):
    """Result of cost optimization analysis."""

    recommended_path: RoutingPath
    cost_savings: float  # Percentage savings vs baseline
    quality_impact: float  # Quality change vs baseline
    confidence: float  # Confidence in optimization

    optimization_reasoning: str
    alternative_paths: List[RoutingPath]
    cost_breakdown: CostMetrics

    # Risk assessment
    risk_level: str  # "low", "medium", "high"
    risk_factors: List[str]


class CostOptimizer:
    """Intelligent cost optimization for routing decisions."""

    def __init__(self, config: Optional[RoutingConfiguration] = None):
        """Initialize the cost optimizer.

        Args:
            config: Routing configuration settings
        """
        self.config = config or RoutingConfiguration()

        # Cost models for different routing paths
        self._path_cost_models = self._initialize_cost_models()

        # Historical cost data for learning
        self._cost_history: List[Dict[str, Any]] = []
        self._performance_baselines: Dict[str, float] = {}

        # Optimization parameters
        self.optimization_strategy = OptimizationStrategy.BALANCED
        self.cost_weight = self.config.cost_weight
        self.quality_weight = self.config.quality_weight

        logger.info("CostOptimizer initialized with balanced optimization strategy")

    def optimize_routing_cost(
        self,
        query: str,
        possible_paths: List[RoutingPath],
        knowledge_req: KnowledgeRequirement,
        complexity: QueryComplexity,
        context: Optional[DecisionContext] = None,
    ) -> CostOptimization:
        """Optimize routing path selection for cost efficiency.

        Args:
            query: User query
            possible_paths: Available routing paths
            knowledge_req: Knowledge requirement level
            complexity: Query complexity
            context: Decision context

        Returns:
            Cost optimization recommendation
        """
        try:
            # Calculate costs for all possible paths
            path_costs = {}
            for path in possible_paths:
                cost_metrics = self._calculate_path_costs(
                    path, query, knowledge_req, complexity, context
                )
                path_costs[path] = cost_metrics

            # Select optimization strategy
            strategy = self._select_optimization_strategy(context)

            # Apply optimization algorithm
            if strategy == OptimizationStrategy.AGGRESSIVE_COST:
                optimization = self._optimize_for_cost(path_costs, context)
            elif strategy == OptimizationStrategy.QUALITY_FIRST:
                optimization = self._optimize_for_quality(path_costs, context)
            elif strategy == OptimizationStrategy.ADAPTIVE:
                optimization = self._adaptive_optimization(path_costs, context)
            else:  # BALANCED
                optimization = self._balanced_optimization(path_costs, context)

            # Record optimization decision
            self._record_optimization_decision(optimization, path_costs)

            logger.info(
                f"Cost optimization: {optimization.recommended_path.value} "
                f"(savings: {optimization.cost_savings:.1f}%, "
                f"quality_impact: {optimization.quality_impact:+.2f})"
            )

            return optimization

        except Exception as e:
            logger.error(f"Error in cost optimization: {e}")
            return self._create_fallback_optimization(possible_paths)

    def _initialize_cost_models(self) -> Dict[RoutingPath, Dict[str, float]]:
        """Initialize cost models for different routing paths."""

        return {
            RoutingPath.DIRECT_LLM: {
                "base_compute_cost": 0.05,  # Very low - just LLM inference
                "api_cost_per_token": 0.0001,  # LLM API cost
                "storage_cost": 0.0,  # No retrieval
                "processing_time": 2.0,  # Fast response
                "quality_baseline": 0.7,  # Good for general queries
                "memory_usage": 100,  # MB
                "bandwidth_usage": 1.0,  # KB
            },
            RoutingPath.SIMPLE_RAG: {
                "base_compute_cost": 0.20,  # Embedding + retrieval + LLM
                "api_cost_per_token": 0.0001,
                "storage_cost": 0.02,  # Vector search cost
                "processing_time": 5.0,  # Moderate response time
                "quality_baseline": 0.8,  # Better factual accuracy
                "memory_usage": 300,
                "bandwidth_usage": 5.0,
            },
            RoutingPath.ENHANCED_RAG: {
                "base_compute_cost": 0.50,  # Full RAG + reranking
                "api_cost_per_token": 0.0002,  # More LLM calls for reranking
                "storage_cost": 0.05,  # Multiple retrievals
                "processing_time": 8.0,  # Slower due to reranking
                "quality_baseline": 0.9,  # High quality
                "memory_usage": 500,
                "bandwidth_usage": 10.0,
            },
            RoutingPath.HYBRID_PARALLEL: {
                "base_compute_cost": 0.35,  # Parallel processing savings
                "api_cost_per_token": 0.00015,
                "storage_cost": 0.03,  # Some retrieval
                "processing_time": 6.0,  # Parallel efficiency
                "quality_baseline": 0.85,  # Good balance
                "memory_usage": 400,
                "bandwidth_usage": 8.0,
            },
            RoutingPath.MULTI_STEP: {
                "base_compute_cost": 0.80,  # Multiple iterations
                "api_cost_per_token": 0.0003,  # Many LLM calls
                "storage_cost": 0.08,  # Multiple retrievals
                "processing_time": 12.0,  # Slow but thorough
                "quality_baseline": 0.95,  # Highest quality
                "memory_usage": 700,
                "bandwidth_usage": 15.0,
            },
            RoutingPath.SPECIALIZED: {
                "base_compute_cost": 1.00,  # Specialized processing
                "api_cost_per_token": 0.0002,
                "storage_cost": 0.10,  # Specialized data access
                "processing_time": 10.0,  # Variable based on domain
                "quality_baseline": 0.92,  # High for domain queries
                "memory_usage": 600,
                "bandwidth_usage": 12.0,
            },
        }

    def _calculate_path_costs(
        self,
        path: RoutingPath,
        query: str,
        knowledge_req: KnowledgeRequirement,
        complexity: QueryComplexity,
        context: Optional[DecisionContext],
    ) -> CostMetrics:
        """Calculate detailed costs for a specific routing path."""

        model = self._path_cost_models[path]

        # Base costs
        compute_cost = model["base_compute_cost"]

        # Token-based API costs (estimate based on query length)
        estimated_tokens = len(query.split()) * 1.3 + 50  # Rough estimate
        api_cost = estimated_tokens * model["api_cost_per_token"]

        # Storage/retrieval costs
        storage_cost = model["storage_cost"]

        # Time costs
        processing_time = model["processing_time"]

        # Adjust costs based on complexity
        complexity_multipliers = {
            QueryComplexity.TRIVIAL: 0.5,
            QueryComplexity.SIMPLE: 0.8,
            QueryComplexity.MODERATE: 1.0,
            QueryComplexity.COMPLEX: 1.5,
            QueryComplexity.EXPERT: 2.0,
        }
        complexity_multiplier = complexity_multipliers[complexity]

        compute_cost *= complexity_multiplier
        processing_time *= complexity_multiplier

        # Adjust costs based on knowledge requirements
        knowledge_multipliers = {
            KnowledgeRequirement.NONE: 0.7,
            KnowledgeRequirement.MINIMAL: 0.9,
            KnowledgeRequirement.MODERATE: 1.0,
            KnowledgeRequirement.EXTENSIVE: 1.3,
            KnowledgeRequirement.SPECIALIZED: 1.8,
        }
        knowledge_multiplier = knowledge_multipliers[knowledge_req]

        storage_cost *= knowledge_multiplier
        processing_time *= knowledge_multiplier

        # Context-based adjustments
        if context:
            # Rush job increases costs
            if context.max_response_time and context.max_response_time < processing_time:
                compute_cost *= 1.5  # Premium for fast processing

            # High quality requirements increase costs
            if context.quality_threshold > 0.9:
                compute_cost *= 1.2
                storage_cost *= 1.2

        # Calculate quality and confidence scores
        quality_score = model["quality_baseline"]
        confidence_score = quality_score * 0.9  # Slightly lower than quality

        # Adjust quality based on appropriateness of path for the query
        quality_score = self._adjust_quality_for_appropriateness(
            path, knowledge_req, complexity, quality_score
        )

        # Total cost calculation
        total_cost = compute_cost + api_cost + storage_cost

        # Cost efficiency (quality per unit cost)
        cost_efficiency = quality_score / max(0.01, total_cost)

        return CostMetrics(
            compute_cost=compute_cost,
            api_cost=api_cost,
            storage_cost=storage_cost,
            processing_time=processing_time,
            user_waiting_time=processing_time * 1.1,  # User perception
            memory_usage=model["memory_usage"],
            bandwidth_usage=model["bandwidth_usage"],
            quality_score=quality_score,
            confidence_score=confidence_score,
            total_cost=total_cost,
            cost_efficiency=cost_efficiency,
        )

    def _adjust_quality_for_appropriateness(
        self,
        path: RoutingPath,
        knowledge_req: KnowledgeRequirement,
        complexity: QueryComplexity,
        base_quality: float,
    ) -> float:
        """Adjust quality score based on how appropriate the path is for the query."""

        # Direct LLM is great for no/minimal knowledge needs
        if path == RoutingPath.DIRECT_LLM:
            if knowledge_req in [
                KnowledgeRequirement.NONE,
                KnowledgeRequirement.MINIMAL,
            ]:
                return min(0.95, base_quality + 0.1)
            elif knowledge_req >= KnowledgeRequirement.EXTENSIVE:
                return base_quality - 0.2  # Not appropriate for knowledge-heavy queries

        # Simple RAG is optimal for moderate knowledge needs
        elif path == RoutingPath.SIMPLE_RAG:
            if knowledge_req == KnowledgeRequirement.MODERATE:
                return min(0.95, base_quality + 0.05)

        # Enhanced RAG is best for extensive knowledge needs
        elif path == RoutingPath.ENHANCED_RAG:
            if knowledge_req == KnowledgeRequirement.EXTENSIVE:
                return min(0.95, base_quality + 0.05)

        # Multi-step is overkill for simple queries
        elif path == RoutingPath.MULTI_STEP:
            if complexity <= QueryComplexity.SIMPLE:
                return base_quality - 0.1  # Overkill penalty

        # Specialized is perfect for specialized needs
        elif path == RoutingPath.SPECIALIZED:
            if knowledge_req == KnowledgeRequirement.SPECIALIZED:
                return min(0.98, base_quality + 0.06)
            else:
                return base_quality - 0.05  # Not needed for general queries

        return base_quality

    def _select_optimization_strategy(
        self, context: Optional[DecisionContext]
    ) -> OptimizationStrategy:
        """Select the appropriate optimization strategy based on context."""

        if not context:
            return OptimizationStrategy.BALANCED

        # Cost-sensitive scenarios
        if context.cost_preference < 0.3:
            return OptimizationStrategy.AGGRESSIVE_COST

        # Quality-sensitive scenarios
        if context.quality_threshold >= 0.9 or context.cost_preference > 0.8:
            return OptimizationStrategy.QUALITY_FIRST

        # High system load - prefer cost optimization
        if context.system_load > 0.8:
            return OptimizationStrategy.AGGRESSIVE_COST

        # Budget constraints
        if context.cost_budget and context.cost_budget < 0.5:
            return OptimizationStrategy.AGGRESSIVE_COST

        # Default to adaptive for complex scenarios
        return OptimizationStrategy.ADAPTIVE

    def _optimize_for_cost(
        self,
        path_costs: Dict[RoutingPath, CostMetrics],
        context: Optional[DecisionContext],
    ) -> CostOptimization:
        """Optimize primarily for cost minimization."""

        # Find the lowest cost path that meets minimum quality requirements
        min_quality = 0.6  # Minimum acceptable quality
        if context and context.quality_threshold:
            min_quality = context.quality_threshold

        valid_paths = {
            path: costs for path, costs in path_costs.items() if costs.quality_score >= min_quality
        }

        if not valid_paths:
            # If no path meets quality requirements, choose best quality among cheapest 50%
            sorted_by_cost = sorted(path_costs.items(), key=lambda x: x[1].total_cost)
            half_point = len(sorted_by_cost) // 2 + 1
            cheapest_half = dict(sorted_by_cost[:half_point])
            best_path = max(cheapest_half.items(), key=lambda x: x[1].quality_score)
        else:
            # Choose cheapest among valid paths
            best_path = min(valid_paths.items(), key=lambda x: x[1].total_cost)

        recommended_path, cost_metrics = best_path

        # Calculate savings vs most expensive option
        max_cost = max(costs.total_cost for costs in path_costs.values())
        cost_savings = (max_cost - cost_metrics.total_cost) / max_cost * 100

        # Calculate quality impact vs best quality option
        best_quality = max(costs.quality_score for costs in path_costs.values())
        quality_impact = cost_metrics.quality_score - best_quality

        return CostOptimization(
            recommended_path=recommended_path,
            cost_savings=cost_savings,
            quality_impact=quality_impact,
            confidence=0.9,  # High confidence in cost optimization
            optimization_reasoning="최소 비용 경로 선택 - 품질 기준 충족하는 가장 저렴한 옵션",
            alternative_paths=self._get_alternative_paths(path_costs, recommended_path, "cost"),
            cost_breakdown=cost_metrics,
            risk_level="medium",
            risk_factors=[
                "품질이 최적화되지 않을 수 있음",
                "복잡한 쿼리에서 정확도 저하 가능",
            ],
        )

    def _optimize_for_quality(
        self,
        path_costs: Dict[RoutingPath, CostMetrics],
        context: Optional[DecisionContext],
    ) -> CostOptimization:
        """Optimize primarily for quality, with secondary cost consideration."""

        # Find the highest quality path within budget constraints
        max_budget = float("inf")
        if context and context.cost_budget:
            max_budget = context.cost_budget

        affordable_paths = {
            path: costs for path, costs in path_costs.items() if costs.total_cost <= max_budget
        }

        if not affordable_paths:
            # If budget is too tight, choose best quality within 120% of cheapest option
            min_cost = min(costs.total_cost for costs in path_costs.values())
            budget_limit = min_cost * 1.2
            affordable_paths = {
                path: costs
                for path, costs in path_costs.items()
                if costs.total_cost <= budget_limit
            }

        # Choose highest quality among affordable paths
        best_path = max(affordable_paths.items(), key=lambda x: x[1].quality_score)
        recommended_path, cost_metrics = best_path

        # Calculate savings vs most expensive option
        max_cost = max(costs.total_cost for costs in path_costs.values())
        cost_savings = (max_cost - cost_metrics.total_cost) / max_cost * 100

        # Quality impact is minimal since we're optimizing for quality
        best_quality = max(costs.quality_score for costs in path_costs.values())
        quality_impact = cost_metrics.quality_score - best_quality

        return CostOptimization(
            recommended_path=recommended_path,
            cost_savings=cost_savings,
            quality_impact=quality_impact,
            confidence=0.85,
            optimization_reasoning="최고 품질 경로 선택 - 예산 내에서 가장 높은 품질 보장",
            alternative_paths=self._get_alternative_paths(path_costs, recommended_path, "quality"),
            cost_breakdown=cost_metrics,
            risk_level="low",
            risk_factors=["비용이 다소 높을 수 있음"],
        )

    def _balanced_optimization(
        self,
        path_costs: Dict[RoutingPath, CostMetrics],
        context: Optional[DecisionContext],
    ) -> CostOptimization:
        """Optimize for the best balance of cost and quality."""

        # Use cost efficiency (quality per unit cost) as the primary metric
        best_path = max(path_costs.items(), key=lambda x: x[1].cost_efficiency)
        recommended_path, cost_metrics = best_path

        # Calculate savings and quality impact
        max_cost = max(costs.total_cost for costs in path_costs.values())
        cost_savings = (max_cost - cost_metrics.total_cost) / max_cost * 100

        best_quality = max(costs.quality_score for costs in path_costs.values())
        quality_impact = cost_metrics.quality_score - best_quality

        return CostOptimization(
            recommended_path=recommended_path,
            cost_savings=cost_savings,
            quality_impact=quality_impact,
            confidence=0.8,
            optimization_reasoning="비용-품질 균형 최적화 - 단위 비용당 최고 품질 효율성",
            alternative_paths=self._get_alternative_paths(path_costs, recommended_path, "balanced"),
            cost_breakdown=cost_metrics,
            risk_level="low",
            risk_factors=["극단적 비용 절약이나 최고 품질을 보장하지 않음"],
        )

    def _adaptive_optimization(
        self,
        path_costs: Dict[RoutingPath, CostMetrics],
        context: Optional[DecisionContext],
    ) -> CostOptimization:
        """Adaptive optimization based on historical performance and context."""

        # This would use ML/historical data in a real implementation
        # For now, we'll use heuristics based on context

        if context:
            # High system load -> prioritize cost
            if context.system_load > 0.8:
                return self._optimize_for_cost(path_costs, context)

            # Tight deadline -> balance with slight quality preference
            if context.max_response_time:
                avg_time = sum(costs.processing_time for costs in path_costs.values()) / len(
                    path_costs
                )
                if context.max_response_time < avg_time:
                    # Need fast response - prioritize speed and cost
                    fast_paths = {
                        path: costs
                        for path, costs in path_costs.items()
                        if costs.processing_time <= context.max_response_time * 1.1
                    }
                    if fast_paths:
                        return self._balanced_optimization(fast_paths, context)

            # Long conversation -> slight cost preference for efficiency
            if len(context.conversation_history) > 10:
                # Weight cost slightly more for long conversations
                weighted_scores = {}
                for path, costs in path_costs.items():
                    # 60% cost efficiency, 40% quality
                    score = (costs.cost_efficiency * 0.6) + (costs.quality_score * 0.4)
                    weighted_scores[path] = (costs, score)

                best_path = max(weighted_scores.items(), key=lambda x: x[1][1])
                recommended_path, (cost_metrics, _) = best_path

                max_cost = max(costs.total_cost for costs in path_costs.values())
                cost_savings = (max_cost - cost_metrics.total_cost) / max_cost * 100

                best_quality = max(costs.quality_score for costs in path_costs.values())
                quality_impact = cost_metrics.quality_score - best_quality

                return CostOptimization(
                    recommended_path=recommended_path,
                    cost_savings=cost_savings,
                    quality_impact=quality_impact,
                    confidence=0.75,
                    optimization_reasoning="적응형 최적화 - 대화 길이와 맥락을 고려한 비용 효율성 우선",
                    alternative_paths=self._get_alternative_paths(
                        path_costs, recommended_path, "adaptive"
                    ),
                    cost_breakdown=cost_metrics,
                    risk_level="medium",
                    risk_factors=["맥락 기반 추론으로 일부 불확실성 존재"],
                )

        # Default to balanced if no special context
        return self._balanced_optimization(path_costs, context)

    def _get_alternative_paths(
        self,
        path_costs: Dict[RoutingPath, CostMetrics],
        selected_path: RoutingPath,
        optimization_type: str,
    ) -> List[RoutingPath]:
        """Get alternative paths based on optimization type."""

        alternatives = []
        remaining_paths = {k: v for k, v in path_costs.items() if k != selected_path}

        if optimization_type == "cost":
            # Get next cheapest options
            sorted_by_cost = sorted(remaining_paths.items(), key=lambda x: x[1].total_cost)
            alternatives = [path for path, _ in sorted_by_cost[:2]]

        elif optimization_type == "quality":
            # Get next best quality options
            sorted_by_quality = sorted(
                remaining_paths.items(), key=lambda x: x[1].quality_score, reverse=True
            )
            alternatives = [path for path, _ in sorted_by_quality[:2]]

        else:  # balanced or adaptive
            # Get best cost efficiency alternatives
            sorted_by_efficiency = sorted(
                remaining_paths.items(),
                key=lambda x: x[1].cost_efficiency,
                reverse=True,
            )
            alternatives = [path for path, _ in sorted_by_efficiency[:2]]

        return alternatives

    def _record_optimization_decision(
        self, optimization: CostOptimization, path_costs: Dict[RoutingPath, CostMetrics]
    ) -> None:
        """Record optimization decision for learning."""

        decision_record = {
            "timestamp": datetime.now().isoformat(),
            "recommended_path": optimization.recommended_path.value,
            "cost_savings": optimization.cost_savings,
            "quality_impact": optimization.quality_impact,
            "confidence": optimization.confidence,
            "total_cost": optimization.cost_breakdown.total_cost,
            "quality_score": optimization.cost_breakdown.quality_score,
            "cost_efficiency": optimization.cost_breakdown.cost_efficiency,
            "strategy": self.optimization_strategy.value,
        }

        self._cost_history.append(decision_record)

        # Keep only recent history (last 200 decisions)
        if len(self._cost_history) > 200:
            self._cost_history = self._cost_history[-200:]

    def _create_fallback_optimization(self, possible_paths: List[RoutingPath]) -> CostOptimization:
        """Create fallback optimization when calculation fails."""

        # Default to simple RAG as a safe middle ground
        fallback_path = RoutingPath.SIMPLE_RAG
        if fallback_path not in possible_paths and possible_paths:
            fallback_path = possible_paths[0]

        return CostOptimization(
            recommended_path=fallback_path,
            cost_savings=50.0,  # Assume moderate savings
            quality_impact=0.0,  # Assume neutral quality
            confidence=0.5,
            optimization_reasoning="최적화 실패로 인한 안전한 기본 경로 선택",
            alternative_paths=possible_paths[1:3] if len(possible_paths) > 1 else [],
            cost_breakdown=CostMetrics(total_cost=0.3, quality_score=0.8, cost_efficiency=2.67),
            risk_level="medium",
            risk_factors=["최적화 계산 실패로 인한 기본값 사용"],
        )

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get cost optimization performance statistics."""

        if not self._cost_history:
            return {"message": "No optimization history available"}

        total_decisions = len(self._cost_history)
        avg_cost_savings = sum(d["cost_savings"] for d in self._cost_history) / total_decisions
        avg_quality_impact = sum(d["quality_impact"] for d in self._cost_history) / total_decisions
        avg_confidence = sum(d["confidence"] for d in self._cost_history) / total_decisions

        # Path selection frequency
        path_usage = {}
        for decision in self._cost_history:
            path = decision["recommended_path"]
            path_usage[path] = path_usage.get(path, 0) + 1

        # Recent performance (last 30 days)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_decisions = [
            d for d in self._cost_history if datetime.fromisoformat(d["timestamp"]) > recent_cutoff
        ]

        recent_avg_savings = 0.0
        if recent_decisions:
            recent_avg_savings = sum(d["cost_savings"] for d in recent_decisions) / len(
                recent_decisions
            )

        return {
            "total_optimizations": total_decisions,
            "average_cost_savings": avg_cost_savings,
            "average_quality_impact": avg_quality_impact,
            "average_confidence": avg_confidence,
            "path_usage_distribution": path_usage,
            "recent_performance": {
                "decisions_last_30_days": len(recent_decisions),
                "average_cost_savings": recent_avg_savings,
            },
            "current_strategy": self.optimization_strategy.value,
            "cost_weight": self.cost_weight,
            "quality_weight": self.quality_weight,
        }
