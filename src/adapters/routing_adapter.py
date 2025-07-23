"""MCP adapter for intelligent routing system integration."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel

from models.request import MCPRequest, MCPResponse
from models.routing_schemas import (
    DecisionContext,
    RoutingConfiguration,
    RoutingDecision,
)
from routing.cost_optimizer import CostOptimizer
from routing.decision_engine import DecisionEngine
from routing.knowledge_assessor import KnowledgeAssessor
from routing.route_controller import RouteController
from utils.logger import logger

from .base_adapter import BaseAdapter


class RoutingRequest(BaseModel):
    """Request for intelligent routing decision."""

    query: str
    context: Optional[DecisionContext] = None
    override_config: Optional[RoutingConfiguration] = None
    request_id: Optional[str] = None


class RoutingResponse(BaseModel):
    """Response with routing decision and detailed analysis."""

    request_id: str
    routing_decision: RoutingDecision
    detailed_analysis: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]


class RoutingAdapter(BaseAdapter):
    """MCP adapter for the intelligent routing system."""

    def __init__(self, config: Optional[RoutingConfiguration] = None):
        """Initialize the routing adapter.

        Args:
            config: Routing configuration settings
        """
        self.config = config or RoutingConfiguration()

        # Initialize routing components
        self.route_controller = RouteController(self.config)
        self.decision_engine = DecisionEngine(self.config)
        self.cost_optimizer = CostOptimizer(self.config)
        self.knowledge_assessor = KnowledgeAssessor()

        # Adapter state
        self._connected = False
        self._active_requests: Dict[str, datetime] = {}
        self._processing_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
        }

        super().__init__()
        logger.info("RoutingAdapter initialized with intelligent routing capabilities")

    async def connect(self) -> bool:
        """Connect the routing adapter."""
        try:
            # Initialize routing system components
            self._connected = True

            # Warm up the system with a test query
            await self._warmup_system()

            logger.info("RoutingAdapter connected successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to connect RoutingAdapter: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> bool:
        """Disconnect the routing adapter."""
        try:
            self._connected = False

            # Clean up active requests
            self._active_requests.clear()

            logger.info("RoutingAdapter disconnected")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting RoutingAdapter: {e}")
            return False

    async def process_request(self, request: MCPRequest) -> MCPResponse:
        """Process an MCP request for intelligent routing.

        Args:
            request: MCP request containing routing query

        Returns:
            MCP response with routing decision
        """
        request_start = datetime.now()
        request_id = str(uuid4())

        try:
            if not self._connected:
                raise Exception("RoutingAdapter not connected")

            # Track active request
            self._active_requests[request_id] = request_start
            self._processing_stats["total_requests"] += 1

            # Parse routing request
            routing_request = self._parse_routing_request(request, request_id)

            # Execute intelligent routing decision
            routing_response = await self._execute_routing_decision(routing_request)

            # Create MCP response
            mcp_response = MCPResponse(
                request_id=request.request_id,
                success=True,
                data=routing_response.dict(),
                metadata={
                    "routing_adapter_version": "1.0",
                    "processing_time": (datetime.now() - request_start).total_seconds(),
                    "selected_path": routing_response.routing_decision.selected_path.value,
                    "cost_savings": routing_response.routing_decision.cost_savings,
                },
            )

            # Update statistics
            self._processing_stats["successful_requests"] += 1
            self._update_response_time_stats(request_start)

            logger.info(
                f"Routing request processed successfully: {routing_response.routing_decision.selected_path.value} "
                f"(savings: {routing_response.routing_decision.cost_savings:.1f}%)"
            )

            return mcp_response

        except Exception as e:
            logger.error(f"Error processing routing request: {e}")
            self._processing_stats["failed_requests"] += 1

            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
                metadata={"routing_adapter_version": "1.0"},
            )

        finally:
            # Clean up active request tracking
            if request_id in self._active_requests:
                del self._active_requests[request_id]

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the routing adapter."""
        try:
            health_status = {
                "status": "healthy" if self._connected else "disconnected",
                "connected": self._connected,
                "active_requests": len(self._active_requests),
                "processing_stats": self._processing_stats.copy(),
                "component_status": {
                    "route_controller": "operational",
                    "decision_engine": "operational",
                    "cost_optimizer": "operational",
                    "knowledge_assessor": "operational",
                },
                "configuration": {
                    "confidence_threshold": self.config.confidence_threshold,
                    "quality_threshold": self.config.quality_threshold,
                    "max_response_time": self.config.max_response_time,
                    "fallback_path": self.config.fallback_path.value,
                },
                "last_check": datetime.now().isoformat(),
            }

            # Test routing system with a simple query
            if self._connected:
                try:
                    test_request = RoutingRequest(
                        query="test health check", request_id="health_check"
                    )
                    test_start = datetime.now()
                    await self._execute_routing_decision(test_request)
                    test_time = (datetime.now() - test_start).total_seconds()

                    health_status["system_test"] = {
                        "status": "passed",
                        "response_time": test_time,
                    }
                except Exception as e:
                    health_status["system_test"] = {"status": "failed", "error": str(e)}
                    health_status["status"] = "degraded"

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "last_check": datetime.now().isoformat(),
            }

    def get_capabilities(self) -> List[str]:
        """Get routing adapter capabilities."""
        return [
            "intelligent_routing",
            "cost_optimization",
            "quality_assessment",
            "decision_explanation",
            "performance_monitoring",
            "adaptive_learning",
            "llm_assisted_decisions",
            "knowledge_requirement_analysis",
            "multi_strategy_optimization",
            "real_time_metrics",
        ]

    async def _execute_routing_decision(self, routing_request: RoutingRequest) -> RoutingResponse:
        """Execute the complete intelligent routing decision process."""

        # Step 1: Make routing decision
        routing_decision = await self.route_controller.make_routing_decision(
            routing_request.query, routing_request.context
        )

        # Step 2: Get detailed knowledge assessment
        knowledge_assessment = self.knowledge_assessor.assess_knowledge_requirement(
            routing_request.query,
            routing_decision.query_type,
            routing_decision.complexity,
            routing_request.context,
        )

        # Step 3: Get cost optimization analysis
        possible_paths = [routing_decision.selected_path] + routing_decision.alternative_paths
        cost_optimization = self.cost_optimizer.optimize_routing_cost(
            routing_request.query,
            possible_paths,
            routing_decision.knowledge_requirement,
            routing_decision.complexity,
            routing_request.context,
        )

        # Step 4: Get detailed decision analysis from decision engine
        decision_analysis = await self.decision_engine.make_intelligent_decision(
            routing_request.query,
            routing_decision.query_type,
            routing_decision.complexity,
            routing_decision.knowledge_requirement,
            routing_request.context,
            routing_decision.confidence,
        )

        # Step 5: Compile detailed analysis
        detailed_analysis = {
            "query_classification": {
                "type": routing_decision.query_type.value,
                "complexity": routing_decision.complexity.value,
                "confidence": routing_decision.confidence,
            },
            "knowledge_assessment": {
                "requirement_level": knowledge_assessment.requirement_level.value,
                "categories": [cat.value for cat in knowledge_assessment.knowledge_categories],
                "recommended_sources": [
                    src.value for src in knowledge_assessment.recommended_sources
                ],
                "temporal_sensitivity": knowledge_assessment.temporal_sensitivity,
                "domain_specificity": knowledge_assessment.domain_specificity,
                "knowledge_gaps": knowledge_assessment.knowledge_gaps,
            },
            "cost_analysis": {
                "recommended_path": cost_optimization.recommended_path.value,
                "cost_savings": cost_optimization.cost_savings,
                "quality_impact": cost_optimization.quality_impact,
                "optimization_confidence": cost_optimization.confidence,
                "cost_breakdown": cost_optimization.cost_breakdown.dict(),
                "risk_assessment": {
                    "level": cost_optimization.risk_level,
                    "factors": cost_optimization.risk_factors,
                },
            },
            "decision_engine_analysis": decision_analysis,
        }

        # Step 6: Generate performance metrics
        performance_metrics = {
            "routing_efficiency": {
                "selected_path": routing_decision.selected_path.value,
                "estimated_cost": routing_decision.estimated_cost,
                "estimated_time": routing_decision.estimated_time,
                "cost_savings_percentage": routing_decision.cost_savings,
            },
            "quality_predictions": {
                "expected_quality": knowledge_assessment.assessment_reasoning,
                "confidence_score": routing_decision.confidence,
                "quality_vs_cost_ratio": cost_optimization.cost_breakdown.cost_efficiency,
            },
            "optimization_impact": {
                "cost_reduction": cost_optimization.cost_savings,
                "quality_change": cost_optimization.quality_impact,
                "efficiency_gain": f"{cost_optimization.cost_breakdown.cost_efficiency:.2f}",
            },
        }

        # Step 7: Generate recommendations
        recommendations = self._generate_recommendations(
            routing_decision,
            knowledge_assessment,
            cost_optimization,
            routing_request.context,
        )

        return RoutingResponse(
            request_id=routing_request.request_id or str(uuid4()),
            routing_decision=routing_decision,
            detailed_analysis=detailed_analysis,
            performance_metrics=performance_metrics,
            recommendations=recommendations,
        )

    def _parse_routing_request(self, request: MCPRequest, request_id: str) -> RoutingRequest:
        """Parse MCP request into routing request."""

        try:
            # Extract query from request data
            query = request.data.get("query", "")
            if not query:
                raise ValueError("Query is required for routing decisions")

            # Extract context if provided
            context_data = request.data.get("context", {})
            context = None
            if context_data:
                context = DecisionContext(**context_data)

            # Extract configuration overrides
            config_data = request.data.get("config", {})
            override_config = None
            if config_data:
                override_config = RoutingConfiguration(**config_data)

            return RoutingRequest(
                query=query,
                context=context,
                override_config=override_config,
                request_id=request_id,
            )

        except Exception as e:
            raise ValueError(f"Invalid routing request format: {e}")

    def _generate_recommendations(
        self,
        routing_decision: RoutingDecision,
        knowledge_assessment,
        cost_optimization,
        context: Optional[DecisionContext],
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""

        recommendations = []

        # Cost optimization recommendations
        if cost_optimization.cost_savings > 70:
            recommendations.append(
                f"💰 우수한 비용 절약: {cost_optimization.cost_savings:.1f}% 절약으로 매우 효율적인 라우팅"
            )
        elif cost_optimization.cost_savings > 40:
            recommendations.append(
                f"💡 적절한 비용 효율성: {cost_optimization.cost_savings:.1f}% 절약 달성"
            )

        # Quality recommendations
        if routing_decision.confidence < 0.7:
            recommendations.append(
                "⚠️ 낮은 분류 신뢰도: 더 구체적인 질문으로 정확도를 높일 수 있습니다"
            )

        # Knowledge gap recommendations
        if knowledge_assessment.knowledge_gaps:
            recommendations.append(
                f"📚 지식 격차 주의: {len(knowledge_assessment.knowledge_gaps)}개 잠재적 제한사항 존재"
            )

        # Time sensitivity recommendations
        if knowledge_assessment.temporal_sensitivity > 0.7:
            recommendations.append("⏰ 시간 민감 정보: 실시간 데이터 소스 활용을 고려하세요")

        # Domain specificity recommendations
        if knowledge_assessment.domain_specificity > 0.8:
            recommendations.append("🎯 전문 도메인: 특화된 지식 베이스 활용이 권장됩니다")

        # Alternative path suggestions
        if routing_decision.alternative_paths:
            alt_paths = [path.value for path in routing_decision.alternative_paths[:2]]
            recommendations.append(f"🔄 대안 경로: {', '.join(alt_paths)} 고려 가능")

        # Context-based recommendations
        if context:
            if context.cost_budget and context.cost_budget < routing_decision.estimated_cost:
                recommendations.append("💸 예산 초과: 더 경제적인 라우팅 경로를 고려하세요")

            if (
                context.max_response_time
                and context.max_response_time < routing_decision.estimated_time
            ):
                recommendations.append("⚡ 응답 시간 주의: 더 빠른 처리 경로 필요")

        return recommendations[:5]  # Limit to top 5 recommendations

    async def _warmup_system(self) -> None:
        """Warm up the routing system with test queries."""

        test_queries = [
            "안녕하세요",  # Greeting
            "Python에서 리스트를 정렬하는 방법은?",  # Procedural
            "2024년 AI 트렌드 분석해줘",  # Complex analytical
        ]

        for query in test_queries:
            try:
                test_request = RoutingRequest(query=query, request_id="warmup")
                await self._execute_routing_decision(test_request)
            except Exception as e:
                logger.warning(f"Warmup query failed: {e}")

        logger.info("Routing system warmup completed")

    def _update_response_time_stats(self, request_start: datetime) -> None:
        """Update response time statistics."""

        response_time = (datetime.now() - request_start).total_seconds()
        current_avg = self._processing_stats["average_response_time"]
        total_successful = self._processing_stats["successful_requests"]

        # Calculate new average response time
        if total_successful == 1:
            new_avg = response_time
        else:
            new_avg = ((current_avg * (total_successful - 1)) + response_time) / total_successful

        self._processing_stats["average_response_time"] = new_avg

    async def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""

        # Get statistics from all components
        controller_stats = self.route_controller.get_routing_statistics()
        decision_stats = self.decision_engine.get_decision_statistics()
        optimizer_stats = self.cost_optimizer.get_optimization_statistics()

        return {
            "adapter_stats": self._processing_stats.copy(),
            "routing_controller": controller_stats,
            "decision_engine": decision_stats,
            "cost_optimizer": optimizer_stats,
            "system_health": await self.health_check(),
            "active_requests": len(self._active_requests),
            "configuration": {
                "confidence_threshold": self.config.confidence_threshold,
                "quality_threshold": self.config.quality_threshold,
                "cost_weight": self.config.cost_weight,
                "quality_weight": self.config.quality_weight,
            },
        }


# Example usage function for testing
async def example_usage():
    """Example usage of the RoutingAdapter."""

    # Initialize adapter
    config = RoutingConfiguration(
        confidence_threshold=0.8,
        quality_threshold=0.85,
        cost_weight=0.3,
        quality_weight=0.7,
    )

    adapter = RoutingAdapter(config)

    # Connect adapter
    await adapter.connect()

    # Create example request
    context = DecisionContext(
        query_text="2024년의 최신 AI 트렌드와 발전 방향은?",
        user_id="example_user",
        quality_threshold=0.9,
        cost_preference=0.4,
        domain_context="technology",
    )

    request = MCPRequest(
        request_id="example_001",
        method="intelligent_routing",
        data={
            "query": "2024년의 최신 AI 트렌드와 발전 방향을 분석해주세요",
            "context": context.dict(),
        },
    )

    # Process request
    response = await adapter.process_request(request)

    # Print results
    if response.success:
        routing_data = response.data
        print(f"Selected Path: {routing_data['routing_decision']['selected_path']}")
        print(f"Cost Savings: {routing_data['routing_decision']['cost_savings']:.1f}%")
        print(f"Confidence: {routing_data['routing_decision']['confidence']:.2f}")
        print(f"Reasoning: {routing_data['routing_decision']['reasoning']}")

    # Disconnect adapter
    await adapter.disconnect()


if __name__ == "__main__":
    # Run example
    import asyncio

    asyncio.run(example_usage())
