"""LLM-based decision engine for intelligent routing path selection."""

import asyncio
import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from ..models.routing_schemas import (
    DecisionContext,
    KnowledgeRequirement,
    QueryComplexity,
    QueryType,
    RoutingConfiguration,
    RoutingPath,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DecisionStrategy(str, Enum):
    """Strategy for making routing decisions."""

    RULE_BASED = "rule_based"
    LLM_FIRST = "llm_first"
    HYBRID = "hybrid"
    COST_OPTIMIZED = "cost_optimized"
    QUALITY_FOCUSED = "quality_focused"


class LLMProvider(str, Enum):
    """Supported LLM providers for decision making."""

    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    CLAUDE_3_OPUS = "claude_3_opus"
    CLAUDE_3_SONNET = "claude_3_sonnet"
    CLAUDE_3_HAIKU = "claude_3_haiku"


class DecisionPrompt(BaseModel):
    """Structured prompt for LLM decision making."""

    system_prompt: str
    user_prompt: str
    query_context: Dict[str, Any]
    expected_format: Dict[str, str]


class DecisionEngine:
    """LLM-based decision engine for routing path selection."""

    def __init__(
        self,
        config: Optional[RoutingConfiguration] = None,
        strategy: DecisionStrategy = DecisionStrategy.HYBRID,
        primary_llm: LLMProvider = LLMProvider.OPENAI_GPT4,
    ):
        """Initialize the decision engine.

        Args:
            config: Routing configuration
            strategy: Decision making strategy
            primary_llm: Primary LLM provider for decisions
        """
        self.config = config or RoutingConfiguration()
        self.strategy = strategy
        self.primary_llm = primary_llm

        # Decision history for learning
        self._decision_history: List[Dict[str, Any]] = []

        # Initialize templates and models
        self._prompt_templates = self._initialize_decision_templates()
        self._cost_models = self._initialize_cost_models()

        logger.info(
            f"DecisionEngine initialized with {strategy.value} strategy and {primary_llm.value}"
        )

    async def make_routing_decision(
        self,
        query: str,
        query_type: QueryType,
        complexity: QueryComplexity,
        knowledge_req: KnowledgeRequirement,
        context: Optional[DecisionContext] = None,
    ) -> Dict[str, Any]:
        """Make an intelligent routing decision using the configured strategy.

        Args:
            query: User query text
            query_type: Classified query type
            complexity: Query complexity level
            knowledge_req: Knowledge requirement level
            context: Additional decision context

        Returns:
            Dictionary containing routing decision and metadata
        """
        start_time = datetime.now()

        try:
            if self.strategy == DecisionStrategy.LLM_FIRST:
                decision_result = await self._make_llm_decision(
                    query, query_type, complexity, knowledge_req, context
                )
                decision_result["decision_strategy"] = "llm_first"

            elif self.strategy == DecisionStrategy.RULE_BASED:
                decision_result = await self._make_rule_based_decision(
                    query, query_type, complexity, knowledge_req, context
                )
                decision_result["decision_strategy"] = "rule_based"

            elif self.strategy == DecisionStrategy.HYBRID:
                decision_result = await self._make_hybrid_decision(
                    query, query_type, complexity, knowledge_req, context
                )
                decision_result["decision_strategy"] = "hybrid"

            elif self.strategy == DecisionStrategy.COST_OPTIMIZED:
                decision_result = await self._make_cost_optimized_decision(
                    query, query_type, complexity, knowledge_req, context
                )
                decision_result["decision_strategy"] = "cost_optimized"

            else:  # QUALITY_FOCUSED
                decision_result = await self._make_quality_focused_decision(
                    query, query_type, complexity, knowledge_req, context
                )
                decision_result["decision_strategy"] = "quality_focused"

            # Add timing information
            decision_time = (datetime.now() - start_time).total_seconds()
            decision_result["decision_time"] = decision_time

            # Record decision for learning
            await self._record_decision(decision_result)

            logger.info(
                f"Routing decision made: {decision_result['selected_path']} "
                f"(confidence: {decision_result['confidence']:.2f}, "
                f"time: {decision_time:.3f}s)"
            )

            return decision_result

        except Exception as e:
            logger.error(f"Error in decision engine: {e}")
            return await self._make_fallback_decision(query, query_type, complexity, knowledge_req)

    async def _make_llm_decision(
        self,
        query: str,
        query_type: QueryType,
        complexity: QueryComplexity,
        knowledge_req: KnowledgeRequirement,
        context: Optional[DecisionContext],
    ) -> Dict[str, Any]:
        """Make decision using LLM as primary method."""

        # Create structured prompt
        prompt = self._create_decision_prompt(query, query_type, complexity, knowledge_req, context)

        # Call LLM for decision
        llm_response = await self._call_llm_for_decision(prompt)

        # Parse and validate response
        decision_result = self._parse_llm_decision(llm_response)

        return decision_result

    async def _make_rule_based_decision(
        self,
        query: str,
        query_type: QueryType,
        complexity: QueryComplexity,
        knowledge_req: KnowledgeRequirement,
        context: Optional[DecisionContext],
    ) -> Dict[str, Any]:
        """Make decision using rule-based logic."""

        # Simple rule-based logic
        if knowledge_req == KnowledgeRequirement.NONE:
            selected_path = RoutingPath.DIRECT_LLM
            cost_savings = 90.0
            confidence = 0.9
            reasoning = "No external knowledge needed - direct LLM response optimal"

        elif knowledge_req == KnowledgeRequirement.MINIMAL:
            if complexity <= QueryComplexity.SIMPLE:
                selected_path = RoutingPath.DIRECT_LLM
                cost_savings = 85.0
                confidence = 0.8
                reasoning = "Simple query with minimal knowledge needs - direct LLM suitable"
            else:
                selected_path = RoutingPath.SIMPLE_RAG
                cost_savings = 60.0
                confidence = 0.75
                reasoning = "Moderate complexity requires basic retrieval"

        elif knowledge_req == KnowledgeRequirement.MODERATE:
            selected_path = RoutingPath.SIMPLE_RAG
            cost_savings = 55.0
            confidence = 0.8
            reasoning = "Moderate knowledge requirement - basic RAG appropriate"

        elif knowledge_req == KnowledgeRequirement.EXTENSIVE:
            if complexity >= QueryComplexity.COMPLEX:
                selected_path = RoutingPath.ENHANCED_RAG
                cost_savings = 30.0
                confidence = 0.85
                reasoning = "Complex query with extensive knowledge needs - enhanced RAG required"
            else:
                selected_path = RoutingPath.SIMPLE_RAG
                cost_savings = 45.0
                confidence = 0.75
                reasoning = "Extensive knowledge but moderate complexity - simple RAG sufficient"

        else:  # SPECIALIZED
            selected_path = RoutingPath.SPECIALIZED
            cost_savings = 25.0
            confidence = 0.8
            reasoning = "Specialized knowledge domain requires specialized processing"

        # Get alternative paths
        alternative_paths = self._get_alternative_paths(selected_path)

        return {
            "selected_path": selected_path,
            "confidence": confidence,
            "reasoning": reasoning,
            "cost_savings": cost_savings,
            "alternative_paths": alternative_paths,
            "decision_factors": {
                "rule_based": True,
                "query_type": query_type.value,
                "complexity": complexity.value,
                "knowledge_req": knowledge_req.value,
            },
        }

    async def _make_hybrid_decision(
        self,
        query: str,
        query_type: QueryType,
        complexity: QueryComplexity,
        knowledge_req: KnowledgeRequirement,
        context: Optional[DecisionContext],
    ) -> Dict[str, Any]:
        """Make decision using rule-based logic validated by LLM."""

        # First, get rule-based decision
        rule_decision = await self._make_rule_based_decision(
            query, query_type, complexity, knowledge_req, context
        )

        # Create validation prompt
        validation_prompt = self._create_validation_prompt(
            query, rule_decision, query_type, complexity, knowledge_req
        )

        # Get LLM validation
        llm_validation = await self._call_llm_for_validation(validation_prompt)

        # Combine decisions
        final_decision = self._combine_decisions(rule_decision, llm_validation)

        return final_decision

    async def _make_cost_optimized_decision(
        self,
        query: str,
        query_type: QueryType,
        complexity: QueryComplexity,
        knowledge_req: KnowledgeRequirement,
        context: Optional[DecisionContext],
    ) -> Dict[str, Any]:
        """Make decision optimized for cost efficiency."""

        # Aggressive cost optimization - prefer cheaper paths
        if query_type == QueryType.GREETING or knowledge_req == KnowledgeRequirement.NONE:
            selected_path = RoutingPath.DIRECT_LLM
            cost_savings = 95.0
            confidence = 0.95
            reasoning = "Cost-optimized: Direct LLM for minimal knowledge queries"

        elif (
            knowledge_req == KnowledgeRequirement.MINIMAL and complexity <= QueryComplexity.MODERATE
        ):
            selected_path = RoutingPath.DIRECT_LLM
            cost_savings = 88.0
            confidence = 0.8
            reasoning = "Cost-optimized: Direct LLM acceptable for simple queries"

        elif knowledge_req <= KnowledgeRequirement.MODERATE:
            selected_path = RoutingPath.SIMPLE_RAG
            cost_savings = 65.0
            confidence = 0.75
            reasoning = "Cost-optimized: Simple RAG for moderate knowledge needs"

        else:
            selected_path = RoutingPath.ENHANCED_RAG
            cost_savings = 35.0
            confidence = 0.7
            reasoning = "Cost-optimized: Enhanced RAG only when necessary"

        alternative_paths = self._get_alternative_paths(selected_path)

        return {
            "selected_path": selected_path,
            "confidence": confidence,
            "reasoning": reasoning,
            "cost_savings": cost_savings,
            "alternative_paths": alternative_paths,
            "decision_factors": {"cost_optimized": True, "cost_preference": "maximum_savings"},
        }

    async def _make_quality_focused_decision(
        self,
        query: str,
        query_type: QueryType,
        complexity: QueryComplexity,
        knowledge_req: KnowledgeRequirement,
        context: Optional[DecisionContext],
    ) -> Dict[str, Any]:
        """Make decision optimized for answer quality."""

        # Quality-first approach - prefer more sophisticated paths
        if knowledge_req == KnowledgeRequirement.NONE and query_type == QueryType.GREETING:
            selected_path = RoutingPath.DIRECT_LLM
            cost_savings = 90.0
            confidence = 0.9
            reasoning = "Quality-focused: Direct LLM sufficient for greetings"

        elif knowledge_req <= KnowledgeRequirement.MINIMAL:
            selected_path = RoutingPath.SIMPLE_RAG
            cost_savings = 50.0
            confidence = 0.85
            reasoning = "Quality-focused: RAG provides better accuracy than direct LLM"

        elif (
            complexity >= QueryComplexity.COMPLEX or knowledge_req >= KnowledgeRequirement.EXTENSIVE
        ):
            if query_type in [QueryType.TECHNICAL, QueryType.DOMAIN_SPECIFIC]:
                selected_path = RoutingPath.SPECIALIZED
                cost_savings = 20.0
                confidence = 0.9
                reasoning = "Quality-focused: Specialized processing for technical queries"
            elif complexity == QueryComplexity.EXPERT:
                selected_path = RoutingPath.MULTI_STEP
                cost_savings = 15.0
                confidence = 0.85
                reasoning = "Quality-focused: Multi-step processing for expert-level queries"
            else:
                selected_path = RoutingPath.ENHANCED_RAG
                cost_savings = 30.0
                confidence = 0.88
                reasoning = "Quality-focused: Enhanced RAG for complex queries"
        else:
            selected_path = RoutingPath.ENHANCED_RAG
            cost_savings = 35.0
            confidence = 0.8
            reasoning = "Quality-focused: Enhanced RAG as default for quality"

        alternative_paths = self._get_alternative_paths(selected_path)

        return {
            "selected_path": selected_path,
            "confidence": confidence,
            "reasoning": reasoning,
            "cost_savings": cost_savings,
            "alternative_paths": alternative_paths,
            "decision_factors": {"quality_focused": True, "quality_preference": "maximum_accuracy"},
        }

    def _create_decision_prompt(
        self,
        query: str,
        query_type: QueryType,
        complexity: QueryComplexity,
        knowledge_req: KnowledgeRequirement,
        context: Optional[DecisionContext],
    ) -> DecisionPrompt:
        """Create structured prompt for LLM decision making."""

        system_prompt = """You are an intelligent routing system for RAG (Retrieval-Augmented Generation) applications.
Your goal is to select the most cost-effective and appropriate processing path for user queries.

Available routing paths:
1. DIRECT_LLM: Direct LLM response (fastest, lowest cost, ~90% cost savings)
2. SIMPLE_RAG: Basic retrieval + generation (moderate cost, ~60% cost savings)
3. ENHANCED_RAG: Advanced RAG with reranking (higher cost, ~30% cost savings)
4. HYBRID_PARALLEL: Parallel processing (balanced approach, ~50% cost savings)
5. MULTI_STEP: Sequential multi-step processing (highest cost, ~10% cost savings)
6. SPECIALIZED: Domain-specific processing (variable cost)

Consider:
- Cost efficiency vs. answer quality
- Query complexity and knowledge requirements
- User context and preferences
- System performance constraints
"""

        context_info = ""
        if context:
            context_info = f"""
User Context:
- Quality threshold: {context.quality_threshold}
- Cost preference: {context.cost_preference} (0=cost-focused, 1=quality-focused)
- Max response time: {context.max_response_time or 'none'} seconds
- Domain: {context.domain_context or 'general'}
- Conversation history length: {len(context.conversation_history)}
"""

        user_prompt = f"""
Analyze this query and recommend the optimal routing path:

Query: "{query}"

Classification Results:
- Type: {query_type.value}
- Complexity: {complexity.value}  
- Knowledge Requirement: {knowledge_req.value}
{context_info}

Provide your decision in JSON format:
{{
  "selected_path": "DIRECT_LLM|SIMPLE_RAG|ENHANCED_RAG|HYBRID_PARALLEL|MULTI_STEP|SPECIALIZED",
  "confidence": 0.0-1.0,
  "reasoning": "detailed explanation",
  "cost_savings": 0-100,
  "alternative_paths": ["path1", "path2"],
  "risk_assessment": "low|medium|high",
  "quality_prediction": 0.0-1.0
}}
"""

        return DecisionPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            query_context={
                "query": query,
                "type": query_type.value,
                "complexity": complexity.value,
                "knowledge_req": knowledge_req.value,
            },
            expected_format={
                "selected_path": "string",
                "confidence": "float",
                "reasoning": "string",
                "cost_savings": "float",
                "alternative_paths": "list",
                "risk_assessment": "string",
                "quality_prediction": "float",
            },
        )

    async def _call_llm_for_decision(self, prompt: DecisionPrompt) -> Dict[str, Any]:
        """Call LLM API for decision making (simulated for now)."""

        # Simulate LLM API call delay
        await asyncio.sleep(0.1)

        # Simulated LLM response based on prompt analysis
        # In real implementation, this would call OpenAI, Claude, etc.

        query = prompt.query_context["query"].lower()
        query_type = prompt.query_context["type"]
        complexity = prompt.query_context["complexity"]
        knowledge_req = prompt.query_context["knowledge_req"]

        # Simulate intelligent LLM decision making
        if "안녕" in query or "hello" in query:
            return {
                "selected_path": "DIRECT_LLM",
                "confidence": 0.95,
                "reasoning": "Simple greeting can be handled directly by LLM without external knowledge",
                "cost_savings": 90.0,
                "alternative_paths": [],
                "risk_assessment": "low",
                "quality_prediction": 0.9,
                "prompt_tokens": 250,
                "completion_tokens": 80,
            }
        elif "how to" in query or "어떻게" in query:
            return {
                "selected_path": "SIMPLE_RAG",
                "confidence": 0.8,
                "reasoning": "Procedural question benefits from knowledge retrieval but doesn't need complex processing",
                "cost_savings": 55.0,
                "alternative_paths": ["ENHANCED_RAG"],
                "risk_assessment": "medium",
                "quality_prediction": 0.85,
                "prompt_tokens": 250,
                "completion_tokens": 95,
            }
        elif complexity == "expert" or "technical" in query_type:
            return {
                "selected_path": "SPECIALIZED",
                "confidence": 0.85,
                "reasoning": "Complex technical query requires specialized processing for accurate results",
                "cost_savings": 25.0,
                "alternative_paths": ["ENHANCED_RAG", "MULTI_STEP"],
                "risk_assessment": "high",
                "quality_prediction": 0.92,
                "prompt_tokens": 250,
                "completion_tokens": 110,
            }
        else:
            return {
                "selected_path": "ENHANCED_RAG",
                "confidence": 0.75,
                "reasoning": "Moderate complexity query benefits from enhanced RAG with reranking",
                "cost_savings": 35.0,
                "alternative_paths": ["SIMPLE_RAG", "HYBRID_PARALLEL"],
                "risk_assessment": "medium",
                "quality_prediction": 0.88,
                "prompt_tokens": 250,
                "completion_tokens": 100,
            }

    async def _call_llm_for_validation(self, prompt: str) -> Dict[str, Any]:
        """Call LLM for validating rule-based decisions."""

        await asyncio.sleep(0.05)  # Shorter delay for validation

        # Simulated validation response
        return {
            "agrees_with_rule": True,
            "confidence_in_agreement": 0.8,
            "suggested_path": None,  # No alternative suggestion
            "validation_reasoning": "Rule-based decision appears appropriate for this query type",
            "concerns": [],
        }

    def _parse_llm_decision(self, llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate LLM decision response."""

        # Map LLM path strings to enum values
        path_mapping = {
            "DIRECT_LLM": RoutingPath.DIRECT_LLM,
            "SIMPLE_RAG": RoutingPath.SIMPLE_RAG,
            "ENHANCED_RAG": RoutingPath.ENHANCED_RAG,
            "HYBRID_PARALLEL": RoutingPath.HYBRID_PARALLEL,
            "MULTI_STEP": RoutingPath.MULTI_STEP,
            "SPECIALIZED": RoutingPath.SPECIALIZED,
        }

        selected_path_str = llm_response.get("selected_path", "SIMPLE_RAG")
        selected_path = path_mapping.get(selected_path_str, RoutingPath.SIMPLE_RAG)

        # Parse alternative paths
        alternative_paths = []
        for alt_path_str in llm_response.get("alternative_paths", []):
            if alt_path_str in path_mapping:
                alternative_paths.append(path_mapping[alt_path_str])

        return {
            "selected_path": selected_path,
            "confidence": float(llm_response.get("confidence", 0.7)),
            "reasoning": llm_response.get("reasoning", "LLM-based decision"),
            "cost_savings": float(llm_response.get("cost_savings", 50.0)),
            "alternative_paths": alternative_paths,
            "risk_assessment": llm_response.get("risk_assessment", "medium"),
            "quality_prediction": float(llm_response.get("quality_prediction", 0.8)),
            "decision_factors": {
                "llm_provider": self.primary_llm.value,
                "model_reasoning": llm_response.get("reasoning", ""),
                "risk_level": llm_response.get("risk_assessment", "medium"),
            },
        }

    def _combine_decisions(
        self, rule_decision: Dict[str, Any], llm_validation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine rule-based decision with LLM validation."""

        if llm_validation.get("agrees_with_rule", True):
            # LLM agrees - boost confidence
            combined_confidence = min(0.95, rule_decision["confidence"] + 0.1)

            return {
                **rule_decision,
                "confidence": combined_confidence,
                "reasoning": f"{rule_decision['reasoning']} (LLM validated)",
                "validation_agreement": True,
            }
        else:
            # LLM disagrees - use LLM suggestion with explanation
            suggested_path = llm_validation.get("suggested_path")
            if suggested_path:
                return {
                    "selected_path": suggested_path,
                    "confidence": 0.8,
                    "reasoning": f"LLM override: {llm_validation.get('validation_reasoning', '')}",
                    "cost_savings": self._estimate_cost_savings(suggested_path),
                    "alternative_paths": [rule_decision["selected_path"]],
                    "validation_agreement": False,
                    "original_rule_decision": rule_decision["selected_path"],
                }

        # Fallback to rule decision with noted disagreement
        return {
            **rule_decision,
            "reasoning": f"{rule_decision['reasoning']} (LLM disagreement noted)",
            "validation_agreement": False,
        }

    def _get_alternative_paths(self, selected_path: RoutingPath) -> List[RoutingPath]:
        """Get reasonable alternative paths for a selected path."""

        alternatives = {
            RoutingPath.DIRECT_LLM: [RoutingPath.SIMPLE_RAG],
            RoutingPath.SIMPLE_RAG: [RoutingPath.DIRECT_LLM, RoutingPath.ENHANCED_RAG],
            RoutingPath.ENHANCED_RAG: [RoutingPath.SIMPLE_RAG, RoutingPath.MULTI_STEP],
            RoutingPath.HYBRID_PARALLEL: [RoutingPath.ENHANCED_RAG, RoutingPath.SIMPLE_RAG],
            RoutingPath.MULTI_STEP: [RoutingPath.ENHANCED_RAG, RoutingPath.SPECIALIZED],
            RoutingPath.SPECIALIZED: [RoutingPath.ENHANCED_RAG, RoutingPath.MULTI_STEP],
        }

        return alternatives.get(selected_path, [RoutingPath.SIMPLE_RAG])

    def _estimate_cost_savings(self, path: RoutingPath) -> float:
        """Estimate cost savings for a routing path."""

        savings = {
            RoutingPath.DIRECT_LLM: 90.0,
            RoutingPath.SIMPLE_RAG: 60.0,
            RoutingPath.ENHANCED_RAG: 30.0,
            RoutingPath.HYBRID_PARALLEL: 50.0,
            RoutingPath.MULTI_STEP: 10.0,
            RoutingPath.SPECIALIZED: 25.0,
        }

        return savings.get(path, 40.0)

    async def _make_fallback_decision(
        self,
        query: str,
        query_type: QueryType,
        complexity: QueryComplexity,
        knowledge_req: KnowledgeRequirement,
    ) -> Dict[str, Any]:
        """Create fallback decision when all methods fail."""

        return {
            "selected_path": RoutingPath.SIMPLE_RAG,
            "confidence": 0.5,
            "reasoning": "Fallback decision due to system error - using safe default",
            "cost_savings": 50.0,
            "alternative_paths": [RoutingPath.ENHANCED_RAG],
            "decision_factors": {
                "fallback_used": True,
                "original_error": "Decision engine failure",
            },
        }

    async def _record_decision(self, decision_result: Dict[str, Any]) -> None:
        """Record decision for learning and performance tracking."""

        decision_record = {
            "timestamp": datetime.now().isoformat(),
            "selected_path": (
                decision_result["selected_path"].value
                if hasattr(decision_result["selected_path"], "value")
                else str(decision_result["selected_path"])
            ),
            "confidence": decision_result["confidence"],
            "cost_savings": decision_result["cost_savings"],
            "strategy": decision_result.get("decision_strategy", "unknown"),
            "decision_time": decision_result.get("decision_time", 0.0),
        }

        self._decision_history.append(decision_record)

        # Keep only recent decisions (last 500)
        if len(self._decision_history) > 500:
            self._decision_history = self._decision_history[-500:]

    def _initialize_decision_templates(self) -> Dict[str, str]:
        """Initialize decision prompt templates."""
        return {
            "classification_validation": "Validate the classification of this query...",
            "cost_optimization": "Optimize the routing path for cost efficiency...",
            "quality_assurance": "Ensure routing path meets quality requirements...",
            "context_adaptation": "Adapt routing decision based on user context...",
        }

    def _initialize_cost_models(self) -> Dict[str, Dict[str, float]]:
        """Initialize cost-benefit models for different paths."""
        return {
            "cost_multipliers": {
                "direct_llm": 0.1,
                "simple_rag": 0.4,
                "enhanced_rag": 1.0,
                "hybrid_parallel": 0.6,
                "multi_step": 1.5,
                "specialized": 2.0,
            },
            "quality_scores": {
                "direct_llm": 0.7,
                "simple_rag": 0.8,
                "enhanced_rag": 0.9,
                "hybrid_parallel": 0.85,
                "multi_step": 0.95,
                "specialized": 0.92,
            },
        }

    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get decision engine performance statistics."""

        if not self._decision_history:
            return {"message": "No decision history available"}

        total_decisions = len(self._decision_history)
        avg_confidence = sum(d["confidence"] for d in self._decision_history) / total_decisions
        avg_cost_savings = sum(d["cost_savings"] for d in self._decision_history) / total_decisions
        avg_decision_time = (
            sum(d["decision_time"] for d in self._decision_history) / total_decisions
        )

        # Strategy usage
        strategy_usage = {}
        for decision in self._decision_history:
            strategy = decision["strategy"]
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1

        # Path selection frequency
        path_usage = {}
        for decision in self._decision_history:
            path = decision["selected_path"]
            path_usage[path] = path_usage.get(path, 0) + 1

        return {
            "total_decisions": total_decisions,
            "average_confidence": avg_confidence,
            "average_cost_savings": avg_cost_savings,
            "average_decision_time": avg_decision_time,
            "strategy_usage": strategy_usage,
            "path_usage": path_usage,
            "current_strategy": self.strategy.value,
            "primary_llm": self.primary_llm.value,
        }

    def _create_validation_prompt(
        self,
        query: str,
        rule_decision: Dict[str, Any],
        query_type: QueryType,
        complexity: QueryComplexity,
        knowledge_req: KnowledgeRequirement,
    ) -> str:
        """Create prompt for LLM validation of rule-based decisions."""

        return f"""
Validate this routing decision:

Query: "{query}"
Query Type: {query_type.value}
Complexity: {complexity.value}
Knowledge Requirement: {knowledge_req.value}

Rule-based Decision:
- Path: {rule_decision['selected_path']}
- Confidence: {rule_decision['confidence']}
- Reasoning: {rule_decision['reasoning']}
- Cost Savings: {rule_decision['cost_savings']}%

Do you agree with this decision? If not, what would you recommend?

Respond in JSON format:
{{
  "agrees_with_rule": true/false,
  "confidence_in_agreement": 0.0-1.0,
  "suggested_path": "alternative_path_if_disagreeing",
  "validation_reasoning": "explanation",
  "concerns": ["list", "of", "concerns"]
}}
"""


# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing the decision engine
    pass
