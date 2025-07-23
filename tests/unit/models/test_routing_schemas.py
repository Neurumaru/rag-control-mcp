"""Unit tests for routing schemas using unittest framework."""

import unittest
from datetime import datetime
from uuid import uuid4

from pydantic import ValidationError

from src.models.module import DataType, ModuleType
from src.models.routing_schemas import (
    INTELLIGENT_ROUTING_PATTERNS,
    DecisionContext,
    KnowledgeRequirement,
    QueryComplexity,
    QueryType,
    RoutingConfiguration,
    RoutingDecision,
    RoutingMetrics,
    RoutingModuleType,
    RoutingPath,
)


class TestQueryEnums(unittest.TestCase):
    """Test cases for query-related enums."""

    def test_query_type_enum(self):
        """Test QueryType enum values."""
        self.assertEqual(QueryType.GREETING, "greeting")
        self.assertEqual(QueryType.FACTUAL_SIMPLE, "factual_simple")
        self.assertEqual(QueryType.FACTUAL_COMPLEX, "factual_complex")
        self.assertEqual(QueryType.PROCEDURAL, "procedural")
        self.assertEqual(QueryType.ANALYTICAL, "analytical")
        self.assertEqual(QueryType.CREATIVE, "creative")
        self.assertEqual(QueryType.TECHNICAL, "technical")
        self.assertEqual(QueryType.CONVERSATIONAL, "conversational")
        self.assertEqual(QueryType.DOMAIN_SPECIFIC, "domain_specific")
        self.assertEqual(QueryType.MULTIMODAL, "multimodal")

    def test_query_complexity_enum(self):
        """Test QueryComplexity enum values."""
        self.assertEqual(QueryComplexity.TRIVIAL, "trivial")
        self.assertEqual(QueryComplexity.SIMPLE, "simple")
        self.assertEqual(QueryComplexity.MODERATE, "moderate")
        self.assertEqual(QueryComplexity.COMPLEX, "complex")
        self.assertEqual(QueryComplexity.EXPERT, "expert")

    def test_knowledge_requirement_enum(self):
        """Test KnowledgeRequirement enum values."""
        self.assertEqual(KnowledgeRequirement.NONE, "none")
        self.assertEqual(KnowledgeRequirement.MINIMAL, "minimal")
        self.assertEqual(KnowledgeRequirement.MODERATE, "moderate")
        self.assertEqual(KnowledgeRequirement.EXTENSIVE, "extensive")
        self.assertEqual(KnowledgeRequirement.SPECIALIZED, "specialized")

    def test_routing_path_enum(self):
        """Test RoutingPath enum values."""
        self.assertEqual(RoutingPath.DIRECT_LLM, "direct_llm")
        self.assertEqual(RoutingPath.SIMPLE_RAG, "simple_rag")
        self.assertEqual(RoutingPath.ENHANCED_RAG, "enhanced_rag")
        self.assertEqual(RoutingPath.HYBRID_PARALLEL, "hybrid_parallel")
        self.assertEqual(RoutingPath.MULTI_STEP, "multi_step")
        self.assertEqual(RoutingPath.SPECIALIZED, "specialized")


class TestRoutingDecision(unittest.TestCase):
    """Test cases for RoutingDecision model."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_data = {
            "query_id": "test_query_123",
            "selected_path": RoutingPath.DIRECT_LLM,
            "confidence": 0.85,
            "query_type": QueryType.GREETING,
            "complexity": QueryComplexity.TRIVIAL,
            "knowledge_requirement": KnowledgeRequirement.NONE,
            "estimated_cost": 0.001,
            "estimated_time": 0.5,
            "reasoning": "Simple greeting detected, direct LLM response optimal",
        }

    def test_routing_decision_creation(self):
        """Test RoutingDecision creation with valid data."""
        decision = RoutingDecision(**self.valid_data)

        self.assertEqual(decision.query_id, "test_query_123")
        self.assertEqual(decision.selected_path, RoutingPath.DIRECT_LLM)
        self.assertEqual(decision.confidence, 0.85)
        self.assertEqual(decision.query_type, QueryType.GREETING)
        self.assertEqual(decision.complexity, QueryComplexity.TRIVIAL)
        self.assertEqual(decision.knowledge_requirement, KnowledgeRequirement.NONE)
        self.assertEqual(decision.estimated_cost, 0.001)
        self.assertEqual(decision.estimated_time, 0.5)
        self.assertEqual(
            decision.reasoning, "Simple greeting detected, direct LLM response optimal"
        )

    def test_routing_decision_defaults(self):
        """Test RoutingDecision with default values."""
        decision = RoutingDecision(**self.valid_data)

        self.assertEqual(decision.cost_savings, 0.0)
        self.assertEqual(decision.alternative_paths, [])
        self.assertEqual(decision.context_factors, {})
        self.assertEqual(decision.model_version, "1.0")
        self.assertIsInstance(decision.decision_timestamp, datetime)

    def test_routing_decision_with_alternatives(self):
        """Test RoutingDecision with alternative paths."""
        data = self.valid_data.copy()
        data["alternative_paths"] = [RoutingPath.SIMPLE_RAG, RoutingPath.ENHANCED_RAG]
        data["context_factors"] = {"user_preference": "fast", "system_load": 0.3}

        decision = RoutingDecision(**data)

        self.assertEqual(len(decision.alternative_paths), 2)
        self.assertIn(RoutingPath.SIMPLE_RAG, decision.alternative_paths)
        self.assertEqual(decision.context_factors["user_preference"], "fast")

    def test_confidence_validation(self):
        """Test confidence validation."""
        # Valid confidence
        data = self.valid_data.copy()
        data["confidence"] = 0.0
        decision = RoutingDecision(**data)
        self.assertEqual(decision.confidence, 0.0)

        data["confidence"] = 1.0
        decision = RoutingDecision(**data)
        self.assertEqual(decision.confidence, 1.0)

        # Invalid confidence - should raise ValidationError
        data["confidence"] = -0.1
        with self.assertRaises(ValidationError):
            RoutingDecision(**data)

        data["confidence"] = 1.1
        with self.assertRaises(ValidationError):
            RoutingDecision(**data)

    def test_cost_savings_validation(self):
        """Test cost savings validation."""
        # Valid cost savings
        data = self.valid_data.copy()
        data["cost_savings"] = 0.9
        decision = RoutingDecision(**data)
        self.assertEqual(decision.cost_savings, 0.9)

        data["cost_savings"] = 0.0
        decision = RoutingDecision(**data)
        self.assertEqual(decision.cost_savings, 0.0)

        # Invalid cost savings - should raise ValidationError
        data["cost_savings"] = -0.1
        with self.assertRaises(ValidationError):
            RoutingDecision(**data)

    def test_routing_decision_serialization(self):
        """Test RoutingDecision serialization."""
        decision = RoutingDecision(**self.valid_data)
        data_dict = decision.dict()

        self.assertEqual(data_dict["query_id"], "test_query_123")
        self.assertEqual(data_dict["selected_path"], "direct_llm")
        self.assertEqual(data_dict["confidence"], 0.85)


class TestDecisionContext(unittest.TestCase):
    """Test cases for DecisionContext model."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_data = {
            "query_text": "안녕하세요, 오늘 날씨는 어떤가요?",
            "query_length": 20,
        }

    def test_decision_context_creation(self):
        """Test DecisionContext creation with valid data."""
        context = DecisionContext(**self.valid_data)

        self.assertEqual(context.query_text, "안녕하세요, 오늘 날씨는 어떤가요?")
        self.assertEqual(context.query_length, 20)
        self.assertEqual(context.query_language, "ko")  # default

    def test_decision_context_defaults(self):
        """Test DecisionContext with default values."""
        context = DecisionContext(**self.valid_data)

        self.assertEqual(context.query_language, "ko")
        self.assertIsNone(context.user_id)
        self.assertIsNone(context.session_id)
        self.assertEqual(context.conversation_history, [])
        self.assertEqual(context.available_modules, [])
        self.assertEqual(context.system_load, 0.5)
        self.assertIsNone(context.cost_budget)
        self.assertIsNone(context.max_response_time)
        self.assertEqual(context.quality_threshold, 0.8)
        self.assertEqual(context.cost_preference, 0.5)

    def test_decision_context_with_modules(self):
        """Test DecisionContext with available modules."""
        data = self.valid_data.copy()
        data["available_modules"] = [ModuleType.VECTOR_STORE, ModuleType.LLM_GENERATOR]
        data["specialized_knowledge"] = ["AI", "Machine Learning"]

        context = DecisionContext(**data)

        self.assertEqual(len(context.available_modules), 2)
        self.assertIn(ModuleType.VECTOR_STORE, context.available_modules)
        self.assertEqual(len(context.specialized_knowledge), 2)
        self.assertIn("AI", context.specialized_knowledge)

    def test_decision_context_conversation_history(self):
        """Test DecisionContext with conversation history."""
        data = self.valid_data.copy()
        data["conversation_history"] = ["안녕하세요", "날씨가 좋네요", "감사합니다"]
        data["user_id"] = "user_123"
        data["session_id"] = "session_456"

        context = DecisionContext(**data)

        self.assertEqual(len(context.conversation_history), 3)
        self.assertEqual(context.user_id, "user_123")
        self.assertEqual(context.session_id, "session_456")


class TestRoutingMetrics(unittest.TestCase):
    """Test cases for RoutingMetrics model."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_data = {
            "decision_id": "decision_123",
            "actual_response_time": 1.2,
            "actual_cost": 0.005,
            "quality_score": 0.9,
            "routing_accuracy": True,
            "cost_efficiency": 0.85,
            "baseline_cost": 0.02,
            "cost_savings_realized": 0.75,
            "quality_vs_baseline": 0.95,
        }

    def test_routing_metrics_creation(self):
        """Test RoutingMetrics creation with valid data."""
        metrics = RoutingMetrics(**self.valid_data)

        self.assertEqual(metrics.decision_id, "decision_123")
        self.assertEqual(metrics.actual_response_time, 1.2)
        self.assertEqual(metrics.actual_cost, 0.005)
        self.assertEqual(metrics.quality_score, 0.9)
        self.assertTrue(metrics.routing_accuracy)
        self.assertEqual(metrics.cost_efficiency, 0.85)

    def test_routing_metrics_defaults(self):
        """Test RoutingMetrics with default values."""
        metrics = RoutingMetrics(**self.valid_data)

        self.assertIsNone(metrics.user_satisfaction)
        self.assertIsNone(metrics.notes)
        self.assertIsInstance(metrics.feedback_timestamp, datetime)

    def test_routing_metrics_quality_score_validation(self):
        """Test quality score validation."""
        # Valid quality scores
        data = self.valid_data.copy()
        data["quality_score"] = 0.0
        metrics = RoutingMetrics(**data)
        self.assertEqual(metrics.quality_score, 0.0)

        data["quality_score"] = 1.0
        metrics = RoutingMetrics(**data)
        self.assertEqual(metrics.quality_score, 1.0)

        # Invalid quality scores
        data["quality_score"] = -0.1
        with self.assertRaises(ValidationError):
            RoutingMetrics(**data)

        data["quality_score"] = 1.1
        with self.assertRaises(ValidationError):
            RoutingMetrics(**data)


class TestRoutingConfiguration(unittest.TestCase):
    """Test cases for RoutingConfiguration model."""

    def test_routing_configuration_defaults(self):
        """Test RoutingConfiguration with default values."""
        config = RoutingConfiguration()

        self.assertEqual(config.confidence_threshold, 0.7)
        self.assertEqual(config.quality_threshold, 0.8)
        self.assertEqual(config.cost_weight, 0.3)
        self.assertEqual(config.quality_weight, 0.7)
        self.assertIsNone(config.max_cost_per_query)
        self.assertEqual(config.max_response_time, 30.0)
        self.assertEqual(config.fallback_path, RoutingPath.DIRECT_LLM)
        self.assertEqual(config.classification_model, "gpt-3.5-turbo")
        self.assertEqual(config.decision_model, "gpt-4")
        self.assertTrue(config.enable_learning)
        self.assertTrue(config.enable_caching)
        self.assertEqual(config.cache_ttl, 3600)

    def test_routing_configuration_custom_values(self):
        """Test RoutingConfiguration with custom values."""
        config = RoutingConfiguration(
            confidence_threshold=0.9,
            quality_threshold=0.85,
            cost_weight=0.4,
            quality_weight=0.6,
            max_cost_per_query=0.1,
            max_response_time=15.0,
            fallback_path=RoutingPath.SIMPLE_RAG,
            classification_model="claude-3-haiku",
            decision_model="claude-3-sonnet",
            enable_learning=False,
            enable_caching=False,
            cache_ttl=1800,
        )

        self.assertEqual(config.confidence_threshold, 0.9)
        self.assertEqual(config.quality_threshold, 0.85)
        self.assertEqual(config.cost_weight, 0.4)
        self.assertEqual(config.quality_weight, 0.6)
        self.assertEqual(config.max_cost_per_query, 0.1)
        self.assertEqual(config.max_response_time, 15.0)
        self.assertEqual(config.fallback_path, RoutingPath.SIMPLE_RAG)
        self.assertEqual(config.classification_model, "claude-3-haiku")
        self.assertEqual(config.decision_model, "claude-3-sonnet")
        self.assertFalse(config.enable_learning)
        self.assertFalse(config.enable_caching)
        self.assertEqual(config.cache_ttl, 1800)


class TestRoutingModuleType(unittest.TestCase):
    """Test cases for RoutingModuleType enum."""

    def test_routing_module_type_values(self):
        """Test RoutingModuleType enum values."""
        self.assertEqual(RoutingModuleType.QUERY_CLASSIFIER, "query_classifier")
        self.assertEqual(RoutingModuleType.ROUTE_CONTROLLER, "route_controller")
        self.assertEqual(RoutingModuleType.KNOWLEDGE_ASSESSOR, "knowledge_assessor")
        self.assertEqual(RoutingModuleType.DECISION_ENGINE, "decision_engine")
        self.assertEqual(RoutingModuleType.COST_OPTIMIZER, "cost_optimizer")
        self.assertEqual(RoutingModuleType.ROUTING_ADAPTER, "routing_adapter")
        self.assertEqual(RoutingModuleType.PARALLEL_PROCESSOR, "parallel_processor")
        self.assertEqual(RoutingModuleType.RESULT_AGGREGATOR, "result_aggregator")
        self.assertEqual(RoutingModuleType.QUALITY_EVALUATOR, "quality_evaluator")
        self.assertEqual(RoutingModuleType.FEEDBACK_PROCESSOR, "feedback_processor")


class TestIntelligentRoutingPatterns(unittest.TestCase):
    """Test cases for intelligent routing patterns."""

    def test_routing_patterns_structure(self):
        """Test the structure of routing patterns."""
        self.assertIsInstance(INTELLIGENT_ROUTING_PATTERNS, dict)
        self.assertIn("query_analysis", INTELLIGENT_ROUTING_PATTERNS)
        self.assertIn("routing_decision", INTELLIGENT_ROUTING_PATTERNS)
        self.assertIn("cost_optimization", INTELLIGENT_ROUTING_PATTERNS)
        self.assertIn("parallel_processing", INTELLIGENT_ROUTING_PATTERNS)

    def test_query_analysis_pattern(self):
        """Test query analysis pattern."""
        pattern = INTELLIGENT_ROUTING_PATTERNS["query_analysis"]
        self.assertIsInstance(pattern, list)
        self.assertEqual(len(pattern), 2)

        # Check first pattern: TEXT -> QUERY_CLASSIFIER -> STRUCTURED
        first_pattern = pattern[0]
        self.assertEqual(first_pattern[0], DataType.TEXT)
        self.assertEqual(first_pattern[1], RoutingModuleType.QUERY_CLASSIFIER)
        self.assertEqual(first_pattern[2], DataType.STRUCTURED)

        # Check second pattern: STRUCTURED -> KNOWLEDGE_ASSESSOR -> STRUCTURED
        second_pattern = pattern[1]
        self.assertEqual(second_pattern[0], DataType.STRUCTURED)
        self.assertEqual(second_pattern[1], RoutingModuleType.KNOWLEDGE_ASSESSOR)
        self.assertEqual(second_pattern[2], DataType.STRUCTURED)

    def test_routing_decision_pattern(self):
        """Test routing decision pattern."""
        pattern = INTELLIGENT_ROUTING_PATTERNS["routing_decision"]
        self.assertIsInstance(pattern, list)
        self.assertEqual(len(pattern), 2)

        # Check patterns involve DECISION_ENGINE and ROUTE_CONTROLLER
        decision_pattern = pattern[0]
        self.assertEqual(decision_pattern[1], RoutingModuleType.DECISION_ENGINE)

        controller_pattern = pattern[1]
        self.assertEqual(controller_pattern[1], RoutingModuleType.ROUTE_CONTROLLER)

    def test_cost_optimization_pattern(self):
        """Test cost optimization pattern."""
        pattern = INTELLIGENT_ROUTING_PATTERNS["cost_optimization"]
        self.assertIsInstance(pattern, list)
        self.assertEqual(len(pattern), 1)

        # Check COST_OPTIMIZER pattern
        cost_pattern = pattern[0]
        self.assertEqual(cost_pattern[1], RoutingModuleType.COST_OPTIMIZER)

    def test_parallel_processing_pattern(self):
        """Test parallel processing pattern."""
        pattern = INTELLIGENT_ROUTING_PATTERNS["parallel_processing"]
        self.assertIsInstance(pattern, list)
        self.assertEqual(len(pattern), 2)

        # Check PARALLEL_PROCESSOR and RESULT_AGGREGATOR patterns
        parallel_pattern = pattern[0]
        self.assertEqual(parallel_pattern[1], RoutingModuleType.PARALLEL_PROCESSOR)

        aggregator_pattern = pattern[1]
        self.assertEqual(aggregator_pattern[1], RoutingModuleType.RESULT_AGGREGATOR)


if __name__ == "__main__":
    unittest.main()
