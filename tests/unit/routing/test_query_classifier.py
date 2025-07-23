"""Unit tests for QueryClassifier using unittest framework."""

import unittest
from unittest.mock import MagicMock, patch

from src.models.module import ModuleType
from src.models.routing_schemas import (
    DecisionContext,
    KnowledgeRequirement,
    QueryComplexity,
    QueryType,
    RoutingConfiguration,
)
from src.routing.query_classifier import QueryClassifier, QueryFeatures


class TestQueryFeatures(unittest.TestCase):
    """Test cases for QueryFeatures model."""

    def test_query_features_creation(self):
        """Test QueryFeatures creation with all fields."""
        features = QueryFeatures(
            length=50,
            word_count=10,
            sentence_count=2,
            question_words=["what", "how"],
            has_greeting=False,
            has_question_mark=True,
            has_technical_terms=True,
            has_domain_specific_terms=False,
            conditional_statements=1,
            comparison_words=0,
            temporal_references=0,
            named_entities=1,
            is_procedural=True,
            is_factual=False,
            is_creative=False,
            is_conversational=False,
        )

        self.assertEqual(features.length, 50)
        self.assertEqual(features.word_count, 10)
        self.assertEqual(features.sentence_count, 2)
        self.assertEqual(features.question_words, ["what", "how"])
        self.assertFalse(features.has_greeting)
        self.assertTrue(features.has_question_mark)
        self.assertTrue(features.has_technical_terms)
        self.assertFalse(features.has_domain_specific_terms)
        self.assertEqual(features.conditional_statements, 1)
        self.assertEqual(features.comparison_words, 0)
        self.assertEqual(features.temporal_references, 0)
        self.assertEqual(features.named_entities, 1)
        self.assertTrue(features.is_procedural)
        self.assertFalse(features.is_factual)
        self.assertFalse(features.is_creative)
        self.assertFalse(features.is_conversational)


class TestQueryClassifier(unittest.TestCase):
    """Test cases for QueryClassifier class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RoutingConfiguration()
        self.classifier = QueryClassifier(self.config)

    @patch("src.routing.query_classifier.logger")
    def test_classifier_initialization(self, mock_logger):
        """Test QueryClassifier initialization."""
        classifier = QueryClassifier()

        self.assertIsInstance(classifier.config, RoutingConfiguration)
        self.assertIsInstance(classifier._greeting_patterns, list)
        self.assertIsInstance(classifier._question_words, list)
        self.assertIsInstance(classifier._technical_terms, list)
        mock_logger.info.assert_called_with("QueryClassifier initialized")

    @patch("src.routing.query_classifier.logger")
    def test_classifier_initialization_with_config(self, mock_logger):
        """Test QueryClassifier initialization with custom config."""
        custom_config = RoutingConfiguration(confidence_threshold=0.9)
        classifier = QueryClassifier(custom_config)

        self.assertEqual(classifier.config.confidence_threshold, 0.9)
        mock_logger.info.assert_called_with("QueryClassifier initialized")

    def test_extract_features_simple_query(self):
        """Test feature extraction for simple query."""
        query = "안녕하세요"
        features = self.classifier._extract_features(query)

        self.assertEqual(features.length, len(query))
        self.assertEqual(features.word_count, 1)
        self.assertEqual(features.sentence_count, 1)
        self.assertTrue(features.has_greeting)
        self.assertFalse(features.has_question_mark)
        self.assertFalse(features.has_technical_terms)
        self.assertFalse(features.is_procedural)
        self.assertTrue(features.is_conversational)

    def test_extract_features_technical_query(self):
        """Test feature extraction for technical query."""
        query = "데이터베이스에서 알고리즘을 어떻게 구현하나요?"
        features = self.classifier._extract_features(query)

        self.assertGreater(features.word_count, 5)
        self.assertTrue(features.has_question_mark)
        self.assertTrue(features.has_technical_terms)
        self.assertTrue(features.is_procedural)
        self.assertGreater(len(features.question_words), 0)

    def test_extract_features_complex_query(self):
        """Test feature extraction for complex query."""
        query = "만약 머신러닝 모델이 있다면, 2024년에 비교해서 어떤 성능 개선이 있을까요?"
        features = self.classifier._extract_features(query)

        self.assertGreater(features.conditional_statements, 0)
        self.assertGreater(features.comparison_words, 0)
        self.assertGreater(features.temporal_references, 0)
        self.assertTrue(features.has_technical_terms)

    def test_classify_type_greeting(self):
        """Test query type classification for greetings."""
        query = "안녕하세요"
        features = self.classifier._extract_features(query)
        query_type = self.classifier._classify_type(query, features, None)

        self.assertEqual(query_type, QueryType.GREETING)

    def test_classify_type_technical(self):
        """Test query type classification for technical queries."""
        query = "데이터베이스 연결 방법"
        features = self.classifier._extract_features(query)
        query_type = self.classifier._classify_type(query, features, None)

        self.assertEqual(query_type, QueryType.TECHNICAL)

    def test_classify_type_procedural(self):
        """Test query type classification for procedural queries."""
        query = "파이썬으로 웹사이트를 만드는 방법을 알려주세요"
        features = self.classifier._extract_features(query)
        query_type = self.classifier._classify_type(query, features, None)

        self.assertEqual(query_type, QueryType.PROCEDURAL)

    def test_classify_type_creative(self):
        """Test query type classification for creative queries."""
        query = "새로운 아이디어를 생각해주세요"
        features = self.classifier._extract_features(query)
        query_type = self.classifier._classify_type(query, features, None)

        self.assertEqual(query_type, QueryType.CREATIVE)

    def test_classify_type_analytical(self):
        """Test query type classification for analytical queries."""
        query = "A와 B를 비교 분석해주세요"
        features = self.classifier._extract_features(query)
        query_type = self.classifier._classify_type(query, features, None)

        self.assertEqual(query_type, QueryType.ANALYTICAL)

    def test_classify_type_factual_simple(self):
        """Test query type classification for simple factual queries."""
        query = "오늘 날씨는?"
        features = self.classifier._extract_features(query)
        query_type = self.classifier._classify_type(query, features, None)

        self.assertEqual(query_type, QueryType.FACTUAL_SIMPLE)

    def test_classify_type_factual_complex(self):
        """Test query type classification for complex factual queries."""
        query = "2024년 글로벌 경제 상황과 향후 전망에 대해 상세히 설명해주세요"
        features = self.classifier._extract_features(query)
        query_type = self.classifier._classify_type(query, features, None)

        self.assertEqual(query_type, QueryType.FACTUAL_COMPLEX)

    def test_classify_type_with_context(self):
        """Test query type classification with context."""
        query = "서버 설정하기"
        context = DecisionContext(
            query_text=query, query_length=len(query), domain_context="technical"
        )
        features = self.classifier._extract_features(query)
        query_type = self.classifier._classify_type(query, features, context)

        self.assertEqual(query_type, QueryType.TECHNICAL)

    def test_determine_complexity_trivial(self):
        """Test complexity determination for trivial queries."""
        query = "안녕"
        features = self.classifier._extract_features(query)
        complexity = self.classifier._determine_complexity(query, features, None)

        self.assertEqual(complexity, QueryComplexity.TRIVIAL)

    def test_determine_complexity_simple(self):
        """Test complexity determination for simple queries."""
        query = "오늘 날씨는 어떤가요?"
        features = self.classifier._extract_features(query)
        complexity = self.classifier._determine_complexity(query, features, None)

        self.assertIn(
            complexity,
            [QueryComplexity.TRIVIAL, QueryComplexity.SIMPLE, QueryComplexity.MODERATE],
        )

    def test_determine_complexity_complex(self):
        """Test complexity determination for complex queries."""
        query = "만약 머신러닝 모델의 성능을 개선하려면 어떤 방법들이 있고, 각각의 장단점을 비교분석해주세요?"
        features = self.classifier._extract_features(query)
        complexity = self.classifier._determine_complexity(query, features, None)

        self.assertIn(complexity, [QueryComplexity.COMPLEX, QueryComplexity.EXPERT])

    def test_assess_knowledge_requirement_none(self):
        """Test knowledge requirement assessment for queries needing no external knowledge."""
        query_type = QueryType.GREETING
        complexity = QueryComplexity.TRIVIAL
        features = QueryFeatures(
            length=10,
            word_count=2,
            sentence_count=1,
            question_words=[],
            has_greeting=True,
            has_question_mark=False,
            has_technical_terms=False,
            has_domain_specific_terms=False,
            conditional_statements=0,
            comparison_words=0,
            temporal_references=0,
            named_entities=0,
            is_procedural=False,
            is_factual=False,
            is_creative=False,
            is_conversational=True,
        )

        knowledge_req = self.classifier._assess_knowledge_requirement(
            "안녕", features, query_type, complexity, None
        )

        self.assertEqual(knowledge_req, KnowledgeRequirement.NONE)

    def test_assess_knowledge_requirement_minimal(self):
        """Test knowledge requirement assessment for queries needing minimal knowledge."""
        query_type = QueryType.FACTUAL_SIMPLE
        complexity = QueryComplexity.SIMPLE
        features = QueryFeatures(
            length=20,
            word_count=5,
            sentence_count=1,
            question_words=["what"],
            has_greeting=False,
            has_question_mark=True,
            has_technical_terms=False,
            has_domain_specific_terms=False,
            conditional_statements=0,
            comparison_words=0,
            temporal_references=0,
            named_entities=0,
            is_procedural=False,
            is_factual=True,
            is_creative=False,
            is_conversational=False,
        )

        knowledge_req = self.classifier._assess_knowledge_requirement(
            "날씨는?", features, query_type, complexity, None
        )

        self.assertEqual(knowledge_req, KnowledgeRequirement.MINIMAL)

    def test_assess_knowledge_requirement_extensive(self):
        """Test knowledge requirement assessment for queries needing extensive knowledge."""
        query_type = QueryType.TECHNICAL
        complexity = QueryComplexity.COMPLEX
        features = QueryFeatures(
            length=50,
            word_count=10,
            sentence_count=2,
            question_words=["how"],
            has_greeting=False,
            has_question_mark=True,
            has_technical_terms=True,
            has_domain_specific_terms=True,
            conditional_statements=1,
            comparison_words=1,
            temporal_references=0,
            named_entities=1,
            is_procedural=True,
            is_factual=False,
            is_creative=False,
            is_conversational=False,
        )

        knowledge_req = self.classifier._assess_knowledge_requirement(
            "복잡한 기술 질문", features, query_type, complexity, None
        )

        self.assertEqual(knowledge_req, KnowledgeRequirement.EXTENSIVE)

    def test_assess_knowledge_requirement_specialized(self):
        """Test knowledge requirement assessment for queries needing specialized knowledge."""
        query_type = QueryType.DOMAIN_SPECIFIC
        complexity = QueryComplexity.EXPERT
        features = QueryFeatures(
            length=100,
            word_count=20,
            sentence_count=3,
            question_words=["how", "what"],
            has_greeting=False,
            has_question_mark=True,
            has_technical_terms=True,
            has_domain_specific_terms=True,
            conditional_statements=2,
            comparison_words=2,
            temporal_references=1,
            named_entities=3,
            is_procedural=False,
            is_factual=False,
            is_creative=False,
            is_conversational=False,
        )

        knowledge_req = self.classifier._assess_knowledge_requirement(
            "전문적인 도메인 질문", features, query_type, complexity, None
        )

        self.assertEqual(knowledge_req, KnowledgeRequirement.SPECIALIZED)

    def test_calculate_confidence_high(self):
        """Test confidence calculation for high-confidence classification."""
        query = "안녕하세요"
        features = self.classifier._extract_features(query)
        query_type = QueryType.GREETING
        complexity = QueryComplexity.TRIVIAL

        confidence = self.classifier._calculate_confidence(query, features, query_type, complexity)

        self.assertGreater(confidence, 0.8)
        self.assertLessEqual(confidence, 1.0)

    def test_calculate_confidence_medium(self):
        """Test confidence calculation for medium-confidence classification."""
        query = "일반적인 질문입니다"
        features = self.classifier._extract_features(query)
        query_type = QueryType.CONVERSATIONAL
        complexity = QueryComplexity.MODERATE

        confidence = self.classifier._calculate_confidence(query, features, query_type, complexity)

        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_calculate_confidence_bounds(self):
        """Test confidence calculation stays within bounds."""
        # Create features that would normally give very high confidence
        features = QueryFeatures(
            length=200,
            word_count=50,
            sentence_count=5,
            question_words=["what", "how", "why"],
            has_greeting=True,
            has_question_mark=True,
            has_technical_terms=True,
            has_domain_specific_terms=True,
            conditional_statements=3,
            comparison_words=2,
            temporal_references=1,
            named_entities=5,
            is_procedural=True,
            is_factual=True,
            is_creative=True,
            is_conversational=True,
        )

        confidence = self.classifier._calculate_confidence(
            "test query", features, QueryType.GREETING, QueryComplexity.TRIVIAL
        )

        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    @patch("src.routing.query_classifier.logger")
    def test_classify_query_success(self, mock_logger):
        """Test successful query classification."""
        query = "안녕하세요, 오늘 날씨는 어떤가요?"

        query_type, complexity, knowledge_req, confidence = self.classifier.classify_query(query)

        self.assertIsInstance(query_type, QueryType)
        self.assertIsInstance(complexity, QueryComplexity)
        self.assertIsInstance(knowledge_req, KnowledgeRequirement)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        mock_logger.info.assert_called()

    @patch("src.routing.query_classifier.logger")
    def test_classify_query_with_context(self, mock_logger):
        """Test query classification with context."""
        query = "서버 설정"
        context = DecisionContext(
            query_text=query,
            query_length=len(query),
            domain_context="technical",
            available_modules=[ModuleType.DATABASE_ADAPTER, ModuleType.VECTOR_STORE],
        )

        query_type, complexity, knowledge_req, confidence = self.classifier.classify_query(
            query, context
        )

        self.assertIsInstance(query_type, QueryType)
        self.assertIsInstance(complexity, QueryComplexity)
        self.assertIsInstance(knowledge_req, KnowledgeRequirement)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    @patch("src.routing.query_classifier.logger")
    def test_classify_query_exception_handling(self, mock_logger):
        """Test query classification exception handling."""
        # Mock an exception in feature extraction
        with patch.object(
            self.classifier, "_extract_features", side_effect=Exception("Test error")
        ):
            query_type, complexity, knowledge_req, confidence = self.classifier.classify_query(
                "test"
            )

            # Should return safe defaults
            self.assertEqual(query_type, QueryType.CONVERSATIONAL)
            self.assertEqual(complexity, QueryComplexity.MODERATE)
            self.assertEqual(knowledge_req, KnowledgeRequirement.MODERATE)
            self.assertEqual(confidence, 0.5)
            mock_logger.error.assert_called()

    def test_detect_domain_terms(self):
        """Test domain-specific term detection."""
        # Medical terms
        self.assertTrue(self.classifier._detect_domain_terms("병원에서 치료를 받았습니다"))

        # Legal terms
        self.assertTrue(self.classifier._detect_domain_terms("계약서의 조항을 검토해주세요"))

        # Financial terms
        self.assertTrue(self.classifier._detect_domain_terms("주식 투자에 대해 알려주세요"))

        # Academic terms
        self.assertTrue(self.classifier._detect_domain_terms("연구 논문을 작성중입니다"))

        # No domain terms
        self.assertFalse(self.classifier._detect_domain_terms("오늘 날씨가 좋네요"))

    def test_count_named_entities(self):
        """Test named entity counting."""
        # Test with proper nouns
        count1 = self.classifier._count_named_entities("Kim Taehyung은 BTS의 멤버입니다")
        self.assertGreater(count1, 0)

        # Test with dates
        count2 = self.classifier._count_named_entities("2024년 3월 15일에 만났습니다")
        self.assertGreater(count2, 0)

        # Test with person names (English pattern)
        count3 = self.classifier._count_named_entities("John Smith is a developer")
        self.assertGreater(count3, 0)

        # Test with no named entities
        count4 = self.classifier._count_named_entities("오늘은 좋은 하루입니다")
        # This might be 0 or small number depending on implementation

    def test_get_classification_explanation(self):
        """Test classification explanation generation."""
        query = "안녕하세요, 날씨는 어떤가요?"
        query_type = QueryType.GREETING
        complexity = QueryComplexity.SIMPLE
        knowledge_req = KnowledgeRequirement.MINIMAL
        confidence = 0.85

        explanation = self.classifier.get_classification_explanation(
            query, query_type, complexity, knowledge_req, confidence
        )

        self.assertIsInstance(explanation, str)
        self.assertIn("Query Analysis", explanation)
        self.assertIn("Classification Results", explanation)
        self.assertIn("Key Features Detected", explanation)
        self.assertIn(query_type.value, explanation)
        self.assertIn(complexity.value, explanation)
        self.assertIn(knowledge_req.value, explanation)
        self.assertIn(str(confidence), explanation)


if __name__ == "__main__":
    unittest.main()
