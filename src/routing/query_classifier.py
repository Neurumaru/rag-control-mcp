"""Query classification module for intelligent routing decisions."""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from ..models.routing_schemas import (
    DecisionContext,
    KnowledgeRequirement,
    QueryComplexity,
    QueryType,
    RoutingConfiguration,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class QueryFeatures(BaseModel):
    """Extracted features from a query for classification."""

    # Basic features
    length: int
    word_count: int
    sentence_count: int
    question_words: List[str]

    # Linguistic features
    has_greeting: bool
    has_question_mark: bool
    has_technical_terms: bool
    has_domain_specific_terms: bool

    # Complexity indicators
    conditional_statements: int
    comparison_words: int
    temporal_references: int
    named_entities: int

    # Intent indicators
    is_procedural: bool
    is_factual: bool
    is_creative: bool
    is_conversational: bool


class QueryClassifier:
    """Intelligent query classifier using rule-based and ML approaches."""

    def __init__(self, config: Optional[RoutingConfiguration] = None):
        """Initialize the query classifier.

        Args:
            config: Routing configuration settings
        """
        self.config = config or RoutingConfiguration()

        # Classification patterns
        self._greeting_patterns = [
            r"^(안녕|hello|hi|hey|좋은|반가워)",
            r"(처음|만나서|반갑)",
            r"^(감사|고마워|thank)",
        ]

        self._question_words = [
            "무엇",
            "언제",
            "어디",
            "누구",
            "왜",
            "어떻게",
            "얼마",
            "what",
            "when",
            "where",
            "who",
            "why",
            "how",
            "which",
        ]

        self._technical_terms = [
            "알고리즘",
            "데이터베이스",
            "API",
            "함수",
            "변수",
            "클래스",
            "algorithm",
            "database",
            "function",
            "variable",
            "class",
            "서버",
            "클라이언트",
            "프로토콜",
            "인터페이스",
        ]

        self._procedural_indicators = [
            "방법",
            "어떻게",
            "단계",
            "절차",
            "과정",
            "how to",
            "step",
            "process",
        ]

        self._creative_indicators = [
            "아이디어",
            "창작",
            "디자인",
            "제안",
            "생각해줘",
            "만들어",
            "idea",
            "create",
            "design",
            "suggest",
            "brainstorm",
        ]

        self._comparison_words = [
            "비교",
            "차이",
            "vs",
            "대신",
            "보다",
            "than",
            "compare",
            "difference",
        ]

        logger.info("QueryClassifier initialized")

    def classify_query(
        self, query: str, context: Optional[DecisionContext] = None
    ) -> Tuple[QueryType, QueryComplexity, KnowledgeRequirement, float]:
        """Classify a query and return type, complexity, knowledge requirement, and confidence.

        Args:
            query: User query text
            context: Additional context for classification

        Returns:
            Tuple of (query_type, complexity, knowledge_requirement, confidence)
        """
        try:
            # Extract features
            features = self._extract_features(query)

            # Classify query type
            query_type = self._classify_type(query, features, context)

            # Determine complexity
            complexity = self._determine_complexity(query, features, context)

            # Assess knowledge requirement
            knowledge_req = self._assess_knowledge_requirement(
                query, features, query_type, complexity, context
            )

            # Calculate confidence
            confidence = self._calculate_confidence(query, features, query_type, complexity)

            logger.info(
                f"Query classified: type={query_type}, complexity={complexity}, "
                f"knowledge_req={knowledge_req}, confidence={confidence:.2f}"
            )

            return query_type, complexity, knowledge_req, confidence

        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            # Fallback to safe defaults
            return (
                QueryType.CONVERSATIONAL,
                QueryComplexity.MODERATE,
                KnowledgeRequirement.MODERATE,
                0.5,
            )

    def _extract_features(self, query: str) -> QueryFeatures:
        """Extract features from query text for classification."""
        query_lower = query.lower()

        # Basic features
        length = len(query)
        words = query.split()
        word_count = len(words)
        sentences = re.split(r"[.!?]+", query)
        sentence_count = len([s for s in sentences if s.strip()])

        # Question words
        question_words = [w for w in self._question_words if w in query_lower]

        # Linguistic features
        has_greeting = any(re.search(pattern, query_lower) for pattern in self._greeting_patterns)
        has_question_mark = "?" in query
        has_technical_terms = any(term in query_lower for term in self._technical_terms)
        has_domain_specific_terms = self._detect_domain_terms(query_lower)

        # Complexity indicators
        conditional_statements = len(re.findall(r"(if|만약|경우|when|then)", query_lower))
        comparison_words = sum(1 for word in self._comparison_words if word in query_lower)
        temporal_references = len(re.findall(r"(때|시간|날|년|월|when|time|date)", query_lower))
        named_entities = self._count_named_entities(query)

        # Intent indicators
        is_procedural = any(indicator in query_lower for indicator in self._procedural_indicators)
        is_factual = len(question_words) > 0 and not is_procedural
        is_creative = any(indicator in query_lower for indicator in self._creative_indicators)
        is_conversational = has_greeting or not (is_factual or is_procedural or is_creative)

        return QueryFeatures(
            length=length,
            word_count=word_count,
            sentence_count=sentence_count,
            question_words=question_words,
            has_greeting=has_greeting,
            has_question_mark=has_question_mark,
            has_technical_terms=has_technical_terms,
            has_domain_specific_terms=has_domain_specific_terms,
            conditional_statements=conditional_statements,
            comparison_words=comparison_words,
            temporal_references=temporal_references,
            named_entities=named_entities,
            is_procedural=is_procedural,
            is_factual=is_factual,
            is_creative=is_creative,
            is_conversational=is_conversational,
        )

    def _classify_type(
        self, query: str, features: QueryFeatures, context: Optional[DecisionContext]
    ) -> QueryType:
        """Classify the query type based on features and context."""
        query_lower = query.lower()

        # Greeting check
        if features.has_greeting and features.word_count <= 5:
            return QueryType.GREETING

        # Technical query check
        if features.has_technical_terms or (context and context.domain_context == "technical"):
            return QueryType.TECHNICAL

        # Procedural query check
        if features.is_procedural:
            return QueryType.PROCEDURAL

        # Creative query check
        if features.is_creative:
            return QueryType.CREATIVE

        # Analytical query check (comparison, analysis)
        if features.comparison_words > 0 or "분석" in query_lower or "analyze" in query_lower:
            return QueryType.ANALYTICAL

        # Domain-specific check
        if features.has_domain_specific_terms:
            return QueryType.DOMAIN_SPECIFIC

        # Factual query classification
        if features.is_factual:
            if features.word_count <= 10 and features.sentence_count == 1:
                return QueryType.FACTUAL_SIMPLE
            else:
                return QueryType.FACTUAL_COMPLEX

        # Default to conversational
        return QueryType.CONVERSATIONAL

    def _determine_complexity(
        self, query: str, features: QueryFeatures, context: Optional[DecisionContext]
    ) -> QueryComplexity:
        """Determine query complexity level."""
        complexity_score = 0

        # Length-based scoring
        if features.word_count > 20:
            complexity_score += 2
        elif features.word_count > 10:
            complexity_score += 1

        # Multiple sentences increase complexity
        if features.sentence_count > 2:
            complexity_score += 2
        elif features.sentence_count > 1:
            complexity_score += 1

        # Conditional statements add complexity
        complexity_score += features.conditional_statements

        # Comparison and analysis add complexity
        complexity_score += features.comparison_words

        # Technical terms add complexity
        if features.has_technical_terms:
            complexity_score += 1

        # Domain-specific terms add complexity
        if features.has_domain_specific_terms:
            complexity_score += 1

        # Multiple question words indicate complex query
        if len(features.question_words) > 2:
            complexity_score += 1

        # Map score to complexity level
        if complexity_score == 0:
            return QueryComplexity.TRIVIAL
        elif complexity_score <= 2:
            return QueryComplexity.SIMPLE
        elif complexity_score <= 4:
            return QueryComplexity.MODERATE
        elif complexity_score <= 6:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.EXPERT

    def _assess_knowledge_requirement(
        self,
        query: str,
        features: QueryFeatures,
        query_type: QueryType,
        complexity: QueryComplexity,
        context: Optional[DecisionContext],
    ) -> KnowledgeRequirement:
        """Assess external knowledge requirement level."""

        # Greetings and simple conversational queries need no external knowledge
        if query_type == QueryType.GREETING:
            return KnowledgeRequirement.NONE

        # Creative queries typically don't need external knowledge
        if query_type == QueryType.CREATIVE and complexity <= QueryComplexity.MODERATE:
            return KnowledgeRequirement.NONE

        # Simple factual queries might need minimal verification
        if query_type == QueryType.FACTUAL_SIMPLE and complexity <= QueryComplexity.SIMPLE:
            return KnowledgeRequirement.MINIMAL

        # Technical and domain-specific queries usually need external knowledge
        if query_type in [QueryType.TECHNICAL, QueryType.DOMAIN_SPECIFIC]:
            if complexity >= QueryComplexity.COMPLEX:
                return KnowledgeRequirement.SPECIALIZED
            else:
                return KnowledgeRequirement.EXTENSIVE

        # Complex analytical queries need extensive knowledge
        if query_type == QueryType.ANALYTICAL or complexity >= QueryComplexity.COMPLEX:
            return KnowledgeRequirement.EXTENSIVE

        # Procedural queries often need specific knowledge
        if query_type == QueryType.PROCEDURAL:
            return KnowledgeRequirement.MODERATE

        # Named entities often require external knowledge
        if features.named_entities > 0:
            return KnowledgeRequirement.MODERATE

        # Default based on complexity
        if complexity >= QueryComplexity.MODERATE:
            return KnowledgeRequirement.MODERATE
        else:
            return KnowledgeRequirement.MINIMAL

    def _calculate_confidence(
        self,
        query: str,
        features: QueryFeatures,
        query_type: QueryType,
        complexity: QueryComplexity,
    ) -> float:
        """Calculate confidence score for the classification."""
        confidence = 0.7  # Base confidence

        # Strong indicators boost confidence
        if features.has_greeting and query_type == QueryType.GREETING:
            confidence += 0.2

        if features.is_procedural and query_type == QueryType.PROCEDURAL:
            confidence += 0.15

        if features.has_technical_terms and query_type == QueryType.TECHNICAL:
            confidence += 0.15

        if features.is_creative and query_type == QueryType.CREATIVE:
            confidence += 0.15

        # Clear question indicators
        if features.has_question_mark and len(features.question_words) > 0:
            confidence += 0.1

        # Reduce confidence for ambiguous cases
        if features.is_conversational and query_type != QueryType.CONVERSATIONAL:
            confidence -= 0.1

        # Very short queries are harder to classify accurately
        if features.word_count <= 3:
            confidence -= 0.1

        # Very long queries can be ambiguous
        if features.word_count > 50:
            confidence -= 0.05

        return max(0.0, min(1.0, confidence))

    def _detect_domain_terms(self, query: str) -> bool:
        """Detect domain-specific terminology."""
        domain_terms = [
            # Medical
            "증상",
            "질병",
            "치료",
            "의학",
            "병원",
            # Legal
            "법",
            "계약",
            "소송",
            "판결",
            "조항",
            # Financial
            "투자",
            "주식",
            "금융",
            "대출",
            "이자",
            # Academic
            "연구",
            "논문",
            "이론",
            "실험",
            "가설",
        ]

        return any(term in query for term in domain_terms)

    def _count_named_entities(self, query: str) -> int:
        """Count potential named entities (simplified approach)."""
        # Look for capitalized words (proper nouns)
        words = query.split()
        named_entities = 0

        for word in words:
            # Skip first word of sentence
            if word != words[0] and word[0].isupper() and len(word) > 1:
                named_entities += 1

        # Look for common entity patterns
        if re.search(r"\d{4}년|\d{1,2}월|\d{1,2}일", query):  # Dates
            named_entities += 1

        if re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", query):  # Person names
            named_entities += 1

        return named_entities

    def get_classification_explanation(
        self,
        query: str,
        query_type: QueryType,
        complexity: QueryComplexity,
        knowledge_req: KnowledgeRequirement,
        confidence: float,
    ) -> str:
        """Generate human-readable explanation for the classification."""
        features = self._extract_features(query)

        explanation = f"Query Analysis for: '{query[:50]}...'\n\n"
        explanation += "Classification Results:\n"
        explanation += f"- Type: {query_type.value}\n"
        explanation += f"- Complexity: {complexity.value}\n"
        explanation += f"- Knowledge Requirement: {knowledge_req.value}\n"
        explanation += f"- Confidence: {confidence:.2f}\n\n"

        explanation += "Key Features Detected:\n"
        explanation += f"- Word count: {features.word_count}\n"
        explanation += f"- Has question words: {len(features.question_words) > 0}\n"
        explanation += f"- Technical terms: {features.has_technical_terms}\n"
        explanation += f"- Procedural indicators: {features.is_procedural}\n"
        explanation += f"- Creative indicators: {features.is_creative}\n"

        return explanation
