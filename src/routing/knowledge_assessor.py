"""Knowledge requirement assessment for intelligent routing decisions."""

import re
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel

from ..models.routing_schemas import (
    DecisionContext,
    KnowledgeRequirement,
    QueryComplexity,
    QueryType,
)
from ..utils.logger import logger


class KnowledgeCategory(str, Enum):
    """Categories of knowledge that might be needed."""

    FACTUAL = "factual"  # Facts, data, statistics
    PROCEDURAL = "procedural"  # How-to, processes, methods
    CONCEPTUAL = "conceptual"  # Abstract concepts, theories
    DOMAIN_SPECIFIC = "domain_specific"  # Specialized domain knowledge
    TEMPORAL = "temporal"  # Time-sensitive information
    COMPARATIVE = "comparative"  # Comparison between entities
    CAUSAL = "causal"  # Cause-effect relationships
    PERSONAL = "personal"  # Personal preferences/opinions


class KnowledgeSource(str, Enum):
    """Potential sources of knowledge."""

    LLM_INTERNAL = "llm_internal"  # LLM's training data
    VECTOR_DB = "vector_db"  # Vector database
    STRUCTURED_DB = "structured_db"  # SQL/NoSQL databases
    REALTIME_API = "realtime_api"  # Real-time APIs
    SPECIALIZED_TOOL = "specialized_tool"  # Domain-specific tools
    WEB_SEARCH = "web_search"  # Web search engines


class KnowledgeAssessment(BaseModel):
    """Assessment of knowledge requirements for a query."""

    requirement_level: KnowledgeRequirement
    confidence: float
    knowledge_categories: List[KnowledgeCategory]
    recommended_sources: List[KnowledgeSource]
    knowledge_gaps: List[str]
    assessment_reasoning: str
    temporal_sensitivity: float  # 0.0 = timeless, 1.0 = very time-sensitive
    domain_specificity: float  # 0.0 = general, 1.0 = highly specialized
    factual_density: float  # 0.0 = opinion-based, 1.0 = fact-heavy


class KnowledgeAssessor:
    """Assess knowledge requirements for optimal routing decisions."""

    def __init__(self):
        """Initialize the knowledge assessor."""

        # Knowledge indicators
        self._factual_indicators = [
            "무엇",
            "언제",
            "어디",
            "누구",
            "얼마",
            "몇",
            "what",
            "when",
            "where",
            "who",
            "how much",
            "how many",
            "정의",
            "의미",
            "definition",
            "meaning",
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
            "procedure",
            "method",
        ]

        self._temporal_indicators = [
            "최신",
            "현재",
            "지금",
            "오늘",
            "2024",
            "2025",
            "latest",
            "current",
            "now",
            "today",
            "recent",
            "업데이트",
            "변화",
            "트렌드",
        ]

        self._comparison_indicators = [
            "vs",
            "비교",
            "차이",
            "대신",
            "보다",
            "compare",
            "difference",
            "versus",
            "than",
            "장단점",
            "pros and cons",
        ]

        # Domain-specific patterns
        self._domain_patterns = {
            "medical": [r"증상|질병|치료|의학|병원|진단", r"symptom|disease|treatment|medical"],
            "legal": [r"법|계약|소송|판결|조항", r"law|legal|contract|lawsuit|court"],
            "technical": [r"알고리즘|코드|프로그래밍|API", r"algorithm|code|programming|software"],
            "financial": [r"투자|주식|금융|대출", r"investment|stock|finance|trading"],
            "academic": [r"연구|논문|이론|실험", r"research|paper|theory|study"],
        }

        # LLM knowledge cutoffs and limitations
        self._llm_limitations = {
            "temporal_cutoff": "2024-04",  # Approximate knowledge cutoff
            "weak_domains": ["real_time_data", "personal_data", "recent_events"],
            "strong_domains": ["general_knowledge", "coding", "analysis", "writing"],
        }

        logger.info("KnowledgeAssessor initialized")

    def assess_knowledge_requirement(
        self,
        query: str,
        query_type: QueryType,
        complexity: QueryComplexity,
        context: Optional[DecisionContext] = None,
    ) -> KnowledgeAssessment:
        """Assess the knowledge requirements for a given query.

        Args:
            query: User query text
            query_type: Classified query type
            complexity: Query complexity level
            context: Additional context

        Returns:
            Comprehensive knowledge assessment
        """
        try:
            query_lower = query.lower()

            # Analyze knowledge categories needed
            knowledge_categories = self._identify_knowledge_categories(query, query_type)

            # Assess temporal sensitivity
            temporal_sensitivity = self._assess_temporal_sensitivity(query, query_lower)

            # Assess domain specificity
            domain_specificity = self._assess_domain_specificity(query, query_lower, context)

            # Assess factual density
            factual_density = self._assess_factual_density(query, query_type, query_lower)

            # Determine requirement level
            requirement_level = self._determine_requirement_level(
                query_type,
                complexity,
                knowledge_categories,
                temporal_sensitivity,
                domain_specificity,
                factual_density,
            )

            # Recommend knowledge sources
            recommended_sources = self._recommend_knowledge_sources(
                knowledge_categories, requirement_level, temporal_sensitivity, domain_specificity
            )

            # Identify potential knowledge gaps
            knowledge_gaps = self._identify_knowledge_gaps(
                query, query_type, knowledge_categories, context
            )

            # Calculate confidence
            confidence = self._calculate_assessment_confidence(
                query_type, complexity, knowledge_categories, temporal_sensitivity
            )

            # Generate reasoning
            reasoning = self._generate_assessment_reasoning(
                requirement_level,
                knowledge_categories,
                temporal_sensitivity,
                domain_specificity,
                recommended_sources,
            )

            assessment = KnowledgeAssessment(
                requirement_level=requirement_level,
                confidence=confidence,
                knowledge_categories=knowledge_categories,
                recommended_sources=recommended_sources,
                knowledge_gaps=knowledge_gaps,
                assessment_reasoning=reasoning,
                temporal_sensitivity=temporal_sensitivity,
                domain_specificity=domain_specificity,
                factual_density=factual_density,
            )

            logger.info(
                f"Knowledge assessment: {requirement_level.value} "
                f"(confidence: {confidence:.2f}, categories: {len(knowledge_categories)})"
            )

            return assessment

        except Exception as e:
            logger.error(f"Error in knowledge assessment: {e}")
            return self._create_fallback_assessment()

    def _identify_knowledge_categories(
        self, query: str, query_type: QueryType
    ) -> List[KnowledgeCategory]:
        """Identify the categories of knowledge needed for the query."""
        categories = []
        query_lower = query.lower()

        # Check for factual knowledge needs
        if any(indicator in query_lower for indicator in self._factual_indicators):
            categories.append(KnowledgeCategory.FACTUAL)

        # Check for procedural knowledge needs
        if any(indicator in query_lower for indicator in self._procedural_indicators):
            categories.append(KnowledgeCategory.PROCEDURAL)

        # Check for temporal knowledge needs
        if any(indicator in query_lower for indicator in self._temporal_indicators):
            categories.append(KnowledgeCategory.TEMPORAL)

        # Check for comparative knowledge needs
        if any(indicator in query_lower for indicator in self._comparison_indicators):
            categories.append(KnowledgeCategory.COMPARATIVE)

        # Domain-specific check
        for domain, patterns in self._domain_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                categories.append(KnowledgeCategory.DOMAIN_SPECIFIC)
                break

        # Query type based categorization
        if query_type == QueryType.ANALYTICAL:
            categories.append(KnowledgeCategory.CONCEPTUAL)
        elif query_type == QueryType.CREATIVE:
            categories.append(KnowledgeCategory.PERSONAL)
        elif query_type == QueryType.TECHNICAL:
            categories.append(KnowledgeCategory.DOMAIN_SPECIFIC)

        # Causal relationship detection
        causal_patterns = [
            r"왜|이유|원인|because|why|cause|reason",
            r"결과|영향|effect|impact|result",
        ]
        if any(re.search(pattern, query_lower) for pattern in causal_patterns):
            categories.append(KnowledgeCategory.CAUSAL)

        return categories or [KnowledgeCategory.FACTUAL]  # Default to factual if none detected

    def _assess_temporal_sensitivity(self, query: str, query_lower: str) -> float:
        """Assess how time-sensitive the query is (0.0 = timeless, 1.0 = very time-sensitive)."""
        temporal_score = 0.0

        # Strong temporal indicators
        strong_temporal = [
            "최신",
            "현재",
            "지금",
            "오늘",
            "이번",
            "요즘",
            "latest",
            "current",
            "now",
            "today",
            "recent",
            "nowadays",
        ]
        if any(term in query_lower for term in strong_temporal):
            temporal_score += 0.8

        # Specific years (especially recent)
        year_matches = re.findall(r"(20[0-9]{2})", query)
        for year in year_matches:
            year_int = int(year)
            if year_int >= 2023:  # Very recent
                temporal_score += 0.6
            elif year_int >= 2020:  # Recent
                temporal_score += 0.4
            else:  # Historical
                temporal_score += 0.1

        # Trend and change indicators
        change_indicators = [
            "트렌드",
            "변화",
            "업데이트",
            "발전",
            "trend",
            "change",
            "update",
            "development",
        ]
        if any(term in query_lower for term in change_indicators):
            temporal_score += 0.5

        # Real-time data indicators
        realtime_indicators = ["실시간", "라이브", "현재", "real-time", "live", "streaming"]
        if any(term in query_lower for term in realtime_indicators):
            temporal_score += 0.9

        return min(1.0, temporal_score)

    def _assess_domain_specificity(
        self, query: str, query_lower: str, context: Optional[DecisionContext]
    ) -> float:
        """Assess how domain-specific the query is (0.0 = general, 1.0 = highly specialized)."""
        specificity_score = 0.0

        # Check for domain-specific patterns
        domain_matches = 0
        for domain, patterns in self._domain_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                domain_matches += 1
                specificity_score += 0.3

        # Technical jargon detection
        technical_patterns = [
            r"[A-Z]{2,}",  # Acronyms
            r"[a-zA-Z]+\.[a-zA-Z]+",  # Domain names or technical notation
            r"v?\d+\.\d+",  # Version numbers
        ]
        for pattern in technical_patterns:
            if re.search(pattern, query):
                specificity_score += 0.2

        # Context-based domain specificity
        if context and context.domain_context:
            if context.domain_context != "general":
                specificity_score += 0.4

        # Specialized terminology indicators
        specialized_terms = [
            "전문",
            "기술적",
            "학술",
            "연구",
            "분석",
            "professional",
            "technical",
            "academic",
            "research",
            "analysis",
        ]
        if any(term in query_lower for term in specialized_terms):
            specificity_score += 0.3

        return min(1.0, specificity_score)

    def _assess_factual_density(self, query: str, query_type: QueryType, query_lower: str) -> float:
        """Assess how fact-heavy vs opinion-based the query is (0.0 = opinion, 1.0 = fact-heavy)."""
        factual_score = 0.0

        # Strong factual indicators
        factual_terms = [
            "사실",
            "데이터",
            "통계",
            "수치",
            "정확",
            "fact",
            "data",
            "statistics",
            "number",
            "accurate",
            "precise",
        ]
        if any(term in query_lower for term in factual_terms):
            factual_score += 0.6

        # Question words that typically require facts
        factual_questions = ["무엇", "언제", "어디", "얼마", "what", "when", "where", "how much"]
        if any(word in query_lower for word in factual_questions):
            factual_score += 0.5

        # Opinion/subjective indicators (reduce factual score)
        opinion_terms = [
            "생각",
            "의견",
            "추천",
            "좋아",
            "싫어",
            "선호",
            "think",
            "opinion",
            "recommend",
            "like",
            "prefer",
            "feel",
        ]
        if any(term in query_lower for term in opinion_terms):
            factual_score -= 0.4

        # Query type adjustments
        if query_type in [QueryType.FACTUAL_SIMPLE, QueryType.FACTUAL_COMPLEX]:
            factual_score += 0.4
        elif query_type in [QueryType.CREATIVE, QueryType.CONVERSATIONAL]:
            factual_score -= 0.3
        elif query_type == QueryType.ANALYTICAL:
            factual_score += 0.2  # Analysis often needs facts but also interpretation

        return max(0.0, min(1.0, factual_score))

    def _determine_requirement_level(
        self,
        query_type: QueryType,
        complexity: QueryComplexity,
        knowledge_categories: List[KnowledgeCategory],
        temporal_sensitivity: float,
        domain_specificity: float,
        factual_density: float,
    ) -> KnowledgeRequirement:
        """Determine the overall knowledge requirement level."""

        # Base requirement from query type
        base_requirements = {
            QueryType.GREETING: KnowledgeRequirement.NONE,
            QueryType.CONVERSATIONAL: KnowledgeRequirement.NONE,
            QueryType.CREATIVE: KnowledgeRequirement.MINIMAL,
            QueryType.FACTUAL_SIMPLE: KnowledgeRequirement.MINIMAL,
            QueryType.FACTUAL_COMPLEX: KnowledgeRequirement.MODERATE,
            QueryType.PROCEDURAL: KnowledgeRequirement.MODERATE,
            QueryType.ANALYTICAL: KnowledgeRequirement.EXTENSIVE,
            QueryType.TECHNICAL: KnowledgeRequirement.EXTENSIVE,
            QueryType.DOMAIN_SPECIFIC: KnowledgeRequirement.SPECIALIZED,
            QueryType.MULTIMODAL: KnowledgeRequirement.EXTENSIVE,
        }

        base_req = base_requirements.get(query_type, KnowledgeRequirement.MODERATE)

        # Calculate enhancement score
        enhancement_score = 0

        # Complexity adjustments
        complexity_scores = {
            QueryComplexity.TRIVIAL: -2,
            QueryComplexity.SIMPLE: -1,
            QueryComplexity.MODERATE: 0,
            QueryComplexity.COMPLEX: 1,
            QueryComplexity.EXPERT: 2,
        }
        enhancement_score += complexity_scores[complexity]

        # Temporal sensitivity (time-sensitive queries often need external data)
        if temporal_sensitivity > 0.7:
            enhancement_score += 2
        elif temporal_sensitivity > 0.4:
            enhancement_score += 1

        # Domain specificity
        if domain_specificity > 0.8:
            enhancement_score += 2
        elif domain_specificity > 0.5:
            enhancement_score += 1

        # Knowledge categories impact
        if KnowledgeCategory.DOMAIN_SPECIFIC in knowledge_categories:
            enhancement_score += 1
        if KnowledgeCategory.TEMPORAL in knowledge_categories:
            enhancement_score += 1
        if len(knowledge_categories) > 3:
            enhancement_score += 1

        # Map base requirement to numeric value for adjustment
        requirement_levels = [
            KnowledgeRequirement.NONE,
            KnowledgeRequirement.MINIMAL,
            KnowledgeRequirement.MODERATE,
            KnowledgeRequirement.EXTENSIVE,
            KnowledgeRequirement.SPECIALIZED,
        ]

        base_index = requirement_levels.index(base_req)
        adjusted_index = max(0, min(len(requirement_levels) - 1, base_index + enhancement_score))

        return requirement_levels[adjusted_index]

    def _recommend_knowledge_sources(
        self,
        knowledge_categories: List[KnowledgeCategory],
        requirement_level: KnowledgeRequirement,
        temporal_sensitivity: float,
        domain_specificity: float,
    ) -> List[KnowledgeSource]:
        """Recommend appropriate knowledge sources based on assessment."""
        sources = []

        # Base recommendations by requirement level
        if requirement_level == KnowledgeRequirement.NONE:
            sources.append(KnowledgeSource.LLM_INTERNAL)

        elif requirement_level == KnowledgeRequirement.MINIMAL:
            sources.extend([KnowledgeSource.LLM_INTERNAL, KnowledgeSource.VECTOR_DB])

        elif requirement_level == KnowledgeRequirement.MODERATE:
            sources.extend([KnowledgeSource.VECTOR_DB, KnowledgeSource.LLM_INTERNAL])

        elif requirement_level == KnowledgeRequirement.EXTENSIVE:
            sources.extend(
                [
                    KnowledgeSource.VECTOR_DB,
                    KnowledgeSource.STRUCTURED_DB,
                    KnowledgeSource.LLM_INTERNAL,
                ]
            )

        else:  # SPECIALIZED
            sources.extend(
                [
                    KnowledgeSource.SPECIALIZED_TOOL,
                    KnowledgeSource.VECTOR_DB,
                    KnowledgeSource.STRUCTURED_DB,
                ]
            )

        # Temporal sensitivity adjustments
        if temporal_sensitivity > 0.7:
            sources.insert(0, KnowledgeSource.REALTIME_API)
            sources.append(KnowledgeSource.WEB_SEARCH)

        # Domain specificity adjustments
        if domain_specificity > 0.8:
            if KnowledgeSource.SPECIALIZED_TOOL not in sources:
                sources.insert(0, KnowledgeSource.SPECIALIZED_TOOL)

        # Category-specific recommendations
        if KnowledgeCategory.TEMPORAL in knowledge_categories:
            if KnowledgeSource.REALTIME_API not in sources:
                sources.append(KnowledgeSource.REALTIME_API)

        if KnowledgeCategory.COMPARATIVE in knowledge_categories:
            if KnowledgeSource.STRUCTURED_DB not in sources:
                sources.append(KnowledgeSource.STRUCTURED_DB)

        return sources[:3]  # Limit to top 3 recommendations

    def _identify_knowledge_gaps(
        self,
        query: str,
        query_type: QueryType,
        knowledge_categories: List[KnowledgeCategory],
        context: Optional[DecisionContext],
    ) -> List[str]:
        """Identify potential knowledge gaps that could affect answer quality."""
        gaps = []

        # Temporal gaps
        if KnowledgeCategory.TEMPORAL in knowledge_categories:
            gaps.append("May lack most recent information beyond training cutoff")

        # Domain-specific gaps
        if KnowledgeCategory.DOMAIN_SPECIFIC in knowledge_categories:
            gaps.append("May need specialized domain knowledge not in general training")

        # Personal/contextual gaps
        if context and context.user_id:
            gaps.append("Lacks personalized user preferences and history")

        # Real-time data gaps
        realtime_indicators = ["실시간", "현재", "지금", "real-time", "current", "now"]
        if any(indicator in query.lower() for indicator in realtime_indicators):
            gaps.append("Cannot access real-time data or current events")

        # Specific entity gaps
        if re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", query):  # Proper nouns
            gaps.append("May lack information about specific entities or recent figures")

        # Procedural gaps for specialized domains
        if (
            query_type == QueryType.PROCEDURAL
            and KnowledgeCategory.DOMAIN_SPECIFIC in knowledge_categories
        ):
            gaps.append("May lack step-by-step procedures for specialized domains")

        return gaps

    def _calculate_assessment_confidence(
        self,
        query_type: QueryType,
        complexity: QueryComplexity,
        knowledge_categories: List[KnowledgeCategory],
        temporal_sensitivity: float,
    ) -> float:
        """Calculate confidence in the knowledge assessment."""
        confidence = 0.8  # Base confidence

        # Clear query types boost confidence
        if query_type in [QueryType.GREETING, QueryType.FACTUAL_SIMPLE, QueryType.PROCEDURAL]:
            confidence += 0.1

        # Complex or ambiguous types reduce confidence
        if query_type in [QueryType.ANALYTICAL, QueryType.MULTIMODAL]:
            confidence -= 0.1

        # High complexity reduces confidence
        if complexity >= QueryComplexity.COMPLEX:
            confidence -= 0.1

        # High temporal sensitivity makes assessment uncertain
        if temporal_sensitivity > 0.8:
            confidence -= 0.15

        # Multiple knowledge categories make assessment more complex
        if len(knowledge_categories) > 3:
            confidence -= 0.05

        return max(0.5, min(0.95, confidence))

    def _generate_assessment_reasoning(
        self,
        requirement_level: KnowledgeRequirement,
        knowledge_categories: List[KnowledgeCategory],
        temporal_sensitivity: float,
        domain_specificity: float,
        recommended_sources: List[KnowledgeSource],
    ) -> str:
        """Generate human-readable reasoning for the assessment."""
        reasoning_parts = []

        # Requirement level explanation
        level_explanations = {
            KnowledgeRequirement.NONE: "LLM의 내재 지식으로 충분",
            KnowledgeRequirement.MINIMAL: "기본 검증 또는 최소한의 외부 지식 필요",
            KnowledgeRequirement.MODERATE: "일반적인 RAG 검색으로 충분한 외부 지식 필요",
            KnowledgeRequirement.EXTENSIVE: "광범위한 검색과 다양한 소스의 지식 필요",
            KnowledgeRequirement.SPECIALIZED: "전문 지식 데이터베이스나 도구 필요",
        }
        reasoning_parts.append(level_explanations[requirement_level])

        # Knowledge categories
        if knowledge_categories:
            categories_str = ", ".join([cat.value for cat in knowledge_categories])
            reasoning_parts.append(f"필요 지식 유형: {categories_str}")

        # Temporal sensitivity
        if temporal_sensitivity > 0.7:
            reasoning_parts.append("시간에 민감한 정보로 실시간 데이터 필요")
        elif temporal_sensitivity > 0.4:
            reasoning_parts.append("일부 시간적 맥락 고려 필요")

        # Domain specificity
        if domain_specificity > 0.8:
            reasoning_parts.append("고도로 전문화된 도메인 지식 필요")
        elif domain_specificity > 0.5:
            reasoning_parts.append("특정 도메인 지식 필요")

        # Recommended sources
        if recommended_sources:
            sources_str = ", ".join([source.value for source in recommended_sources[:2]])
            reasoning_parts.append(f"권장 지식 소스: {sources_str}")

        return " | ".join(reasoning_parts)

    def _create_fallback_assessment(self) -> KnowledgeAssessment:
        """Create fallback assessment when assessment fails."""
        return KnowledgeAssessment(
            requirement_level=KnowledgeRequirement.MODERATE,
            confidence=0.5,
            knowledge_categories=[KnowledgeCategory.FACTUAL],
            recommended_sources=[KnowledgeSource.VECTOR_DB, KnowledgeSource.LLM_INTERNAL],
            knowledge_gaps=["Assessment failed - using safe defaults"],
            assessment_reasoning="Fallback assessment due to processing error",
            temporal_sensitivity=0.3,
            domain_specificity=0.3,
            factual_density=0.5,
        )
