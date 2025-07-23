"""Intelligent routing system for MCP-RAG-Control.

This module implements LLM-based dynamic routing that can achieve 90% cost savings
by intelligently deciding when to use RAG vs direct LLM responses.

Components:
- QueryClassifier: Analyzes query complexity and knowledge requirements
- RouteController: Makes routing decisions based on classification
- DecisionEngine: LLM-powered decision making for optimal route selection
- KnowledgeAssessor: Evaluates whether external knowledge is needed
- CostOptimizer: Optimizes cost-performance tradeoffs
"""

# Import types from models
from ..models.routing_schemas import (
    DecisionContext,
    KnowledgeRequirement,
    QueryComplexity,
    QueryType,
    RoutingDecision,
    RoutingPath,
)

from .cost_optimizer import CostOptimizer
from .decision_engine import DecisionEngine
from .knowledge_assessor import KnowledgeAssessor
from .query_classifier import QueryClassifier
from .route_controller import RouteController

__all__ = [
    "QueryClassifier",
    "QueryType",
    "QueryComplexity",
    "RouteController",
    "RoutingDecision",
    "RoutingPath",
    "DecisionEngine",
    "DecisionContext",
    "KnowledgeAssessor",
    "KnowledgeRequirement",
    "CostOptimizer",
]
