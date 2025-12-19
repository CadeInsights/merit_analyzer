"""Predicates library for AI-focused assertions."""

from .client import get_predicate_api_client, close_predicate_api_client
from .base import Predicate, PredicateResult, PredicateMetadata, predicate

from .condition_predicates import satisfies
from .fact_predicates import contradicts, supported, contains, matches
from .style_predicates import layout_matches, syntax_matches, tone_matches, vocabulary_matches

__all__ = [
    # Predicate abstractions
    "Predicate",
    "PredicateResult",
    "PredicateMetadata",
    "predicate",
    # Client for remote checks
    "get_predicate_api_client",
    "close_predicate_api_client",
    # Condition predicates
    "satisfies",
    # Fact predicates
    "contradicts",
    "supported",
    "contains",
    "matches",  
    # Style predicates
    "layout_matches",
    "syntax_matches",
    "tone_matches",
    "vocabulary_matches",
]
