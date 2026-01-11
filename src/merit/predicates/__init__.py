"""Predicates library for AI-focused assertions."""

from .ai_predicates import (
    follows_policy,
    has_conflicting_facts,
    has_facts,
    has_topics,
    has_unsupported_facts,
    matches_facts,
    matches_writing_layout,
    matches_writing_style,
)
from .base import Predicate, PredicateMetadata, PredicateResult, predicate
from .client import (
    close_predicate_api_client,
    create_predicate_api_client,
    get_predicate_api_client,
)


__all__ = [
    # Predicate abstractions
    "Predicate",
    "PredicateResult",
    "PredicateMetadata",
    "predicate",
    # Client for remote checks
    "create_predicate_api_client",
    "get_predicate_api_client",
    "close_predicate_api_client",
    # Condition predicates
    "follows_policy",
    "has_conflicting_facts",
    "has_unsupported_facts",
    "has_facts",
    "has_topics",
    "matches_facts",
    "matches_writing_layout",
    "matches_writing_style",
]
