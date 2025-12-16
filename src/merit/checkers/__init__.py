"""Checkers library for AI-focused assertions."""

from .client import get_checker_api_client, close_checker_api_client
from .base import Checker, CheckerResult, CheckerMetadata

from .condition_checkers import satisfies
from .fact_checkers import contradicts, supported, contains, matches
from .style_checkers import layout_matches, syntax_matches, tone_matches, vocabulary_matches

__all__ = [
    # Checker abstractions
    "Checker",
    "CheckerResult",
    "CheckerMetadata",
    # Client for remote checks
    "get_checker_api_client",
    "close_checker_api_client",
    # Condition checkers
    "satisfies",
    # Fact checkers
    "contradicts",
    "supported",
    "contains",
    "matches",  
    # Style checkers
    "layout_matches",
    "syntax_matches",
    "tone_matches",
    "vocabulary_matches",
]
