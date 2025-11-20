"""Merit - Testing framework for AI agents."""

from .assertions import Assertion, AssertionResult, ExactMatch
from .metrics import AverageScore, Metric, PassRate
from .testing import Case, CaseSet, Suite


__version__ = "0.1.0"

__all__ = [
    # Core testing
    "Case",
    "CaseSet",
    "Suite",
    # Assertions
    "Assertion",
    "AssertionResult",
    "ExactMatch",
    # Metrics
    "Metric",
    "PassRate",
    "AverageScore",
]
