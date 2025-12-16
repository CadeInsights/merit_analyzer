"""Merit - Testing framework for AI agents."""

# from .assertions import (
#     Assertion,
#     AssertionFailedError,
#     AssertionResult,
#     Contains,
#     ExactMatch,
#     StartsWith,
# )
from .metrics import AverageScore, Metric, PassRate
from .testing import Case, Suite, parametrize, resource, tag
from .version import __version__


__all__ = [
    # Core testing
    "Case",
    "Suite",
    "parametrize",
    "tag",
    "resource",
    # Assertions
    "Assertion",
    "AssertionFailedError",
    "AssertionResult",
    "Contains",
    "ExactMatch",
    "StartsWith",
    # Metrics
    "Metric",
    "PassRate",
    "AverageScore",
]
