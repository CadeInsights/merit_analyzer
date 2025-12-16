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
from .testing.sut import sut
from .tracing import init_tracing, trace_step
from .version import __version__


__all__ = [
    # Core testing
    "Case",
    "Suite",
    "parametrize",
    "tag",
    "resource",
    "sut",
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
    # Tracing
    "init_tracing",
    "trace_step",
]
