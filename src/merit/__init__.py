"""Merit - Testing framework for AI agents."""

from .predicates import predicate, PredicateResult, PredicateMetadata, Predicate
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
    # Predicates
    "predicate",
    "PredicateResult",
    "PredicateMetadata",
    "Predicate",
    # Metrics
    "Metric",
    "PassRate",
    "AverageScore",
    # Tracing
    "init_tracing",
    "trace_step",
]
