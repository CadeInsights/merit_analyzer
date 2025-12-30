"""Merit - Testing framework for AI agents."""

from .predicates import predicate, PredicateResult, PredicateMetadata, Predicate
from .metrics import Metric, metric
from .testing import Case, parametrize, repeat, resource, tag, iter_cases, valididate_cases_for_sut
from .testing.sut import sut
from .tracing import init_tracing, trace_step
from .version import __version__


__all__ = [
    # Core testing
    "Case",
    "iter_cases",
    "valididate_cases_for_sut",
    "parametrize",
    "repeat",
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
    "metric",
    # Tracing
    "init_tracing",
    "trace_step",
]
