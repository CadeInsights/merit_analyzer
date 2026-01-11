"""Merit - Testing framework for AI agents."""

from .context import metrics
from .metrics import Metric, metric
from .predicates import Predicate, PredicateMetadata, PredicateResult, predicate
from .testing import Case, iter_cases, parametrize, repeat, resource, tag, valididate_cases_for_sut
from .testing.sut import sut
from .tracing import init_tracing, trace_step, TraceContext
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
    "metrics",
    # Tracing
    "init_tracing",
    "trace_step",
    "TraceContext",
]