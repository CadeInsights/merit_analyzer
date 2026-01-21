"""Merit - Testing framework for AI agents."""

from .context import metrics
from .metrics_ import Metric, metric
from .predicates import Predicate, PredicateMetadata, PredicateResult, predicate
from .testing import (
    Case,
    fail,
    iter_cases,
    parametrize,
    repeat,
    resource,
    skip,
    tag,
    validate_cases_for_sut,
    xfail,
)
from .testing.sut import sut
from .tracing import TraceContext, init_tracing, trace_step


__all__ = [
    # Core testing
    "Case",
    "iter_cases",
    "validate_cases_for_sut",
    "parametrize",
    "repeat",
    "tag",
    "resource",
    "skip",
    "fail",
    "xfail",
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
