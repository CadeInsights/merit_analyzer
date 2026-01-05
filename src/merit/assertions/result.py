from dataclasses import dataclass, field

from merit.predicates.base import PredicateResult
from merit.metrics.result import MetricResult

@dataclass
class AssertionResult:
    """Result of an assertion."""

    expression: str
    passed: bool
    error_message: str | None = None

    captured_metric_results: list[MetricResult] = field(default_factory=list)
    captured_predicate_results: list[PredicateResult] = field(default_factory=list)