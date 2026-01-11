from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from merit.context import ASSERTION_RESULTS_COLLECTOR, METRIC_CONTEXT, TEST_CONTEXT


if TYPE_CHECKING:
    from merit.metrics.base import MetricSnapshot
    from merit.predicates.base import PredicateResult


@dataclass
class AssertionResult:
    """Result of evaluating a single assertion.

    This dataclass stores the outcome of an assertion evaluation along with
    optional rich debugging/analysis artifacts (predicate results and metric
    values).

    Parameters
    ----------
    expression_repr
        Human-readable representation of the asserted expression (e.g.,
        ``"x == y"``) for reporting/debugging.
    error_message
        Optional error/details string explaining a failure (or exception) for
        reporting.
    predicate_results
        Optional list of PredicateResult objects collected during the assertion.
    metric_values
        Optional list of MetricSnapshot objects collected during the assertion.

    Attributes:
    ----------
    passed
        Boolean pass/fail state. Setting this property records the boolean value into
        currently attached metrics.
    """

    expression_repr: str
    passed: bool
    error_message: str | None = None
    predicate_results: list[PredicateResult] = field(default_factory=list)
    metric_values: set[MetricSnapshot] = field(default_factory=set)

    def __post_init__(self) -> None:
        collector = ASSERTION_RESULTS_COLLECTOR.get()
        if collector is not None:
            collector.append(self)

        metrics = METRIC_CONTEXT.get()
        if metrics is not None:
            for metric in metrics:
                metric.add_record(self.passed)

        test_ctx = TEST_CONTEXT.get()
        if test_ctx is not None:
            if test_ctx.item.fail_fast and not self.passed:
                msg = self.error_message or f"Assertion failed: {self.expression_repr}"
                raise AssertionError(msg)
