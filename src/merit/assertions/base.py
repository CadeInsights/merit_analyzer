from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from merit.context import METRIC_CONTEXT, TEST_CONTEXT

if TYPE_CHECKING:
    from merit.predicates.base import PredicateResult
    from merit.metrics.base import MetricValue



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
        Optional list of MetricValue objects collected during the assertion.

    Attributes
    ----------
    passed
        Boolean pass/fail state. Setting this property records the boolean value into
        currently attached metrics.
    """

    expression_repr: str
    error_message: str | None = None
    predicate_results: list[PredicateResult] = field(default_factory=list)
    metric_values: set[MetricValue] = field(default_factory=set)

    _passed: bool | None = field(default=None)

    def __post_init__(self) -> None:
        if test_ctx := TEST_CONTEXT.get():
            test_ctx.assertion_results.append(self)

    @property
    def passed(self) -> bool:
        if self._passed is None:
            raise ValueError("AssertionResult.passed is not set")
        return self._passed

    @passed.setter
    def passed(self, passed: bool) -> None:
        if metrics := METRIC_CONTEXT.get():
            for metric in metrics:
                metric.add_record(passed)

        if test_ctx := TEST_CONTEXT.get():
            if test_ctx.item.fail_fast and not passed:
                msg = self.error_message or f"Assertion failed: {self.expression_repr}"
                raise AssertionError(msg)

        self._passed = passed
