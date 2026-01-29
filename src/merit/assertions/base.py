from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from merit.context import ASSERTION_RESULTS_COLLECTOR, METRIC_CONTEXT, TEST_CONTEXT


if TYPE_CHECKING:
    from merit.predicates.base import PredicateResult


@dataclass
class AssertionRepr:
    """Represents a human-readable representation of an assertion expression.

    This class captures the assertion expression along with surrounding context
    lines and resolved argument values for debugging and reporting purposes.

    Attributes:
    ----------
    expr : str
        The assertion expression as a string.
    lines_above : str
        Source code lines appearing above the assertion expression.
    lines_below : str
        Source code lines appearing below the assertion expression.
    resolved_args : dict[str, str]
        A dictionary mapping argument names to their string representations,
        showing the resolved values of variables used in the assertion.
    """

    expr: str
    lines_above: str
    lines_below: str
    resolved_args: dict[str, str]


@dataclass
class AssertionResult:
    """Represents the result of an assertion evaluation in a merit test.

    This class captures the outcome of an assertion, including whether it passed,
    the original expression, any error messages, and results from predicate evaluations.

    Attributes:
    ----------
    expression_repr : AssertionRepr
        A human-readable representation of the assertion expression, including
        the expression string and resolved argument values.
    passed : bool
        Whether the assertion passed (True) or failed (False).
    error_message : str or None, optional
        An error message describing why the assertion failed, if applicable.
    predicate_results : list[PredicateResult], optional
        Results from any predicate evaluations that were part of this assertion.
    """

    expression_repr: AssertionRepr
    passed: bool
    error_message: str | None = None
    predicate_results: list[PredicateResult] = field(default_factory=list)

    @property
    def pretty(self) -> str:
        exp = self.expression_repr
        parts = ["Assertion Passed!" if self.passed else "Assertion Failed!"]
        if self.error_message:
            parts.append(f"Error message: {self.error_message}")
        
        above = textwrap.dedent(exp.lines_above).splitlines() if exp.lines_above else []
        expr = exp.expr.splitlines() if exp.expr else []
        below = textwrap.dedent(exp.lines_below).splitlines() if exp.lines_below else []
        
        parts.extend(f"│  {line}" for line in above)
        if expr:
            parts.append(f">  {expr[0]}")
            parts.extend(f">  {line}" for line in expr[1:])
        parts.extend(f"│  {line}" for line in below)
        
        if exp.resolved_args:
            parts.append("╰─ where:")
            for name, value in exp.resolved_args.items():
                name_oneline = " ".join(name.split())
                name_oneline = re.sub(r"\s+([)\]}.,])", r"\1", name_oneline)
                name_oneline = re.sub(r"([(\[{])\s+", r"\1", name_oneline)
                name_oneline = re.sub(r"\s*\.\s*", ".", name_oneline)
                value_oneline = " ".join(value.split())
                parts.append(f"     {name_oneline} = {value_oneline}")
        
        return "\n".join(parts)

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
                msg = self.error_message or f"Assertion failed: {self.expression_repr.expr}"
                raise AssertionError(msg)


def capture_var(values: dict[str, str], name: str, value: object) -> object:
    """Capture a variable's value and store it in a dictionary."""
    values[name] = repr(value)
    return value
