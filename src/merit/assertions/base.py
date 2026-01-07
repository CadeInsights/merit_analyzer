from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from merit.predicates.base import PredicateResult


@dataclass
class AssertionResult:
    """Result of an assertion."""

    expression_repr: str
    passed: bool
    error_message: str | None = None
    predicate_results: list[PredicateResult] = field(default_factory=list)
