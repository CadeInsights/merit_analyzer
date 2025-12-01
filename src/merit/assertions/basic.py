"""Basic assertion implementations."""

from typing import Any

from merit.assertions._base import Assertion, AssertionResult
from merit.testing import Case


class ExactMatch(Assertion):
    """Assertion that checks for exact match between actual and expected output."""

    name = "ExactMatch"

    def __init__(self, expected: Any):
        self.expected = expected

    def __call__(self, actual: Any, case: Case) -> AssertionResult:
        """Check if actual output exactly matches expected."""
        if isinstance(actual, str) and isinstance(self.expected, str):
            passed = actual.strip() == self.expected.strip()
        else:
            passed = actual == self.expected

        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            confidence=1.0,
            message=None if passed else f"Expected: {self.expected!r}, Got: {actual!r}",
        )
