"""Basic assertion implementations."""

from typing import Any

from merit.assertions._base import Assertion, AssertionResult
from merit.testing import Case


class ExactMatch(Assertion):
    """Assertion that checks for exact match between actual and expected output."""

    name = "ExactMatch"

    def __call__(self, actual: Any, case: Case) -> AssertionResult:
        """Check if actual output exactly matches expected."""
        # Handle string comparison with strip
        if isinstance(actual, str) and isinstance(case.expected_output, str):
            passed = actual.strip() == case.expected_output.strip()
        else:
            passed = actual == case.expected_output

        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            confidence=1.0,
            message=None if passed else f"Expected: {case.expected_output!r}, Got: {actual!r}",
        )
