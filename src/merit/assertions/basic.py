"""Basic assertion implementations."""

from typing import Any

from merit.assertions._base import Assertion, AssertionResult, _truncate


class ExactMatch(Assertion):
    """Assertion that checks for exact match between actual and expected output."""

    def __init__(self, expected: Any):
        self.expected = expected
        self.name = f"ExactMatch({_truncate(expected)})"

    def evaluate(self, actual: Any) -> AssertionResult:
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


class Contains(Assertion):
    """Assertion that checks if actual output contains expected substring."""

    def __init__(self, substring: str):
        self.substring = substring
        self.name = f"Contains({_truncate(substring)})"

    def evaluate(self, actual: Any) -> AssertionResult:
        """Check if actual output contains the substring."""
        actual_str = str(actual)
        passed = self.substring in actual_str

        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            confidence=1.0,
            message=None if passed else f"Expected to contain: {self.substring!r}, Got: {actual_str!r}",
        )


class StartsWith(Assertion):
    """Assertion that checks if actual output starts with expected prefix."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.name = f"StartsWith({_truncate(prefix)})"

    def evaluate(self, actual: Any) -> AssertionResult:
        """Check if actual output starts with the prefix."""
        actual_str = str(actual)
        passed = actual_str.startswith(self.prefix)

        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            confidence=1.0,
            message=None if passed else f"Expected to start with: {self.prefix!r}",
        )
