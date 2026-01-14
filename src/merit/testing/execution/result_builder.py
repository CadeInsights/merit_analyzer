"""Test result building logic."""

from __future__ import annotations

from dataclasses import dataclass

from merit.assertions.base import AssertionResult
from merit.testing.models.definition import MeritTestDefinition
from merit.testing.models.result import TestResult, TestStatus


@dataclass
class ResultBuilder:
    """Builds TestResult from test execution data."""

    def build(
        self,
        definition: MeritTestDefinition,
        duration_ms: float,
        assertion_results: list[AssertionResult],
        error: Exception | None,
    ) -> TestResult:
        """Create TestResult based on assertion results and raised exceptions."""
        status, error = self._determine_status(definition, assertion_results, error)
        return TestResult(
            status=status,
            duration_ms=duration_ms,
            error=error,
            assertion_results=assertion_results.copy(),
        )

    def _determine_status(
        self,
        definition: MeritTestDefinition,
        assertion_results: list[AssertionResult],
        error: Exception | None,
    ) -> tuple[TestStatus, Exception | None]:
        """Determine test status and normalize error."""
        expect_failure = definition.xfail_reason is not None
        failed_assertions = [ar for ar in assertion_results if not ar.passed]
        has_assertion_failure = len(failed_assertions) > 0

        if error is not None and not isinstance(error, AssertionError):
            return (TestStatus.XFAILED if expect_failure else TestStatus.ERROR, error)

        if has_assertion_failure or isinstance(error, AssertionError):
            status = TestStatus.XFAILED if expect_failure else TestStatus.FAILED
            error = self._normalize_error(error, failed_assertions)
            return (status, error)

        if expect_failure:
            if definition.xfail_strict:
                return (
                    TestStatus.FAILED,
                    AssertionError(definition.xfail_reason or "xfail test passed"),
                )
            return (TestStatus.XPASSED, None)

        return (TestStatus.PASSED, None)

    def _normalize_error(
        self,
        error: Exception | None,
        failed_assertions: list[AssertionResult],
    ) -> Exception | None:
        """Convert failed assertion to error if no error exists."""
        if error is None and failed_assertions:
            msg = failed_assertions[0].error_message or failed_assertions[0].expression_repr
            return AssertionError(msg)
        return error
