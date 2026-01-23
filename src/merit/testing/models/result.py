"""Test result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4

from merit.assertions.base import AssertionResult
from merit.testing.models.definition import MeritTestDefinition


class TestStatus(Enum):
    """Test execution status."""

    __test__ = False  # Prevent pytest from collecting this as a test class

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    XFAILED = "xfailed"
    XPASSED = "xpassed"

    @property
    def is_failure(self) -> bool:
        """Check if this status represents a failure."""
        return self in {TestStatus.FAILED, TestStatus.ERROR}


@dataclass
class TestResult:
    """Result of a single test execution."""

    __test__ = False  # Prevent pytest from collecting this as a test class

    status: TestStatus
    duration_ms: float
    error: Exception | None = None
    assertion_results: list[AssertionResult] = field(default_factory=list)


@dataclass
class TestExecution:
    """Complete record of a test execution, combining context and result.

    Encapsulates both the test context (inputs/setup) and the result (outcome)
    as a single execution record.
    """

    __test__ = False  # Prevent pytest from collecting this as a test class

    definition: MeritTestDefinition
    result: TestResult
    execution_id: UUID = field(default_factory=uuid4)
    trace_id: str | None = None
    sub_executions: list[TestExecution] = field(default_factory=list)

    @property
    def item(self) -> MeritTestDefinition:
        """The test item that was executed."""
        return self.definition

    @property
    def status(self) -> TestStatus:
        """Convenience access to result status."""
        return self.result.status

    @property
    def duration_ms(self) -> float:
        """Convenience access to result duration."""
        return self.result.duration_ms
