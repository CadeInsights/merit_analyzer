"""Data models for test discovery and execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union
from uuid import UUID, uuid4


if TYPE_CHECKING:
    from merit.assertions.base import AssertionResult
    from merit.metrics.base import MetricResult


@dataclass(frozen=True)
class ParameterSet:
    """Concrete parameter combination for an individual test run."""

    values: dict[str, Any]
    id_suffix: str


@dataclass(frozen=True)
class RepeatModifier:
    """Repeat the inner execution N times."""

    count: int
    min_passes: int


@dataclass(frozen=True)
class ParametrizeModifier:
    """Run the inner execution for each parameter set."""

    parameter_sets: tuple[ParameterSet, ...]


Modifier = Union[RepeatModifier, ParametrizeModifier]


@dataclass
class TestItem:
    """A discovered test function or method."""

    __test__ = False  # Prevent pytest from collecting this as a test class

    name: str
    fn: Callable[..., Any]
    module_path: Path
    is_async: bool
    params: list[str] = field(default_factory=list)
    class_name: str | None = None
    modifiers: list[Modifier] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)
    skip_reason: str | None = None
    xfail_reason: str | None = None
    xfail_strict: bool = False
    fail_fast: bool = False
    id_suffix: str | None = None

    @property
    def full_name(self) -> str:
        """Full qualified name for display."""
        if self.class_name:
            base = f"{self.module_path.stem}::{self.class_name}::{self.name}"
        else:
            base = f"{self.module_path.stem}::{self.name}"
        return base


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
    sub_runs: list[TestResult] | None = None
    id_suffix: str | None = None


@dataclass
class TestExecution:
    """Complete record of a test execution, combining context and result.

    Encapsulates both the test context (inputs/setup) and the result (outcome)
    as a single execution record.
    """

    __test__ = False  # Prevent pytest from collecting this as a test class

    context: Any  # TestContext - avoid circular import
    result: TestResult

    @property
    def item(self) -> TestItem:
        """The test item that was executed."""
        return self.context.item

    @property
    def status(self) -> TestStatus:
        """Convenience access to result status."""
        return self.result.status

    @property
    def duration_ms(self) -> float:
        """Convenience access to result duration."""
        return self.result.duration_ms


@dataclass
class RunResult:
    """Result of a complete test run."""

    executions: list[TestExecution] = field(default_factory=list)
    metric_results: list[MetricResult] = field(default_factory=list)
    total_duration_ms: float = 0
    stopped_early: bool = False

    @property
    def passed(self) -> int:
        """Count of passed tests."""
        return sum(1 for e in self.executions if e.status == TestStatus.PASSED)

    @property
    def failed(self) -> int:
        """Count of failed tests."""
        return sum(1 for e in self.executions if e.status == TestStatus.FAILED)

    @property
    def errors(self) -> int:
        """Count of errored tests."""
        return sum(1 for e in self.executions if e.status == TestStatus.ERROR)

    @property
    def skipped(self) -> int:
        """Count of skipped tests."""
        return sum(1 for e in self.executions if e.status == TestStatus.SKIPPED)

    @property
    def xfailed(self) -> int:
        """Count of expected failures."""
        return sum(1 for e in self.executions if e.status == TestStatus.XFAILED)

    @property
    def xpassed(self) -> int:
        """Count of unexpected passes for xfail tests."""
        return sum(1 for e in self.executions if e.status == TestStatus.XPASSED)

    @property
    def total(self) -> int:
        """Total test count."""
        return len(self.executions)


def _get_python_version() -> str:
    import sys

    return sys.version.split()[0]


def _get_platform() -> str:
    import platform

    return platform.platform()


def _get_hostname() -> str:
    import socket

    return socket.gethostname()


def _get_cwd() -> str:
    import os

    return os.getcwd()


def _get_merit_version() -> str:
    from merit.version import __version__

    return __version__


@dataclass
class RunEnvironment:
    """Metadata about the environment where tests were executed."""

    # Git info
    commit_hash: str | None = None
    branch: str | None = None
    dirty: bool | None = None

    # System info
    python_version: str = field(default_factory=_get_python_version)
    platform: str = field(default_factory=_get_platform)
    hostname: str = field(default_factory=_get_hostname)
    working_directory: str = field(default_factory=_get_cwd)
    merit_version: str = field(default_factory=_get_merit_version)

    # Environment variables (filtered)
    env_vars: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "commit_hash": self.commit_hash,
            "branch": self.branch,
            "dirty": self.dirty,
            "python_version": self.python_version,
            "platform": self.platform,
            "hostname": self.hostname,
            "working_directory": self.working_directory,
            "merit_version": self.merit_version,
            "env_vars": self.env_vars,
        }


@dataclass
class MeritRun:
    """Complete record of a test run, combining environment and results.

    This is created at the top level of a merit test run and encapsulates
    all information about the run, including environment metadata and
    test executions with their contexts.

    Access result data via merit_run.result.* (e.g., merit_run.result.passed).
    """

    run_id: UUID = field(default_factory=uuid4)
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None

    environment: RunEnvironment = field(default_factory=RunEnvironment)
    result: RunResult = field(default_factory=RunResult)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete run to a dictionary."""
        return {
            "run_id": str(self.run_id),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "environment": self.environment.to_dict(),
            "total_duration_ms": self.result.total_duration_ms,
            "stopped_early": self.result.stopped_early,
            "passed": self.result.passed,
            "failed": self.result.failed,
            "errors": self.result.errors,
            "skipped": self.result.skipped,
            "total": self.result.total,
        }
