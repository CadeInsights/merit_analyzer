"""Run-level models."""

from __future__ import annotations

import os
import platform
import socket
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from typing import Any
from uuid import UUID, uuid4

from merit.metrics_.base import MetricResult
from merit.testing.models.result import TestExecution, TestStatus


def _get_python_version() -> str:
    return sys.version.split()[0]


def _get_platform() -> str:
    return platform.platform()


def _get_hostname() -> str:
    return socket.gethostname()


def _get_cwd() -> str:
    return os.getcwd()


def _get_merit_version() -> str:
    return version("appmerit")


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
            "xfailed": self.result.xfailed,
            "xpassed": self.result.xpassed,
            "total": self.result.total,
        }
