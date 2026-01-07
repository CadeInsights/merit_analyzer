"""Base reporter protocol for merit test output."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from merit.testing.discovery import TestItem
    from merit.testing.runner import RunResult, TestResult


class Reporter(Protocol):
    """Protocol defining the interface for test reporters.

    All methods are async to support I/O-bound reporters (web dashboard, file output, etc.).
    Sync reporters can implement these as regular methods that don't await anything.
    """

    async def on_no_tests_found(self) -> None:
        """Called when test collection finds no tests."""
        ...

    async def on_collection_complete(self, items: list[TestItem]) -> None:
        """Called after test collection completes."""
        ...

    async def on_test_complete(self, result: TestResult) -> None:
        """Called after each test completes."""
        ...

    async def on_run_complete(self, run_result: RunResult) -> None:
        """Called after all tests complete."""
        ...

    async def on_run_stopped_early(self, failure_count: int) -> None:
        """Called when run stops early due to maxfail limit."""
        ...

    async def on_tracing_enabled(self, output_path: Path) -> None:
        """Called when tracing is enabled to report output location."""
        ...
