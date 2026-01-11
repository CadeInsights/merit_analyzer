"""Base reporter ABC for merit test output."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from merit.testing.discovery import TestItem
    from merit.testing.runner import MeritRun, TestExecution


class Reporter(ABC):
    """Abstract base class for test reporters.

    All methods are async to support I/O-bound reporters (web dashboard, file output, etc.).
    """

    @abstractmethod
    async def on_no_tests_found(self) -> None:
        """Called when test collection finds no tests."""

    @abstractmethod
    async def on_collection_complete(self, items: list[TestItem]) -> None:
        """Called after test collection completes."""

    @abstractmethod
    async def on_test_complete(self, execution: TestExecution) -> None:
        """Called after each test completes."""

    @abstractmethod
    async def on_run_complete(self, merit_run: MeritRun) -> None:
        """Called after all tests complete."""

    @abstractmethod
    async def on_run_stopped_early(self, failure_count: int) -> None:
        """Called when run stops early due to maxfail limit."""

    @abstractmethod
    async def on_tracing_enabled(self, output_path: Path) -> None:
        """Called when tracing is enabled to report output location."""
