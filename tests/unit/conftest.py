"""Shared fixtures for unit tests."""

from pathlib import Path

import pytest

from merit.reports.base import Reporter


class NullReporter(Reporter):
    """Silent reporter for testing."""

    async def on_no_tests_found(self) -> None:
        pass

    async def on_collection_complete(self, items) -> None:
        pass

    async def on_test_complete(self, execution) -> None:
        pass

    async def on_run_complete(self, merit_run) -> None:
        pass

    async def on_run_stopped_early(self, failure_count: int) -> None:
        pass

    async def on_tracing_enabled(self, output_path: Path) -> None:
        pass


@pytest.fixture
def null_reporter() -> NullReporter:
    """Provide a silent reporter for tests."""
    return NullReporter()
