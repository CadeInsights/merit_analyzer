"""Abstract base class for storage backends."""

from abc import ABC, abstractmethod
from uuid import UUID

from merit.testing.models.run import MeritRun


class Store(ABC):
    """Abstract storage backend for Merit test runs."""

    @abstractmethod
    def save_run(self, run: MeritRun) -> None:
        """Save a complete test run."""

    @abstractmethod
    def get_run(self, run_id: UUID) -> MeritRun | None:
        """Retrieve a test run by ID."""

    @abstractmethod
    def list_runs(self, limit: int = 10) -> list[MeritRun]:
        """List recent runs, ordered by start_time descending."""
