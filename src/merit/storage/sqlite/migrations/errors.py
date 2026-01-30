"""Migration error types."""

from pathlib import Path


class MigrationConfigError(Exception):
    """Raised when migration files are misconfigured (developer error)."""


class MigrationError(Exception):
    """Raised when database migration is not possible."""

    def __init__(
        self,
        current_version: int,
        target_version: int,
        db_path: Path,
        cause: Exception | None = None,
    ) -> None:
        self.current_version = current_version
        self.target_version = target_version
        self.db_path = db_path
        self.cause = cause

        message = f"""
Database migration not possible.

Current schema version: {current_version}
Required schema version: {target_version}
Database path: {db_path}

To preserve your data, run:
    merit db backup

Then reset the database:
    merit db reset --yes

This will delete all existing test run data.
"""
        if cause:
            message += f"\nCause: {cause}"

        super().__init__(message)
