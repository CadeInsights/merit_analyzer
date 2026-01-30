"""Migration runner for SQLite databases."""

import importlib
import pkgutil
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from merit.storage.sqlite.migrations import versions as versions_package
from merit.storage.sqlite.migrations.errors import MigrationConfigError


@dataclass
class Migration:
    """Represents a single migration."""

    version: int
    name: str
    up: Callable[[sqlite3.Connection], None]


class MigrationRunner:
    """Discovers and executes database migrations."""

    def __init__(self, db_path: Path) -> None:
        self.path = db_path
        self._migrations = self._discover_migrations()
        self._validate_migrations(self._migrations)

    def _discover_migrations(self) -> list[Migration]:
        """Discover all migration modules in the versions package."""
        migrations: list[Migration] = []

        for module_info in pkgutil.iter_modules(versions_package.__path__):
            if module_info.name.startswith("_"):
                continue

            module = importlib.import_module(
                f"merit.storage.sqlite.migrations.versions.{module_info.name}"
            )
            migration = self._load_migration(module, module_info.name)
            if migration:
                migrations.append(migration)

        return sorted(migrations, key=lambda m: m.version)

    def _load_migration(self, module: ModuleType, name: str) -> Migration | None:
        """Load a migration from a module."""
        version = getattr(module, "VERSION", None)
        up_fn = getattr(module, "up", None)

        if version is None or up_fn is None:
            return None

        return Migration(version=version, name=name, up=up_fn)

    def _validate_migrations(self, migrations: list[Migration]) -> None:
        """Validate migration files are correctly structured."""
        if not migrations:
            raise MigrationConfigError

        versions = [m.version for m in migrations]

        if len(versions) != len(set(versions)):
            msg = "Duplicate migration versions detected"
            raise MigrationConfigError(msg)

        if versions[0] != 1:
            msg = f"Migrations must start at version 1, found {versions[0]}"
            raise MigrationConfigError(msg)

        for i, version in enumerate(versions):
            expected = i + 1
            if version != expected:
                msg = f"Migration version gap: expected {expected}, found {version}"
                raise MigrationConfigError(msg)

    def _connect(self) -> sqlite3.Connection:
        """Create a database connection."""
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _get_version(self, conn: sqlite3.Connection) -> int:
        """Get current schema version from database."""
        result: int = conn.execute("PRAGMA user_version").fetchone()[0]
        return result

    def get_current_version(self) -> int:
        """Read PRAGMA user_version from DB."""
        if not self.path.exists():
            return 0
        with self._connect() as conn:
            return self._get_version(conn)

    def get_target_version(self) -> int:
        """Return highest VERSION from migration files."""
        if not self._migrations:
            return 0
        return self._migrations[-1].version

    def get_pending_migrations(self) -> list[Migration]:
        """Return migrations between current and target version."""
        current = self.get_current_version()
        return [m for m in self._migrations if m.version > current]

    def needs_migration(self) -> bool:
        """Return True if current_version < target_version."""
        return self.get_current_version() < self.get_target_version()

    def can_migrate(self) -> bool:
        """Check if migration path exists."""
        current = self.get_current_version()
        target = self.get_target_version()

        if current > target:
            return False

        for version in range(current + 1, target + 1):
            if not any(m.version == version for m in self._migrations):
                return False

        return True

    def migrate(self) -> None:
        """Apply all pending migrations in a single exclusive transaction."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.execute("BEGIN EXCLUSIVE")

            current = self._get_version(conn)
            target = self.get_target_version()

            if current >= target:
                conn.rollback()
                return

            pending = [m for m in self._migrations if m.version > current]

            for migration in pending:
                migration.up(conn)

            conn.execute(f"PRAGMA user_version = {target}")
            conn.commit()
