"""Tests for MigrationRunner."""

import sqlite3
from pathlib import Path

import pytest

from merit.storage.sqlite.migrations import MigrationConfigError, MigrationRunner
from merit.storage.sqlite.migrations.runner import Migration


class TestMigrationRunner:
    """Tests for migration runner functionality."""

    def test_fresh_db_runs_all_migrations(self, tmp_path: Path) -> None:
        """Fresh database should run all migrations."""
        db_path = tmp_path / "test.db"

        runner = MigrationRunner(db_path)
        runner.migrate()

        assert db_path.exists()
        conn = sqlite3.connect(db_path)
        version: int = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == runner.get_target_version()

        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {t[0] for t in tables}
        assert "runs" in table_names
        assert "test_executions" in table_names
        conn.close()

    def test_existing_db_applies_pending_migrations(self, tmp_path: Path) -> None:
        """Existing database should only apply pending migrations."""
        db_path = tmp_path / "test.db"

        runner = MigrationRunner(db_path)
        runner.migrate()

        initial_version = runner.get_current_version()
        runner2 = MigrationRunner(db_path)
        assert not runner2.needs_migration()
        assert runner2.get_current_version() == initial_version

    def test_needs_migration_returns_false_when_current(self, tmp_path: Path) -> None:
        """needs_migration returns False when DB is up to date."""
        db_path = tmp_path / "test.db"

        runner = MigrationRunner(db_path)
        runner.migrate()

        runner2 = MigrationRunner(db_path)
        assert not runner2.needs_migration()

    def test_can_migrate_returns_false_for_downgrade(self, tmp_path: Path) -> None:
        """can_migrate returns False when DB version is ahead of code."""
        db_path = tmp_path / "test.db"

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA user_version = 999")
        conn.commit()
        conn.close()

        runner = MigrationRunner(db_path)
        assert not runner.can_migrate()

    def test_get_current_version_fresh_db(self, tmp_path: Path) -> None:
        """Fresh DB should have version 0."""
        db_path = tmp_path / "nonexistent.db"
        runner = MigrationRunner(db_path)
        assert runner.get_current_version() == 0

    def test_get_pending_migrations(self, tmp_path: Path) -> None:
        """Should return list of migrations to apply."""
        db_path = tmp_path / "test.db"
        runner = MigrationRunner(db_path)

        pending = runner.get_pending_migrations()
        assert len(pending) > 0
        assert all(isinstance(m, Migration) for m in pending)

    def test_data_preserved_across_migration(self, tmp_path: Path) -> None:
        """Core value test: existing data survives schema migration."""
        db_path = tmp_path / "test.db"

        runner = MigrationRunner(db_path)
        runner.migrate()

        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO runs (run_id, start_time, total_duration_ms, "
            "passed, failed, errors, skipped, xfailed, xpassed, total, stopped_early) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("run-1", "2024-01-01T00:00:00", 100.0, 1, 0, 0, 0, 0, 0, 1, 0),
        )
        conn.execute(
            "INSERT INTO runs (run_id, start_time, total_duration_ms, "
            "passed, failed, errors, skipped, xfailed, xpassed, total, stopped_early) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("run-2", "2024-01-02T00:00:00", 200.0, 2, 0, 0, 0, 0, 0, 2, 0),
        )
        conn.commit()
        conn.close()

        runner2 = MigrationRunner(db_path)
        assert not runner2.needs_migration()

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT run_id FROM runs ORDER BY run_id").fetchall()
        assert [r[0] for r in rows] == ["run-1", "run-2"]
        conn.close()

    def test_injected_migration_applies_successfully(self, tmp_path: Path) -> None:
        """Injected migration should apply and bump schema version."""
        db_path = tmp_path / "test.db"

        runner = MigrationRunner(db_path)
        runner.migrate()
        original_version = runner.get_current_version()

        def add_dummy_table(conn: sqlite3.Connection) -> None:
            conn.execute("CREATE TABLE dummy_migration (id INTEGER)")

        dummy_migration = Migration(
            version=original_version + 1,
            name="dummy",
            up=add_dummy_table,
        )

        runner2 = MigrationRunner.__new__(MigrationRunner)
        runner2.path = db_path
        runner2._migrations = runner._migrations + [dummy_migration]

        runner2.migrate()

        conn = sqlite3.connect(db_path)
        version: int = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == original_version + 1

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dummy_migration'"
        ).fetchall()
        assert len(tables) == 1
        conn.close()

    def test_corrupted_db_no_user_version(self, tmp_path: Path) -> None:
        """DB exists but has no tables - should still work."""
        db_path = tmp_path / "corrupted.db"

        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE dummy (id INTEGER)")
        conn.commit()
        conn.close()

        runner = MigrationRunner(db_path)
        assert runner.get_current_version() == 0
        runner.migrate()
        assert runner.get_current_version() == runner.get_target_version()


class TestMigrationFailure:
    """Tests for migration failure scenarios."""

    def test_failed_migration_rolls_back(self, tmp_path: Path) -> None:
        """If migration.up() raises, all pending migrations roll back."""
        db_path = tmp_path / "test.db"

        runner = MigrationRunner(db_path)
        runner.migrate()
        original_version = runner.get_current_version()

        def failing_up(conn: sqlite3.Connection) -> None:
            conn.execute("CREATE TABLE new_table (id INTEGER)")
            raise RuntimeError("Simulated failure")

        bad_migration = Migration(version=original_version + 1, name="bad", up=failing_up)

        runner2 = MigrationRunner.__new__(MigrationRunner)
        runner2.path = db_path
        runner2._migrations = runner._migrations + [bad_migration]

        with pytest.raises(Exception, match="Simulated failure"):
            runner2.migrate()

        conn = sqlite3.connect(db_path)
        version: int = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == original_version

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='new_table'"
        ).fetchall()
        assert len(tables) == 0
        conn.close()


class TestMigrationValidation:
    """Tests for migration file validation."""

    def test_version_gap_raises_config_error(self, tmp_path: Path) -> None:
        """Gap in version numbers should raise MigrationConfigError."""
        migrations = [
            Migration(version=1, name="v001", up=lambda conn: None),
            Migration(version=3, name="v003", up=lambda conn: None),
        ]

        runner = MigrationRunner.__new__(MigrationRunner)
        runner.path = tmp_path / "test.db"
        runner._migrations = []

        with pytest.raises(MigrationConfigError, match="version gap"):
            runner._validate_migrations(migrations)

    def test_duplicate_version_raises_config_error(self, tmp_path: Path) -> None:
        """Duplicate versions should raise MigrationConfigError."""
        migrations = [
            Migration(version=1, name="v001_a", up=lambda conn: None),
            Migration(version=2, name="v002", up=lambda conn: None),
            Migration(version=2, name="v002_dup", up=lambda conn: None),
        ]

        runner = MigrationRunner.__new__(MigrationRunner)
        runner.path = tmp_path / "test.db"
        runner._migrations = []

        with pytest.raises(MigrationConfigError, match="Duplicate"):
            runner._validate_migrations(migrations)

    def test_no_migrations_raises_config_error(self, tmp_path: Path) -> None:
        """Empty migrations list should raise MigrationConfigError."""
        runner = MigrationRunner.__new__(MigrationRunner)
        runner.path = tmp_path / "test.db"
        runner._migrations = []

        with pytest.raises(MigrationConfigError):
            runner._validate_migrations([])

    def test_migrations_not_starting_at_1_raises_error(self, tmp_path: Path) -> None:
        """Migrations must start at version 1."""
        migrations = [
            Migration(version=2, name="v002", up=lambda conn: None),
        ]

        runner = MigrationRunner.__new__(MigrationRunner)
        runner.path = tmp_path / "test.db"
        runner._migrations = []

        with pytest.raises(MigrationConfigError, match="start at version 1"):
            runner._validate_migrations(migrations)


class TestMigrationVersions:
    """Validate actual migration files are correctly structured."""

    def test_all_migrations_have_version(self) -> None:
        """All migration files should have VERSION constant."""

        runner = MigrationRunner.__new__(MigrationRunner)
        runner.path = Path("/tmp/unused.db")
        migrations = runner._discover_migrations()

        assert len(migrations) > 0
        for m in migrations:
            assert m.version > 0

    def test_versions_are_sequential(self) -> None:
        """Migration versions should be sequential starting from 1."""
        runner = MigrationRunner.__new__(MigrationRunner)
        runner.path = Path("/tmp/unused.db")
        migrations = runner._discover_migrations()

        versions = [m.version for m in migrations]
        expected = list(range(1, len(versions) + 1))
        assert versions == expected

    def test_initial_migration_creates_all_tables(self, tmp_path: Path) -> None:
        """Initial migration should create all required tables."""
        db_path = tmp_path / "test.db"

        runner = MigrationRunner(db_path)
        runner.migrate()

        conn = sqlite3.connect(db_path)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {t[0] for t in tables}

        expected_tables = {
            "runs",
            "test_executions",
            "metrics",
            "assertions",
            "predicates",
            "trace_spans",
        }
        assert expected_tables.issubset(table_names)
        conn.close()
