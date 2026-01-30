"""Tests for merit db CLI commands."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

from merit.cli import _db_backup, _db_migrate, _db_reset, _db_status
from merit.storage.sqlite.migrations import MigrationRunner


class TestDbStatus:
    """Tests for merit db status command."""

    def test_status_fresh_db(self, tmp_path: Path) -> None:
        """Status should show migration required for fresh DB."""
        db_path = tmp_path / "test.db"
        console = MagicMock()

        result = _db_status(console, db_path)

        assert result == 0
        calls = [str(c) for c in console.print.call_args_list]
        assert any("Current version: 0" in c for c in calls)
        assert any("Migration required" in c for c in calls)

    def test_status_up_to_date(self, tmp_path: Path) -> None:
        """Status should show up to date after migration."""
        db_path = tmp_path / "test.db"
        runner = MigrationRunner(db_path)
        runner.migrate()

        console = MagicMock()
        result = _db_status(console, db_path)

        assert result == 0
        calls = [str(c) for c in console.print.call_args_list]
        assert any("Up to date" in c for c in calls)


class TestDbMigrate:
    """Tests for merit db migrate command."""

    def test_migrate_fresh_db(self, tmp_path: Path) -> None:
        """Migrate should create and populate fresh DB."""
        db_path = tmp_path / "test.db"
        console = MagicMock()

        result = _db_migrate(console, db_path)

        assert result == 0
        assert db_path.exists()

        conn = sqlite3.connect(db_path)
        version: int = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version > 0
        conn.close()

    def test_migrate_already_current(self, tmp_path: Path) -> None:
        """Migrate should report up to date if no migrations needed."""
        db_path = tmp_path / "test.db"
        runner = MigrationRunner(db_path)
        runner.migrate()

        console = MagicMock()
        result = _db_migrate(console, db_path)

        assert result == 0
        calls = [str(c) for c in console.print.call_args_list]
        assert any("up to date" in c for c in calls)

    def test_migrate_dry_run_shows_pending(self, tmp_path: Path) -> None:
        """Dry run should show pending migrations without applying."""
        db_path = tmp_path / "test.db"
        console = MagicMock()

        result = _db_migrate(console, db_path, dry_run=True)

        assert result == 0
        assert not db_path.exists()  # DB should not be created
        calls = [str(c) for c in console.print.call_args_list]
        assert any("Dry run" in c for c in calls)

    def test_migrate_dry_run_when_current(self, tmp_path: Path) -> None:
        """Dry run on current DB should report up to date."""
        db_path = tmp_path / "test.db"
        runner = MigrationRunner(db_path)
        runner.migrate()

        console = MagicMock()
        result = _db_migrate(console, db_path, dry_run=True)

        assert result == 0
        calls = [str(c) for c in console.print.call_args_list]
        assert any("up to date" in c for c in calls)


class TestDbBackup:
    """Tests for merit db backup command."""

    def test_backup_creates_timestamped_copy(self, tmp_path: Path) -> None:
        """Backup should create a timestamped copy of the database."""
        db_path = tmp_path / "test.db"
        runner = MigrationRunner(db_path)
        runner.migrate()

        console = MagicMock()
        result = _db_backup(console, db_path)

        assert result == 0
        backup_files = list(tmp_path.glob("*.backup.*"))
        assert len(backup_files) == 1

    def test_backup_missing_db_fails(self, tmp_path: Path) -> None:
        """Backup should fail if database doesn't exist."""
        db_path = tmp_path / "nonexistent.db"
        console = MagicMock()

        result = _db_backup(console, db_path)

        assert result == 1
        calls = [str(c) for c in console.print.call_args_list]
        assert any("not found" in c for c in calls)

    def test_backup_includes_wal_contents(self, tmp_path: Path) -> None:
        """Backup should include uncommitted WAL data."""
        db_path = tmp_path / "test.db"
        runner = MigrationRunner(db_path)
        runner.migrate()

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(
            "INSERT INTO runs (run_id, start_time, total_duration_ms, "
            "passed, failed, errors, skipped, xfailed, xpassed, total, stopped_early) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("wal-test-run", "2024-01-01T00:00:00", 100.0, 1, 0, 0, 0, 0, 0, 1, 0),
        )
        conn.commit()
        conn.close()

        console = MagicMock()
        result = _db_backup(console, db_path)
        assert result == 0

        backup_files = list(tmp_path.glob("*.backup.*"))
        assert len(backup_files) == 1

        backup_conn = sqlite3.connect(backup_files[0])
        rows = backup_conn.execute("SELECT run_id FROM runs").fetchall()
        assert ("wal-test-run",) in rows
        backup_conn.close()


class TestDbReset:
    """Tests for merit db reset command."""

    def test_reset_without_yes_shows_warning(self, tmp_path: Path) -> None:
        """Reset without --yes should show warning and not delete."""
        db_path = tmp_path / "test.db"
        runner = MigrationRunner(db_path)
        runner.migrate()

        console = MagicMock()
        result = _db_reset(console, db_path, confirmed=False)

        assert result == 1
        assert db_path.exists()
        calls = [str(c) for c in console.print.call_args_list]
        assert any("WARNING" in c for c in calls)

    def test_reset_with_yes_recreates_db(self, tmp_path: Path) -> None:
        """Reset with --yes should delete and recreate database."""
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
        conn.commit()
        conn.close()

        console = MagicMock()
        result = _db_reset(console, db_path, confirmed=True)

        assert result == 0
        assert db_path.exists()

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert rows == 0
        conn.close()

    def test_reset_fresh_db_creates_new(self, tmp_path: Path) -> None:
        """Reset on non-existent DB should create new one."""
        db_path = tmp_path / "new.db"
        console = MagicMock()

        result = _db_reset(console, db_path, confirmed=True)

        assert result == 0
        assert db_path.exists()
