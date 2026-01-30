"""CLI module for Merit test runner."""

from __future__ import annotations

import argparse
import asyncio
import shlex
import sqlite3
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

from merit.config import MeritConfig, load_config
from merit.storage.sqlite.migrations import MigrationRunner
from merit.storage.sqlite.store import DEFAULT_DB_NAME, MAX_STORED_RUNS, find_project_root
from merit.testing import MeritTestDefinition, collect
from merit.testing.runner import Runner


# Backwards compatibility alias
TestItem = MeritTestDefinition


def main() -> None:
    """Entry point for merit CLI."""
    config = load_config()
    parser = _build_parser()
    argv = [*config.addopts, *sys.argv[1:]] if config.addopts else sys.argv[1:]
    args = parser.parse_args(argv)

    if args.command == "test":
        exit_code = asyncio.run(_run_tests(args, config))
        raise SystemExit(exit_code)

    if args.command == "db":
        exit_code = _run_db_command(args, config)
        raise SystemExit(exit_code)

    parser.print_help()
    raise SystemExit(0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="merit", description="Merit AI Testing Framework")
    subparsers = parser.add_subparsers(dest="command")

    test_parser = subparsers.add_parser("test", help="Run merit tests")
    test_parser.add_argument("paths", nargs="*", help="Test files or directories")
    test_parser.add_argument("-k", "--keyword", help="Filter tests by keyword expression")
    test_parser.add_argument(
        "-t", "--tag", dest="include_tags", action="append", help="Run tests with given tag"
    )
    test_parser.add_argument(
        "--skip-tag",
        dest="exclude_tags",
        action="append",
        help="Skip tests that match this tag",
    )
    test_parser.add_argument("--maxfail", type=int, help="Stop after this many failures")
    test_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop test after the first failed assertion",
    )
    test_parser.add_argument(
        "--concurrency",
        type=int,
        help="Number of concurrent tests (default: 1, 0 for unlimited up to 10)",
    )
    test_parser.add_argument(
        "--timeout",
        type=float,
        help="Global test run timeout in seconds",
    )
    test_parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable OpenTelemetry tracing of tests and SUT calls",
    )
    test_parser.add_argument(
        "--trace-output",
        type=str,
        default=".merit/traces.jsonl",
        help="Output path for trace data (default: .merit/traces.jsonl)",
    )
    test_parser.add_argument("-q", "--quiet", action="count", default=0, help="Reduce CLI output")
    test_parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase CLI output"
    )
    test_parser.add_argument(
        "-s",
        "--show-output",
        action="store_true",
        help="Show stdout/stderr live (still captured)",
    )
    test_parser.add_argument(
        "--db-path",
        type=str,
        help="Path to the Merit SQLite database",
    )
    test_parser.add_argument(
        "--no-db",
        action="store_true",
        help="Disable writing run data to SQLite",
    )

    db_parser = subparsers.add_parser("db", help="Database management commands")
    db_subparsers = db_parser.add_subparsers(dest="db_command")

    db_subparsers.add_parser("status", help="Show database schema version status")

    migrate_parser = db_subparsers.add_parser("migrate", help="Run pending migrations")
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what migrations would run without applying them",
    )
    db_subparsers.add_parser("backup", help="Create a backup of the database")

    reset_parser = db_subparsers.add_parser("reset", help="Delete and recreate the database")
    reset_parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive reset without prompting",
    )

    for p in [db_parser, *db_subparsers.choices.values()]:
        p.add_argument("--db-path", type=str, help="Path to the Merit SQLite database")

    return parser


def _get_db_path(args: argparse.Namespace, config: MeritConfig) -> Path:
    """Resolve database path from args or config."""
    if args.db_path:
        return Path(args.db_path)
    if config.db_path:
        return Path(config.db_path)
    return find_project_root() / DEFAULT_DB_NAME


def _run_db_command(args: argparse.Namespace, config: MeritConfig) -> int:
    """Handle db subcommands."""
    console = Console()
    db_path = _get_db_path(args, config)

    if args.db_command == "status":
        return _db_status(console, db_path)
    if args.db_command == "migrate":
        return _db_migrate(console, db_path, dry_run=args.dry_run)
    if args.db_command == "backup":
        return _db_backup(console, db_path)
    if args.db_command == "reset":
        return _db_reset(console, db_path, confirmed=args.yes)

    console.print("Usage: merit db <status|migrate|backup|reset>")
    return 1


def _db_status(console: Console, db_path: Path) -> int:
    """Show database schema version status."""
    runner = MigrationRunner(db_path)
    current = runner.get_current_version()
    target = runner.get_target_version()
    pending = len(runner.get_pending_migrations())

    console.print(f"Database: {db_path}")
    console.print(f"Current version: {current}")
    console.print(f"Target version: {target}")
    console.print(f"Max stored runs: {MAX_STORED_RUNS}")

    if current == target:
        console.print("[green]Status: Up to date[/green]")
    elif current > target:
        console.print("[red]Status: Database ahead of code (downgrade not supported)[/red]")
    else:
        console.print(f"[yellow]Status: Migration required ({pending} pending)[/yellow]")

    return 0


def _db_migrate(console: Console, db_path: Path, *, dry_run: bool = False) -> int:
    """Run pending migrations."""
    runner = MigrationRunner(db_path)

    if not runner.needs_migration():
        console.print("[green]Database is up to date.[/green]")
        return 0

    if not runner.can_migrate():
        console.print("[red]Migration not possible.[/red]")
        console.print("Run 'merit db backup' then 'merit db reset --yes'")
        return 1

    pending = runner.get_pending_migrations()

    if dry_run:
        console.print(f"[yellow]Dry run: {len(pending)} migration(s) would be applied:[/yellow]")
        for migration in pending:
            console.print(f"  - v{migration.version:03d}: {migration.name}")
        return 0

    console.print(f"Applying {len(pending)} migration(s)...")
    runner.migrate()
    console.print(f"[green]Migrated to version {runner.get_target_version()}[/green]")
    return 0


def _db_backup(console: Console, db_path: Path) -> int:
    """Create a backup of the database using SQLite's backup API for WAL safety."""
    if not db_path.exists():
        console.print(f"[red]Database not found: {db_path}[/red]")
        return 1

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_suffix(f".db.backup.{timestamp}")

    with sqlite3.connect(db_path) as source, sqlite3.connect(backup_path) as dest:
        source.backup(dest)

    console.print(f"Backed up database to: {backup_path}")
    return 0


def _db_reset(console: Console, db_path: Path, *, confirmed: bool) -> int:
    """Delete and recreate the database."""
    if not confirmed:
        console.print("[bold red]WARNING: This will DELETE all test run history.[/bold red]")
        console.print()
        console.print(f"Database: {db_path}")
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            console.print(f"Size: {size_mb:.1f} MB")
        console.print()
        console.print("To proceed, run:")
        console.print("    merit db reset --yes")
        console.print()
        console.print("Consider backing up first:")
        console.print("    merit db backup")
        return 1

    if db_path.exists():
        db_path.unlink()
        # Clean up WAL mode auxiliary files if present
        for suffix in ("-wal", "-shm"):
            wal_file = db_path.with_name(db_path.name + suffix)
            if wal_file.exists():
                wal_file.unlink()

    runner = MigrationRunner(db_path)
    runner.migrate()

    console.print(f"[green]Database reset complete: {db_path}[/green]")
    console.print(f"Schema version: {runner.get_target_version()}")
    return 0


def _resolve_paths(args: argparse.Namespace, config: MeritConfig) -> list[str]:
    if args.paths:
        return args.paths
    return config.test_paths


def _resolve_tags(args: argparse.Namespace, config: MeritConfig) -> tuple[list[str], list[str]]:
    include = list(config.include_tags)
    exclude = list(config.exclude_tags)
    if args.include_tags:
        include.extend(args.include_tags)
    if args.exclude_tags:
        exclude.extend(args.exclude_tags)
    return include, exclude


def _resolve_keyword(args: argparse.Namespace, config: MeritConfig) -> str | None:
    return args.keyword or config.keyword


def _resolve_maxfail(args: argparse.Namespace, config: MeritConfig) -> int | None:
    if args.maxfail is not None:
        return args.maxfail if args.maxfail > 0 else None
    return config.maxfail


def _resolve_verbosity(args: argparse.Namespace, config: MeritConfig) -> int:
    return config.verbosity + args.verbose - args.quiet


def _resolve_concurrency(args: argparse.Namespace, config: MeritConfig) -> int:
    if args.concurrency is not None:
        return max(0, args.concurrency)
    return config.concurrency


def _resolve_timeout(args: argparse.Namespace, config: MeritConfig) -> float | None:
    if args.timeout is not None:
        return args.timeout if args.timeout > 0 else None
    return config.timeout


def _collect_items(paths: Sequence[str]) -> list[TestItem]:
    items: list[TestItem] = []
    for path in paths:
        items.extend(collect(path))
    return items


def _filter_items(
    items: list[TestItem],
    include_tags: Sequence[str],
    exclude_tags: Sequence[str],
    keyword: str | None,
) -> list[TestItem]:
    filtered = items

    if include_tags:
        include = set(include_tags)
        filtered = [item for item in filtered if item.tags & include]

    if exclude_tags:
        exclude = set(exclude_tags)
        filtered = [item for item in filtered if not (item.tags & exclude)]

    if keyword:
        matcher = KeywordMatcher(keyword)
        filtered = [item for item in filtered if matcher.match(item.full_name)]

    return filtered


async def _run_tests(args: argparse.Namespace, config: MeritConfig) -> int:
    paths = _resolve_paths(args, config)
    include_tags, exclude_tags = _resolve_tags(args, config)
    keyword = _resolve_keyword(args, config)
    maxfail = _resolve_maxfail(args, config)
    verbosity = _resolve_verbosity(args, config)
    concurrency = _resolve_concurrency(args, config)
    timeout = _resolve_timeout(args, config)
    db_path = args.db_path or config.db_path
    save_to_db = config.save_to_db
    if args.no_db:
        save_to_db = False

    items = _collect_items(paths)
    try:
        items = _filter_items(items, include_tags, exclude_tags, keyword)
    except ValueError as exc:
        Console().print(f"[red]{exc}[/red]")
        return 2

    runner = Runner(
        maxfail=maxfail,
        verbosity=verbosity,
        concurrency=concurrency,
        timeout=timeout,
        enable_tracing=args.trace,
        trace_output=args.trace_output,
        fail_fast=args.fail_fast,
        capture_output=not args.show_output,
        save_to_db=save_to_db,
        db_path=db_path,
    )
    merit_run = await runner.run(items=items)

    return 0 if merit_run.result.failed == 0 and merit_run.result.errors == 0 else 1


class KeywordMatcher:
    """Evaluate pytest-style -k expressions."""

    def __init__(self, expression: str) -> None:
        self.tokens = shlex.split(expression)
        self.index = 0
        self.func = self._parse_or()
        if self._peek() is not None:
            msg = "Invalid keyword expression"
            raise ValueError(msg)

    def match(self, text: str) -> bool:
        return self.func(text)

    def _parse_or(self) -> Callable[[str], bool]:
        left = self._parse_and()
        while self._peek_word("or"):
            self._advance()
            right = self._parse_and()
            prev = left
            left = lambda text, prev=prev, right=right: prev(text) or right(text)
        return left

    def _parse_and(self) -> Callable[[str], bool]:
        left = self._parse_not()
        while self._peek_word("and"):
            self._advance()
            right = self._parse_not()
            prev = left
            left = lambda text, prev=prev, right=right: prev(text) and right(text)
        return left

    def _parse_not(self) -> Callable[[str], bool]:
        if self._peek_word("not"):
            self._advance()
            operand = self._parse_not()
            return lambda text, operand=operand: not operand(text)
        return self._parse_term()

    def _parse_term(self) -> Callable[[str], bool]:
        token = self._peek()
        if token is None:
            msg = "Unexpected end of keyword expression"
            raise ValueError(msg)
        if token == "(":
            self._advance()
            expr = self._parse_or()
            if not self._peek_word(")"):
                msg = "Unmatched '(' in keyword expression"
                raise ValueError(msg)
            self._advance()
            return expr
        if token == ")":
            msg = "Unexpected ')' in keyword expression"
            raise ValueError(msg)
        self._advance()
        literal = token
        return lambda text, literal=literal: literal in text

    def _peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def _peek_word(self, word: str) -> bool:
        token = self._peek()
        return token is not None and token.lower() == word

    def _advance(self) -> None:
        self.index += 1


__all__ = ["KeywordMatcher", "main"]
