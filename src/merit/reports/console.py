"""Console reporter for merit test output using Rich."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from merit.reports.base import Reporter

if TYPE_CHECKING:
    from merit.testing.discovery import TestItem
    from merit.testing.runner import MeritRun, TestExecution


class ConsoleReporter(Reporter):
    """Reporter that outputs test results to the console using Rich formatting."""

    def __init__(self, console: Console | None = None, verbosity: int = 0) -> None:
        self.console = console or Console()
        self.verbosity = verbosity

    async def on_no_tests_found(self) -> None:
        self.console.print("[yellow]No tests found.[/yellow]")

    async def on_collection_complete(self, items: list[TestItem]) -> None:
        self.console.print(f"[bold]Collected {len(items)} tests[/bold]\n")

    async def on_test_complete(self, execution: TestExecution) -> None:
        from merit.testing.runner import TestStatus

        result = execution.result
        item = execution.item

        if self.verbosity < 0 and result.status not in {TestStatus.FAILED, TestStatus.ERROR}:
            return

        # Handle repeated tests with sub-results
        if result.repeat_runs:
            passed_count = sum(1 for r in result.repeat_runs if r.status == TestStatus.PASSED)
            total_count = len(result.repeat_runs)
            min_passes = item.repeat_min_passes or total_count

            if result.status == TestStatus.PASSED:
                self.console.print(
                    f"  [green]✓[/green] {item.full_name} [dim]({result.duration_ms:.1f}ms)[/dim] "
                    f"[green]{passed_count}/{total_count} passed (≥{min_passes} required)[/green]"
                )
            else:
                self.console.print(
                    f"  [red]✗[/red] {item.full_name} [dim]({result.duration_ms:.1f}ms)[/dim] "
                    f"[red]{passed_count}/{total_count} passed (≥{min_passes} required)[/red]"
                )
            return

        if result.status == TestStatus.PASSED:
            self.console.print(
                f"  [green]✓[/green] {item.full_name} [dim]({result.duration_ms:.1f}ms)[/dim]"
            )
        elif result.status == TestStatus.FAILED:
            self.console.print(
                f"  [red]✗[/red] {item.full_name} [dim]({result.duration_ms:.1f}ms)[/dim]"
            )
            if result.error:
                self.console.print(f"    [red]{result.error}[/red]")
        elif result.status == TestStatus.ERROR:
            self.console.print(
                f"  [yellow]![/yellow] {item.full_name} [dim]({result.duration_ms:.1f}ms)[/dim]"
            )
            if result.error:
                self.console.print(
                    f"    [yellow]{type(result.error).__name__}: {result.error}[/yellow]"
                )
        elif result.status == TestStatus.SKIPPED:
            reason = result.error.args[0] if result.error else "skipped"
            self.console.print(
                f"  [yellow]-[/yellow] {item.full_name} [dim]skipped ({reason})[/dim]"
            )
        elif result.status == TestStatus.XFAILED:
            reason = result.error.args[0] if result.error else "expected failure"
            self.console.print(
                f"  [blue]x[/blue] {item.full_name} [dim]xfailed ({reason})[/dim]"
            )
        elif result.status == TestStatus.XPASSED:
            self.console.print(f"  [magenta]![/magenta] {item.full_name} [dim]XPASS[/dim]")

    async def on_run_complete(self, merit_run: MeritRun) -> None:
        self.console.print()
        result = merit_run.result
        parts = []
        if result.passed:
            parts.append(f"[green]{result.passed} passed[/green]")
        if result.failed:
            parts.append(f"[red]{result.failed} failed[/red]")
        if result.errors:
            parts.append(f"[yellow]{result.errors} errors[/yellow]")
        if result.skipped:
            parts.append(f"[yellow]{result.skipped} skipped[/yellow]")
        if result.xfailed:
            parts.append(f"[blue]{result.xfailed} xfailed[/blue]")
        if result.xpassed:
            parts.append(f"[magenta]{result.xpassed} xpassed[/magenta]")

        summary = ", ".join(parts) if parts else "[dim]0 tests[/dim]"
        self.console.print(f"[bold]{summary}[/bold] in {result.total_duration_ms:.0f}ms")

        if result.stopped_early:
            self.console.print("[yellow]Run terminated early due to maxfail limit.[/yellow]")

    async def on_run_stopped_early(self, failure_count: int) -> None:
        self.console.print(f"[red]Stopping early after {failure_count} failure(s).[/red]")

    async def on_tracing_enabled(self, output_path: Path) -> None:
        if output_path.exists():
            self.console.print(
                f"[dim]Tracing written to {output_path} ({output_path.stat().st_size} bytes)[/dim]"
            )
