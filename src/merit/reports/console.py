"""Console reporter for merit test output using Rich."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from merit.testing.discovery import TestItem
    from merit.testing.runner import RunResult, TestResult


class ConsoleReporter:
    """Reporter that outputs test results to the console using Rich formatting."""

    def __init__(self, console: Console | None = None, verbosity: int = 0) -> None:
        self.console = console or Console()
        self.verbosity = verbosity

    async def on_no_tests_found(self) -> None:
        self.console.print("[yellow]No tests found.[/yellow]")

    async def on_collection_complete(self, items: list[TestItem]) -> None:
        self.console.print(f"[bold]Collected {len(items)} tests[/bold]\n")

    async def on_test_complete(self, result: TestResult) -> None:
        from merit.testing.runner import TestStatus

        if self.verbosity < 0 and result.status not in {TestStatus.FAILED, TestStatus.ERROR}:
            return

        # Handle repeated tests with sub-results
        if result.repeat_runs:
            passed_count = sum(1 for r in result.repeat_runs if r.status == TestStatus.PASSED)
            total_count = len(result.repeat_runs)
            min_passes = result.item.repeat_min_passes or total_count

            if result.status == TestStatus.PASSED:
                self.console.print(
                    f"  [green]✓[/green] {result.item.full_name} [dim]({result.duration_ms:.1f}ms)[/dim] "
                    f"[green]{passed_count}/{total_count} passed (≥{min_passes} required)[/green]"
                )
            else:
                self.console.print(
                    f"  [red]✗[/red] {result.item.full_name} [dim]({result.duration_ms:.1f}ms)[/dim] "
                    f"[red]{passed_count}/{total_count} passed (≥{min_passes} required)[/red]"
                )
            return

        if result.status == TestStatus.PASSED:
            self.console.print(
                f"  [green]✓[/green] {result.item.full_name} [dim]({result.duration_ms:.1f}ms)[/dim]"
            )
        elif result.status == TestStatus.FAILED:
            self.console.print(
                f"  [red]✗[/red] {result.item.full_name} [dim]({result.duration_ms:.1f}ms)[/dim]"
            )
            if result.error:
                self.console.print(f"    [red]{result.error}[/red]")
        elif result.status == TestStatus.ERROR:
            self.console.print(
                f"  [yellow]![/yellow] {result.item.full_name} [dim]({result.duration_ms:.1f}ms)[/dim]"
            )
            if result.error:
                self.console.print(
                    f"    [yellow]{type(result.error).__name__}: {result.error}[/yellow]"
                )
        elif result.status == TestStatus.SKIPPED:
            reason = result.error.args[0] if result.error else "skipped"
            self.console.print(
                f"  [yellow]-[/yellow] {result.item.full_name} [dim]skipped ({reason})[/dim]"
            )
        elif result.status == TestStatus.XFAILED:
            reason = result.error.args[0] if result.error else "expected failure"
            self.console.print(
                f"  [blue]x[/blue] {result.item.full_name} [dim]xfailed ({reason})[/dim]"
            )
        elif result.status == TestStatus.XPASSED:
            self.console.print(f"  [magenta]![/magenta] {result.item.full_name} [dim]XPASS[/dim]")

    async def on_run_complete(self, run_result: RunResult) -> None:
        self.console.print()
        parts = []
        if run_result.passed:
            parts.append(f"[green]{run_result.passed} passed[/green]")
        if run_result.failed:
            parts.append(f"[red]{run_result.failed} failed[/red]")
        if run_result.errors:
            parts.append(f"[yellow]{run_result.errors} errors[/yellow]")
        if run_result.skipped:
            parts.append(f"[yellow]{run_result.skipped} skipped[/yellow]")
        if run_result.xfailed:
            parts.append(f"[blue]{run_result.xfailed} xfailed[/blue]")
        if run_result.xpassed:
            parts.append(f"[magenta]{run_result.xpassed} xpassed[/magenta]")

        summary = ", ".join(parts) if parts else "[dim]0 tests[/dim]"
        self.console.print(f"[bold]{summary}[/bold] in {run_result.total_duration_ms:.0f}ms")

        if run_result.stopped_early:
            self.console.print("[yellow]Run terminated early due to maxfail limit.[/yellow]")

    async def on_run_stopped_early(self, failure_count: int) -> None:
        self.console.print(f"[red]Stopping early after {failure_count} failure(s).[/red]")

    async def on_tracing_enabled(self, output_path: Path) -> None:
        if output_path.exists():
            self.console.print(
                f"[dim]Tracing written to {output_path} ({output_path.stat().st_size} bytes)[/dim]"
            )
