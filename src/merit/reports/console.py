"""Console reporter for merit test output using Rich."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markup import escape

from merit.reports.base import Reporter


if TYPE_CHECKING:
    from merit.metrics.base import MetricResult
    from merit.testing.discovery import TestItem
    from merit.testing.models import TestResult
    from merit.testing.runner import MeritRun, TestExecution

from merit.metrics.base import Scope


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
        from merit.testing.models import TestStatus

        result = execution.result
        item = execution.item

        if self.verbosity < 0 and result.status not in {TestStatus.FAILED, TestStatus.ERROR}:
            return

        # Handle tests with sub-results (repeat/parametrize)
        if result.sub_runs:
            passed_count = sum(1 for r in result.sub_runs if r.status == TestStatus.PASSED)
            total_count = len(result.sub_runs)

            if result.status == TestStatus.PASSED:
                self.console.print(
                    f"  [green]✓[/green] {item.full_name} [dim]({result.duration_ms:.1f}ms)[/dim] "
                    f"[green]{passed_count}/{total_count} passed[/green]"
                )
            else:
                self.console.print(
                    f"  [red]✗[/red] {item.full_name} [dim]({result.duration_ms:.1f}ms)[/dim] "
                    f"[red]{passed_count}/{total_count} passed[/red]"
                )

            self._print_sub_runs(result.sub_runs, indent=4)
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
            self.console.print(f"  [blue]x[/blue] {item.full_name} [dim]xfailed ({reason})[/dim]")
        elif result.status == TestStatus.XPASSED:
            self.console.print(f"  [magenta]![/magenta] {item.full_name} [dim]XPASS[/dim]")

    def _print_sub_runs(self, sub_runs: list[TestResult], indent: int) -> None:
        """Print sub-runs with their id_suffix, recursively handling nested sub-runs."""
        from merit.testing.models import TestStatus

        prefix = " " * indent
        for sub in sub_runs:
            suffix = f"\\[{escape(sub.id_suffix)}]" if sub.id_suffix else ""

            if sub.status == TestStatus.PASSED:
                self.console.print(
                    f"{prefix}[green]✓[/green] {suffix} [dim]({sub.duration_ms:.1f}ms)[/dim]"
                )
            elif sub.status == TestStatus.FAILED:
                self.console.print(
                    f"{prefix}[red]✗[/red] {suffix} [dim]({sub.duration_ms:.1f}ms)[/dim]"
                )
                if sub.error:
                    self.console.print(f"{prefix}  [red]{escape(str(sub.error))}[/red]")
            elif sub.status == TestStatus.ERROR:
                self.console.print(
                    f"{prefix}[yellow]![/yellow] {suffix} [dim]({sub.duration_ms:.1f}ms)[/dim]"
                )
                if sub.error:
                    self.console.print(
                        f"{prefix}  [yellow]{type(sub.error).__name__}: {escape(str(sub.error))}[/yellow]"
                    )

            if sub.sub_runs:
                self._print_sub_runs(sub.sub_runs, indent + 2)

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

        self._print_metric_results(result.metric_results)

    def _print_metric_results(self, metric_results: list[MetricResult]) -> None:
        if not metric_results:
            return

        failed_metrics = []
        passed_metrics = []

        for metric in metric_results:
            assertions = metric.assertion_results
            has_failures = any(not a.passed for a in assertions)
            if has_failures:
                failed_metrics.append(metric)
            else:
                passed_metrics.append(metric)

        if self.verbosity < 0 and not failed_metrics:
            return

        self.console.print("\n[bold]Metrics:[/bold]")

        for metric in failed_metrics:
            self._print_metric(metric, failed=True)

        if self.verbosity >= 0:
            for metric in passed_metrics:
                self._print_metric(metric, failed=False)

    def _print_metric(self, metric: MetricResult, failed: bool) -> None:
        value = metric.value
        if isinstance(value, float) and math.isnan(value):
            value_str = "N/A"
        else:
            value_str = str(value)

        assertions = metric.assertion_results
        passed = sum(1 for a in assertions if a.passed)
        total = len(assertions)

        name = metric.name
        if metric.metadata.scope == Scope.CASE and metric.metadata.collected_from_merits:
            merits = sorted(metric.metadata.collected_from_merits)
            name = f"{name}::{merits[0]}"

        if total > 0:
            color = "red" if failed else "green"
            self.console.print(
                f"  {name}: [bold]{value_str}[/bold] "
                f"[{color}]({passed}/{total} assertions passed)[/{color}]"
            )
        else:
            self.console.print(f"  {name}: [bold]{value_str}[/bold]")

    async def on_run_stopped_early(self, failure_count: int) -> None:
        self.console.print(f"[red]Stopping early after {failure_count} failure(s).[/red]")

    async def on_tracing_enabled(self, output_path: Path) -> None:
        if output_path.exists():
            self.console.print(
                f"[dim]Tracing written to {output_path} ({output_path.stat().st_size} bytes)[/dim]"
            )
