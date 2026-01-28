"""Console reporter for merit test output using Rich."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.traceback import Traceback

from merit.context import get_runner
from merit.reports.base import Reporter
from merit.testing.models.run import RunEnvironment


if TYPE_CHECKING:
    from merit.assertions.base import AssertionResult
    from merit.metrics_.base import MetricResult
    from merit.testing import MeritTestDefinition
    from merit.testing.models.result import TestExecution, TestResult
    from merit.testing.models.run import MeritRun

from merit.resources import Scope
from merit.testing.models import TestStatus


_STATUS_CONFIG: dict[TestStatus, tuple[str, str, str]] = {
    TestStatus.PASSED: ("✓", "green", "PASSED"),
    TestStatus.FAILED: ("✗", "red", "FAILED"),
    TestStatus.ERROR: ("!", "yellow", "ERROR"),
    TestStatus.SKIPPED: ("-", "yellow", "SKIPPED"),
    TestStatus.XFAILED: ("x", "blue", "XFAILED"),
    TestStatus.XPASSED: ("!", "magenta", "XPASSED"),
}


class ConsoleReporter(Reporter):
    """Reporter that outputs test results to the console using Rich formatting."""

    def __init__(self, console: Console | None = None, verbosity: int = 0) -> None:
        self.console = console or Console(file=sys.__stdout__)
        self.verbosity = verbosity
        self._failures: list[TestExecution] = []
        self._current_module: Path | None = None

    def _status_symbol(self, status: TestStatus) -> str:
        return _STATUS_CONFIG[status][0]

    def _status_color(self, status: TestStatus) -> str:
        return _STATUS_CONFIG[status][1]

    def _status_label(self, status: TestStatus) -> str:
        return _STATUS_CONFIG[status][2]

    def _print_section_header(self, title: str, *, include_footer: bool = False) -> None:
        width = self.console.width
        header_title = f" {title} "
        fill = max(width - len(header_title), 0)
        left = fill // 2
        right = fill - left
        self.console.print("=" * left + header_title + "=" * right)
        if include_footer:
            self.console.print("=" * width)

    def _safe_relative_path(self, path: Path) -> Path:
        try:
            return path.relative_to(Path.cwd())
        except ValueError:
            return path

    def _print_file_header(self, module_path: Path) -> None:
        if self._current_module is not None:
            self.console.print()
        relative_path = self._safe_relative_path(module_path)
        self.console.print(f"• {relative_path.as_posix()}")
        self._current_module = module_path

    def _print_run_header(self, environment: RunEnvironment, run_id: object) -> None:
        self._print_section_header("MERIT RUN STARTS")
        self.console.print(
            f"platform {environment.platform} -- python {environment.python_version} "
            f"-- merit {environment.merit_version}"
        )
        self.console.print(f"rootdir: {environment.working_directory}")
        if run_id:
            self.console.print(f"run_id: {run_id}")
        if environment.branch and environment.commit_hash:
            commit = environment.commit_hash[:8]
            dirty = " dirty" if environment.dirty else ""
            self.console.print(f"git: {environment.branch} ({commit}){dirty}")
        self.console.print()

    def _print_test_line(
        self, name: str, result: TestResult, indent: int = 2, extra: str = ""
    ) -> None:
        color = self._status_color(result.status)
        label = self._status_label(result.status)
        prefix = " " * indent + "• "
        duration = f"[dim]({result.duration_ms:.1f}ms)[/dim]"
        self.console.print(f"{prefix}{name} {duration} {extra}[{color}]{label}[/{color}]")

    def _print_sub_execution_line(self, sub: TestExecution, indent: int, marker: str) -> None:
        suffix = f"\\[{escape(sub.definition.id_suffix)}]" if sub.definition.id_suffix else ""
        color = self._status_color(sub.result.status)
        label = self._status_label(sub.result.status)
        prefix = " " * indent + marker + " • "
        duration = f"[dim]({sub.result.duration_ms:.1f}ms)[/dim]"
        self.console.print(f"{prefix}{suffix} {duration} [{color}]{label}[/{color}]")

    def _format_assertion_repr(self, assertion: AssertionResult) -> list[str]:
        lines = []
        expr = assertion.expression_repr
        lines.append(f"> {expr.expr}")
        if assertion.error_message:
            lines.append(assertion.error_message)
        else:
            lines.append(f"Assertion failed: {expr.expr}")
        if expr.resolved_args:
            lines.append(f"{expr.resolved_args}")
        return lines

    def _format_assertions(self, assertion_results: list[AssertionResult]) -> list[str]:
        lines: list[str] = []
        failed = [a for a in assertion_results if not a.passed]
        for assertion in failed:
            if lines:
                lines.append("")
            lines.extend(self._format_assertion_repr(assertion))
        return lines

    def _format_error(self, error: BaseException | None) -> list[str] | Traceback:
        if not error:
            return []
        if error.__traceback__:
            return Traceback.from_exception(
                type(error),
                error,
                error.__traceback__,
                suppress=[__import__("merit")],
                show_locals=self.verbosity >= 2,
            )
        return [f"{type(error).__name__}: {error}"]

    def _build_failure_lines(self, result: TestResult) -> list[str] | Traceback:
        lines = self._format_assertions(result.assertion_results)
        if lines:
            return lines
        return self._format_error(result.error)

    def _build_failure_panel(
        self, title: str, lines: list[str] | Traceback, color: str
    ) -> Panel:
        content: str | Traceback = (
            lines
            if isinstance(lines, Traceback)
            else "\n".join(escape(line) for line in lines) or " "
        )
        return Panel(
            content,
            title=title,
            title_align="left",
            border_style=color,
            expand=True,
            padding=(1, 1),
        )

    def _format_metric_value(self, metric: MetricResult) -> tuple[str, int, int, bool]:
        value = metric.value
        value_str = "N/A" if isinstance(value, float) and math.isnan(value) else str(value)
        assertions = metric.assertion_results
        passed = sum(1 for a in assertions if a.passed)
        total = len(assertions)
        has_failures = any(not a.passed for a in assertions)
        return value_str, passed, total, has_failures

    def _print_metric_row(
        self, label: str, stats: tuple[str, int, int, bool], indent: int = 1
    ) -> None:
        value_str, passed, total, failed = stats
        prefix = " " * indent + "• "
        color = "red" if failed else "green"
        if total > 0:
            self.console.print(
                f"{prefix}{label}: [bold]{value_str}[/bold] "
                f"[{color}]({passed}/{total} assertions passed)[/{color}]"
            )
        else:
            self.console.print(f"{prefix}{label}: [bold]{value_str}[/bold]")

    def _group_case_metrics(self, metrics: list[MetricResult]) -> dict[str, list[MetricResult]]:
        grouped: dict[str, list[MetricResult]] = {}
        for metric in metrics:
            grouped.setdefault(metric.name, []).append(metric)
        return grouped

    def _get_case_label(self, metric: MetricResult) -> str:
        cases = sorted(metric.metadata.collected_from_cases)
        if cases:
            return cases[0]
        if metric.execution_id:
            return str(metric.execution_id)[:8]
        return "case"

    async def on_no_tests_found(self) -> None:
        self.console.print("[yellow]No tests found.[/yellow]")

    async def on_collection_complete(self, items: list[MeritTestDefinition]) -> None:
        environment = RunEnvironment()
        runner = get_runner()
        run_id = runner.merit_run.run_id if runner and runner.merit_run else None
        self._print_run_header(environment, run_id)
        if self.verbosity >= 0:
            self.console.print(f"[bold]Collected {len(items)} tests[/bold]\n")

    async def on_test_complete(self, execution: TestExecution) -> None:
        result = execution.result
        item = execution.item

        if result.status in {TestStatus.FAILED, TestStatus.ERROR}:
            self._failures.append(execution)

        if self.verbosity < 0:
            return
        if self.verbosity == 0:
            self._print_compact_test(item, result)
            return

        self._print_verbose_test(execution)

    def _print_compact_test(self, item: MeritTestDefinition, result: TestResult) -> None:
        color = self._status_color(result.status)
        symbol = f"[{color}]{self._status_symbol(result.status)}[/{color}]"
        if self._current_module != item.module_path:
            if self._current_module is not None:
                self.console.print()
            module_path = self._safe_relative_path(item.module_path)
            self.console.print(f" • {module_path.as_posix()} ", end="")
            self._current_module = item.module_path
        self.console.print(symbol, end="")

    def _print_verbose_test(self, execution: TestExecution) -> None:
        result = execution.result
        item = execution.item

        if self._current_module != item.module_path:
            self._print_file_header(item.module_path)

        if execution.sub_executions:
            passed_count = sum(
                1 for e in execution.sub_executions if e.result.status == TestStatus.PASSED
            )
            total_count = len(execution.sub_executions)
            color = self._status_color(result.status)
            extra = f"[{color}]{passed_count}/{total_count} passed[/{color}] "
            self._print_test_line(item.full_name, result, extra=extra)
            self._print_sub_executions(execution.sub_executions, indent=4, marker="↳")
            return

        extra = self._get_status_extra(result)
        self._print_test_line(item.full_name, result, extra=extra)

    def _get_status_extra(self, result: TestResult) -> str:
        if result.status == TestStatus.SKIPPED:
            reason = result.error.args[0] if result.error else "skipped"
            return f"[dim]skipped ({reason})[/dim] "
        if result.status == TestStatus.XFAILED:
            reason = result.error.args[0] if result.error else "expected failure"
            return f"[dim]xfailed ({reason})[/dim] "
        if result.status == TestStatus.XPASSED:
            return "[dim]XPASS[/dim] "
        return ""

    def _print_sub_executions(
        self, sub_executions: list[TestExecution], indent: int, marker: str
    ) -> None:
        for sub in sub_executions:
            if sub.result.status in {TestStatus.PASSED, TestStatus.FAILED, TestStatus.ERROR}:
                self._print_sub_execution_line(sub, indent, marker)
            if sub.sub_executions:
                self._print_sub_executions(sub.sub_executions, indent + 2, marker)

    async def on_run_complete(self, merit_run: MeritRun) -> None:
        result = merit_run.result
        if self.verbosity == 0 and self._current_module is not None:
            self.console.print()

        if self.verbosity != 0 and self._failures:
            self._print_failures()

        if result.stopped_early:
            self.console.print("[yellow]Run terminated early.[/yellow]")

        self._print_metric_results(result.metric_results)
        self._print_summary(merit_run)

    def _print_failures(self) -> None:
        self.console.print()
        self._print_section_header("FAILURES")

        for index, failure in enumerate(self._failures):
            if index:
                self.console.print()

            item = failure.item
            failure_result = failure.result
            color = self._status_color(failure_result.status)

            sub_failures = [
                sub
                for sub in failure.sub_executions
                if sub.result.status.is_failure
            ]

            if sub_failures:
                nested_panels = []
                for sub in sub_failures:
                    sub_color = self._status_color(sub.result.status)
                    if sub.definition.id_suffix:
                        title = escape(f"[{sub.definition.id_suffix}]")
                    elif sub.execution_id:
                        title = escape(f"[{str(sub.execution_id)[:8]}]")
                    else:
                        title = self._status_label(sub.result.status)

                    lines = self._build_failure_lines(sub.result)
                    nested_panels.append(self._build_failure_panel(title, lines, sub_color))

                self.console.print(
                    Panel(
                        Group(*nested_panels),
                        title=item.full_name,
                        title_align="left",
                        border_style=color,
                        expand=True,
                        padding=(1, 1),
                    )
                )
            else:
                lines = self._build_failure_lines(failure_result)
                self.console.print(self._build_failure_panel(item.full_name, lines, color))

        self.console.print()

    def _print_summary(self, merit_run: MeritRun) -> None:
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
        summary_line = f"run_id: {merit_run.run_id}\n{summary} in {result.total_duration_ms:.0f}ms"
        self.console.print()
        self._print_section_header("SUMMARY")
        self.console.print(f"[bold]{summary_line}[/bold]", justify="center")
        self.console.print("=" * self.console.width)

    def _print_metric_results(self, metric_results: list[MetricResult]) -> None:
        if not metric_results:
            return

        self.console.print()
        self._print_section_header("METRICS")

        if self.verbosity == 0:
            self._print_session_metrics(metric_results)
            return

        if self.verbosity < 0:
            failed_count = sum(1 for m in metric_results if self._format_metric_value(m)[3])
            if failed_count:
                self.console.print(f"[yellow]{failed_count} failed — see DB for details[/yellow]")
                self.console.print()
            return

        self._print_verbose_metrics(metric_results)

    def _print_session_metrics(self, metric_results: list[MetricResult]) -> None:
        session_metrics = [m for m in metric_results if m.metadata.scope == Scope.SESSION]
        for metric in session_metrics:
            stats = self._format_metric_value(metric)
            self._print_metric_row(metric.name, stats)
        if session_metrics:
            self.console.print()

    def _print_verbose_metrics(self, metric_results: list[MetricResult]) -> None:
        case_metrics = [m for m in metric_results if m.metadata.scope == Scope.CASE]
        other_metrics = [m for m in metric_results if m.metadata.scope != Scope.CASE]

        for metric in other_metrics:
            stats = self._format_metric_value(metric)
            name = self._get_metric_display_name(metric)
            self._print_metric_row(name, stats)

        if case_metrics:
            self._print_case_metrics(case_metrics)

    def _get_metric_display_name(self, metric: MetricResult) -> str:
        name = metric.name
        if metric.metadata.scope == Scope.CASE and metric.metadata.collected_from_merits:
            merits = sorted(metric.metadata.collected_from_merits)
            case_suffix = ""
            if metric.metadata.collected_from_cases:
                cases = sorted(metric.metadata.collected_from_cases)
                case_suffix = escape(f"[{cases[0]}]")
            name = f"{name}::{merits[0]}{case_suffix}"
        return name

    def _print_case_metrics(self, case_metrics: list[MetricResult]) -> None:
        grouped = self._group_case_metrics(case_metrics)
        for metric_name in sorted(grouped):
            self.console.print(f" • {metric_name}")
            for metric in grouped[metric_name]:
                case_label = escape(f"[{self._get_case_label(metric)}]")
                stats = self._format_metric_value(metric)
                self._print_metric_row(f"↳ {case_label}", stats, indent=4)

    async def on_run_stopped_early(self, failure_count: int) -> None:
        self.console.print(f"\n\n[red]Stopping early after {failure_count} failure(s).[/red]")

    async def on_tracing_enabled(self, output_path: Path) -> None:
        if output_path.exists():
            self.console.print(
                f"[dim]Tracing written to {output_path} ({output_path.stat().st_size} bytes)[/dim]"
            )
