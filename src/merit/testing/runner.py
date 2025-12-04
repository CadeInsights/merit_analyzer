"""Test runner for executing discovered tests."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rich.console import Console

from merit.assertions import AssertionResult
from merit.testing.discovery import TestItem, collect
from merit.testing.resources import ResourceResolver, Scope, get_registry


class TestStatus(Enum):
    """Test execution status."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    XFAILED = "xfailed"
    XPASSED = "xpassed"


@dataclass
class TestResult:
    """Result of a single test execution."""

    item: TestItem
    status: TestStatus
    duration_ms: float
    error: Exception | None = None
    output: Any = None
    assertion_result: AssertionResult | None = None


@dataclass
class RunResult:
    """Result of a complete test run."""

    results: list[TestResult] = field(default_factory=list)
    total_duration_ms: float = 0
    stopped_early: bool = False

    @property
    def passed(self) -> int:
        """Count of passed tests."""
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)

    @property
    def failed(self) -> int:
        """Count of failed tests."""
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)

    @property
    def errors(self) -> int:
        """Count of errored tests."""
        return sum(1 for r in self.results if r.status == TestStatus.ERROR)

    @property
    def skipped(self) -> int:
        """Count of skipped tests."""
        return sum(1 for r in self.results if r.status == TestStatus.SKIPPED)

    @property
    def xfailed(self) -> int:
        """Count of expected failures."""
        return sum(1 for r in self.results if r.status == TestStatus.XFAILED)

    @property
    def xpassed(self) -> int:
        """Count of unexpected passes for xfail tests."""
        return sum(1 for r in self.results if r.status == TestStatus.XPASSED)

    @property
    def total(self) -> int:
        """Total test count."""
        return len(self.results)


class Runner:
    """Executes discovered tests with resource injection."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        maxfail: int | None = None,
        verbosity: int = 0,
    ) -> None:
        self.console = console or Console()
        self.maxfail = maxfail if maxfail and maxfail > 0 else None
        self.verbosity = verbosity

    async def run(self, items: list[TestItem] | None = None, path: str | None = None) -> RunResult:
        """Run tests and return results.

        Args:
            items: Pre-collected test items, or None to discover.
            path: Path to discover tests from if items not provided.
        """
        if items is None:
            items = collect(path)

        if not items:
            self.console.print("[yellow]No tests found.[/yellow]")
            return RunResult()

        self.console.print(f"[bold]Collected {len(items)} tests[/bold]\n")

        run_result = RunResult()
        start = time.perf_counter()

        resolver = ResourceResolver(get_registry())

        failures = 0
        stopped_early = False

        for item in items:
            result = await self._run_test(item, resolver)
            run_result.results.append(result)
            self._print_result(result)

            # Clear case-scoped resources after each test
            await resolver.teardown_scope(Scope.CASE)

            if result.status in {TestStatus.FAILED, TestStatus.ERROR}:
                failures += 1
                if self.maxfail and failures >= self.maxfail:
                    stopped_early = True
                    self.console.print(f"[red]Stopping early after {self.maxfail} failure(s).[/red]")
                    break

        # Teardown all resources
        await resolver.teardown()

        run_result.total_duration_ms = (time.perf_counter() - start) * 1000
        run_result.stopped_early = stopped_early
        self._print_summary(run_result)

        return run_result

    async def _run_test(self, item: TestItem, resolver: ResourceResolver) -> TestResult:
        """Execute a single test with resource injection."""
        start = time.perf_counter()

        if item.skip_reason:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                item=item,
                status=TestStatus.SKIPPED,
                duration_ms=duration,
                error=AssertionError(item.skip_reason),
            )

        expect_failure = item.xfail_reason is not None

        # Resolve resources for this test's parameters
        kwargs = {}
        if item.param_values:
            kwargs.update(item.param_values)

        for param in item.params:
            if param in kwargs:
                continue
            try:
                kwargs[param] = await resolver.resolve(param)
            except ValueError:
                # Unknown resource - might be a non-resource param
                pass

        # For class methods, instantiate the class
        instance = None
        if item.class_name:
            # Get the class from the function's globals
            cls = item.fn.__globals__.get(item.class_name)
            if cls:
                instance = cls()

        try:
            if instance:
                if item.is_async:
                    await item.fn(instance, **kwargs)
                else:
                    item.fn(instance, **kwargs)
            elif item.is_async:
                await item.fn(**kwargs)
            else:
                item.fn(**kwargs)

            duration = (time.perf_counter() - start) * 1000
            if expect_failure:
                if item.xfail_strict:
                    err = AssertionError(item.xfail_reason or "xfail marked test passed")
                    return TestResult(item=item, status=TestStatus.FAILED, duration_ms=duration, error=err)
                return TestResult(item=item, status=TestStatus.XPASSED, duration_ms=duration)
            return TestResult(item=item, status=TestStatus.PASSED, duration_ms=duration)

        except AssertionError as e:
            duration = (time.perf_counter() - start) * 1000
            assertion_result = getattr(e, "assertion_result", None)
            if expect_failure:
                err = AssertionError(item.xfail_reason) if item.xfail_reason else e
                return TestResult(
                    item=item,
                    status=TestStatus.XFAILED,
                    duration_ms=duration,
                    error=err,
                    assertion_result=assertion_result,
                )
            return TestResult(
                item=item, status=TestStatus.FAILED, duration_ms=duration, error=e, assertion_result=assertion_result
            )

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            if expect_failure:
                err = AssertionError(item.xfail_reason) if item.xfail_reason else e
                return TestResult(item=item, status=TestStatus.XFAILED, duration_ms=duration, error=err)
            return TestResult(item=item, status=TestStatus.ERROR, duration_ms=duration, error=e)

    def _print_result(self, result: TestResult) -> None:
        """Print a single test result."""
        if self.verbosity < 0 and result.status not in {TestStatus.FAILED, TestStatus.ERROR}:
            return

        if result.status == TestStatus.PASSED:
            self.console.print(f"  [green]✓[/green] {result.item.full_name} [dim]({result.duration_ms:.1f}ms)[/dim]")
        elif result.status == TestStatus.FAILED:
            self.console.print(f"  [red]✗[/red] {result.item.full_name} [dim]({result.duration_ms:.1f}ms)[/dim]")
            if result.assertion_result:
                ar = result.assertion_result
                self.console.print(f"    [red]{ar.assertion_name}: {ar.message}[/red]")
                if ar.score is not None:
                    self.console.print(f"    [dim]score={ar.score:.2f}, confidence={ar.confidence:.2f}[/dim]")
            elif result.error:
                self.console.print(f"    [red]{result.error}[/red]")
        elif result.status == TestStatus.ERROR:
            self.console.print(f"  [yellow]![/yellow] {result.item.full_name} [dim]({result.duration_ms:.1f}ms)[/dim]")
            if result.error:
                self.console.print(f"    [yellow]{type(result.error).__name__}: {result.error}[/yellow]")
        elif result.status == TestStatus.SKIPPED:
            reason = result.error.args[0] if result.error else "skipped"
            self.console.print(f"  [yellow]-[/yellow] {result.item.full_name} [dim]skipped ({reason})[/dim]")
        elif result.status == TestStatus.XFAILED:
            reason = result.error.args[0] if result.error else "expected failure"
            self.console.print(f"  [blue]x[/blue] {result.item.full_name} [dim]xfailed ({reason})[/dim]")
        elif result.status == TestStatus.XPASSED:
            self.console.print(f"  [magenta]![/magenta] {result.item.full_name} [dim]XPASS[/dim]")

    def _print_summary(self, run_result: RunResult) -> None:
        """Print test run summary."""
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


def run(path: str | None = None) -> RunResult:
    """Run tests synchronously (convenience wrapper).

    Args:
        path: Path to discover tests from.

    Returns:
        RunResult with all test outcomes.
    """
    return asyncio.run(Runner().run(path=path))
