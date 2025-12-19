"""Test runner for executing discovered tests."""

import asyncio
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from pathlib import Path
from uuid import UUID, uuid4

from opentelemetry.trace import StatusCode
from rich.console import Console

from merit.predicates import PredicateResult, close_predicate_api_client
from merit.testing.discovery import TestItem, collect
from merit.testing.resources import ResourceResolver, Scope, get_registry
from merit.tracing import clear_traces, get_tracer, init_tracing
from merit.version import __version__


@dataclass
class RunEnvironment:
    """Metadata about the environment where tests were executed."""

    run_id: UUID = field(default_factory=uuid4)
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None

    # Git info
    commit_hash: str | None = None
    branch: str | None = None
    dirty: bool | None = None

    # System info
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    hostname: str = field(default_factory=socket.gethostname)
    working_directory: str = field(default_factory=os.getcwd)
    merit_version: str = __version__

    # Environment variables (filtered)
    env_vars: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "run_id": str(self.run_id),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "commit_hash": self.commit_hash,
            "branch": self.branch,
            "dirty": self.dirty,
            "python_version": self.python_version,
            "platform": self.platform,
            "hostname": self.hostname,
            "working_directory": self.working_directory,
            "merit_version": self.merit_version,
            "env_vars": self.env_vars,
        }


def _get_git_info() -> tuple[str | None, str | None, bool | None]:
    """Capture git metadata if available."""
    try:
        # Check if inside git repo
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            check=True,
            capture_output=True,
            timeout=1,
        )

        # Get hash
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=1,
        ).stdout.strip()

        # Get branch
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=1,
        ).stdout.strip()

        # Check dirty status
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=1,
        ).stdout.strip()
        dirty = bool(status)

        return commit, branch, dirty

    except (subprocess.SubprocessError, FileNotFoundError):
        return None, None, None


def _filter_env_vars() -> dict[str, str]:
    """Capture and mask relevant environment variables."""
    # TODO: detect ci_provider from env vars

    allowlist = {
        "MODEL_VENDOR",
        "INFERENCE_VENDOR",
        "CLOUD_ML_REGION",
        "GOOGLE_CLOUD_PROJECT",
        "AWS_REGION",
    }

    captured = {}
    for key, value in os.environ.items():
        if key in allowlist:
            captured[key] = value

    # Explicitly check for known keys to capture masked versions
    sensitive_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ]

    for key in sensitive_keys:
        if key in os.environ:
            val = os.environ[key]
            if len(val) > 4:
                captured[key] = f"***{val[-4:]}"
            else:
                captured[key] = "***"

    return captured


def capture_environment() -> RunEnvironment:
    """Capture current environment metadata."""
    # TODO: capture frozen package versions for full reproducibility

    commit, branch, dirty = _get_git_info()

    return RunEnvironment(
        commit_hash=commit,
        branch=branch,
        dirty=dirty,
        env_vars=_filter_env_vars(),
    )


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
    predicate_results: list[PredicateResult] = field(default_factory=list) # TODO: implement collecting and printing predicate results


@dataclass
class RunResult:
    """Result of a complete test run."""

    results: list[TestResult] = field(default_factory=list)
    total_duration_ms: float = 0
    stopped_early: bool = False
    environment: RunEnvironment | None = None

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
    """Executes discovered tests with resource injection.

    Examples:
        # Sequential execution (default)
        runner = Runner()
        result = await runner.run(path="tests/")

        # Concurrent execution with 5 workers
        runner = Runner(concurrency=5)
        result = await runner.run(path="tests/")

        # Unlimited concurrency (capped at 10)
        runner = Runner(concurrency=0)
        result = await runner.run(path="tests/")
    """

    DEFAULT_MAX_CONCURRENCY = 10

    def __init__(
        self,
        console: Console | None = None,
        *,
        maxfail: int | None = None,
        verbosity: int = 0,
        concurrency: int = 1,
        timeout: float | None = None,
        enable_tracing: bool = False,
        trace_output: Path | str | None = None,
    ) -> None:
        self.console = console or Console()
        self.maxfail = maxfail if maxfail and maxfail > 0 else None
        self.verbosity = verbosity
        self.timeout = timeout  # Per-test timeout in seconds
        # 0 = unlimited (capped at DEFAULT_MAX_CONCURRENCY), 1 = sequential, >1 = concurrent
        self.concurrency = concurrency if concurrency > 0 else self.DEFAULT_MAX_CONCURRENCY
        self.enable_tracing = enable_tracing
        self.trace_output = Path(trace_output) if trace_output else Path("traces.jsonl")

    async def run(self, items: list[TestItem] | None = None, path: str | None = None) -> RunResult:
        """Run tests and return results.

        Args:
            items: Pre-collected test items, or None to discover.
            path: Path to discover tests from if items not provided.
        """
        run_result = RunResult()
        run_result.environment = capture_environment()

        # Initialize tracing if enabled
        if self.enable_tracing:
            init_tracing(output_path=self.trace_output)
            clear_traces()

        if items is None:
            items = collect(path)

        if not items:
            self.console.print("[yellow]No tests found.[/yellow]")
            if run_result.environment:
                run_result.environment.end_time = datetime.now(UTC)
            return run_result

        self.console.print(f"[bold]Collected {len(items)} tests[/bold]\n")

        start = time.perf_counter()

        resolver = ResourceResolver(get_registry())

        if self.concurrency == 1:
            await self._run_sequential(items, resolver, run_result)
        else:
            await self._run_concurrent(items, resolver, run_result)

        # Teardown all resources
        await resolver.teardown()
        await close_predicate_api_client()

        run_result.total_duration_ms = (time.perf_counter() - start) * 1000
        if run_result.environment:
            run_result.environment.end_time = datetime.now(UTC)
        self._print_summary(run_result)

        # Tracing is streamed; surface the path for the user
        if self.enable_tracing and self.trace_output.exists():
            self.console.print(
                f"[dim]Tracing written to {self.trace_output} ({self.trace_output.stat().st_size} bytes)[/dim]"
            )

        return run_result

    async def _run_sequential(self, items: list[TestItem], resolver: ResourceResolver, run_result: RunResult) -> None:
        """Run tests sequentially."""
        failures = 0
        for item in items:
            result = await self._run_test(item, resolver)
            run_result.results.append(result)
            self._print_result(result)
            await resolver.teardown_scope(Scope.CASE)

            if result.status in {TestStatus.FAILED, TestStatus.ERROR}:
                failures += 1
                if self.maxfail and failures >= self.maxfail:
                    run_result.stopped_early = True
                    self.console.print(f"[red]Stopping early after {self.maxfail} failure(s).[/red]")
                    break

    async def _run_concurrent(self, items: list[TestItem], resolver: ResourceResolver, run_result: RunResult) -> None:
        """Run tests concurrently with semaphore control."""
        semaphore = asyncio.Semaphore(self.concurrency)
        lock = asyncio.Lock()
        failures = 0
        stop_flag = False

        async def run_one(idx: int, item: TestItem) -> tuple[int, TestResult | None]:
            nonlocal failures, stop_flag
            if stop_flag:
                return (idx, None)
            async with semaphore:
                if stop_flag:
                    return (idx, None)
                child_resolver = resolver.fork_for_case()
                start = time.perf_counter()
                result: TestResult
                try:
                    if self.timeout:
                        result = await asyncio.wait_for(
                            self._run_test(item, child_resolver),
                            timeout=self.timeout,
                        )
                    else:
                        result = await self._run_test(item, child_resolver)
                except TimeoutError:
                    duration = (time.perf_counter() - start) * 1000
                    result = TestResult(
                        item=item,
                        status=TestStatus.ERROR,
                        duration_ms=duration,
                        error=TimeoutError(f"Test timed out after {self.timeout}s"),
                    )
                except Exception as e:
                    duration = (time.perf_counter() - start) * 1000
                    result = TestResult(item=item, status=TestStatus.ERROR, duration_ms=duration, error=e)
                finally:
                    await child_resolver.teardown_scope(Scope.CASE)

                if result.status in {TestStatus.FAILED, TestStatus.ERROR}:
                    async with lock:
                        failures += 1
                        if self.maxfail and failures >= self.maxfail:
                            stop_flag = True
                            run_result.stopped_early = True
                return (idx, result)

        indexed_results = await asyncio.gather(
            *[run_one(i, item) for i, item in enumerate(items)], return_exceptions=True
        )

        # Sort by original index for deterministic output order
        sorted_results = sorted(
            (r for r in indexed_results if isinstance(r, tuple) and r[1] is not None),
            key=lambda x: x[0],
        )

        for _, result in sorted_results:
            run_result.results.append(result)
            self._print_result(result)

        if run_result.stopped_early:
            self.console.print(f"[red]Stopping early after {self.maxfail} failure(s).[/red]")

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

        # Execute test, optionally wrapped in a trace span
        return await self._execute_test_body(item, instance, kwargs, start, expect_failure)

    async def _execute_test_body(
        self,
        item: TestItem,
        instance: Any,
        kwargs: dict[str, Any],
        start: float,
        expect_failure: bool,
    ) -> TestResult:
        """Execute the test body, optionally wrapped in a trace span."""
        tracer = get_tracer() if self.enable_tracing else None
        span_context = tracer.start_as_current_span(f"test.{item.full_name}") if tracer else None

        try:
            if span_context:
                span = span_context.__enter__()
                span.set_attribute("test.name", item.name)
                span.set_attribute("test.module", str(item.module_path))
                if item.tags:
                    span.set_attribute("test.tags", list(item.tags))

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

            if span_context:
                span.set_attribute("test.status", "passed")
                span.set_attribute("test.duration_ms", duration)

            if expect_failure:
                if item.xfail_strict:
                    err = AssertionError(item.xfail_reason or "xfail marked test passed")
                    return TestResult(item=item, status=TestStatus.FAILED, duration_ms=duration, error=err)
                return TestResult(item=item, status=TestStatus.XPASSED, duration_ms=duration)
            return TestResult(item=item, status=TestStatus.PASSED, duration_ms=duration)

        except AssertionError as e:
            duration = (time.perf_counter() - start) * 1000

            if span_context:
                span.set_attribute("test.status", "failed")
                span.set_attribute("test.duration_ms", duration)
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)

            if expect_failure:
                err = AssertionError(item.xfail_reason) if item.xfail_reason else e
                return TestResult(
                    item=item,
                    status=TestStatus.XFAILED,
                    duration_ms=duration,
                    error=err
                )
            return TestResult(
                item=item, status=TestStatus.FAILED, duration_ms=duration, error=e
            )

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000

            if span_context:
                span.set_attribute("test.status", "error")
                span.set_attribute("test.duration_ms", duration)
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)

            if expect_failure:
                err = AssertionError(item.xfail_reason) if item.xfail_reason else e
                return TestResult(item=item, status=TestStatus.XFAILED, duration_ms=duration, error=err)
            return TestResult(item=item, status=TestStatus.ERROR, duration_ms=duration, error=e)

        finally:
            if span_context:
                span_context.__exit__(None, None, None)

    def _print_result(self, result: TestResult) -> None:
        """Print a single test result."""
        if self.verbosity < 0 and result.status not in {TestStatus.FAILED, TestStatus.ERROR}:
            return

        if result.status == TestStatus.PASSED:
            self.console.print(f"  [green]✓[/green] {result.item.full_name} [dim]({result.duration_ms:.1f}ms)[/dim]")
        elif result.status == TestStatus.FAILED:
            self.console.print(f"  [red]✗[/red] {result.item.full_name} [dim]({result.duration_ms:.1f}ms)[/dim]")
            if result.error:
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
